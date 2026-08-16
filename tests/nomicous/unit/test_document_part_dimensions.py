"""Page dimensions are persisted at upload and backfilled lazily for legacy parts.

Covers the DocumentPart.width/height write path (previously written only by a dead
repository method) and the bounded, single decode of untrusted upload bytes.
"""

from __future__ import annotations

import struct
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image

from backend.core.exceptions import ValidationError
from backend.document.application.document_access import DocumentContext
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.media_store.encoding import (
    encode_part_image_with_size,
    read_image_size,
    render_part_thumbnail,
)
from backend.document.infrastructure.orm_models import Document, DocumentPart

PILLOW_DEFAULT_MAX_PIXELS = 89_478_485


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_header_claiming(width: int, height: int) -> bytes:
    """A complete, openable PNG whose IHDR claims a raster it does not carry.

    Built by rewriting the header of a real 1x1 file (and its CRC) rather than by
    emitting a bare IHDR: a truncated file fails in the PNG parser before any size check
    runs, which would let a decode-bound test pass for the wrong reason.
    """
    source = _png_bytes(1, 1)
    ihdr = struct.pack(">II", width, height) + source[24:29]
    chunk = b"IHDR" + ihdr
    return (
        source[:8]
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
        + source[33:]
    )


class _RecordingSession:
    def __init__(self) -> None:
        self.commits = 0
        self.added: list[object] = []

    def add(self, item: object) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _item: object) -> None:
        pass

    async def rollback(self) -> None:
        pass


class _Repository:
    async def next_part_order(self, _session, _document_id) -> int:
        return 0


class _StubAccess:
    """Stands in for the authorization seam; these tests are about pixels, not permissions."""

    def __init__(self, document: Document | None = None) -> None:
        self._document = document

    async def require_document(self, *_args, **_kwargs) -> DocumentContext:
        return DocumentContext(project=object(), document=self._document)


class _Store:
    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self.blobs = blobs or {}
        self.reads: list[str] = []

    def part_image_key(self, part_id, **_kwargs) -> str:
        return f"parts/{part_id}.webp"

    def write(self, image_key: str, data: bytes) -> None:
        self.blobs[image_key] = data

    def read(self, image_key: str) -> bytes:
        self.reads.append(image_key)
        return self.blobs[image_key]

    def delete(self, image_key: str) -> None:
        self.blobs.pop(image_key, None)

    def signed_object_url(self, image_key, *, expires_at):
        return f"/media/signed/{image_key}"

    def create_upload_url(self, image_key, *, expires_at):
        raise ValueError("cannot presign")


# --- Upload persists dimensions ---


@pytest.mark.asyncio
async def test_upload_part_persists_source_dimensions() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    service = DocumentPartService(
        documents=_Repository(), media=_Store(), access=_StubAccess(document)
    )
    session = _RecordingSession()

    part = await service.upload_part(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        data=_png_bytes(37, 19),
        filename="folio.png",
    )

    assert (part.width, part.height) == (37, 19)


# --- Lazy backfill for rows uploaded before dimensions were persisted ---


@pytest.mark.asyncio
async def test_backfill_recovers_dimensions_from_stored_image() -> None:
    store = _Store()
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="legacy.webp")
    store.blobs["legacy.webp"] = encode_part_image_with_size(_png_bytes(12, 9)).data
    service = DocumentPartService(documents=_Repository(), media=store)
    session = _RecordingSession()

    await service.backfill_part_dimensions(session, [part])

    assert (part.width, part.height) == (12, 9)
    assert session.commits == 1


@pytest.mark.asyncio
async def test_backfill_skips_parts_that_already_have_dimensions() -> None:
    store = _Store()
    part = DocumentPart(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        order=0,
        image_key="present.webp",
        width=100,
        height=50,
    )
    service = DocumentPartService(documents=_Repository(), media=store)
    session = _RecordingSession()

    await service.backfill_part_dimensions(session, [part])

    assert store.reads == []
    assert session.commits == 0


@pytest.mark.asyncio
async def test_backfill_tolerates_missing_blob() -> None:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="gone.webp")
    service = DocumentPartService(documents=_Repository(), media=_Store())
    session = _RecordingSession()

    await service.backfill_part_dimensions(session, [part])

    assert part.width is None
    assert session.commits == 0


# --- Bounded, single decode ---


def test_bound_is_enforced_by_this_module_not_by_pillows_global(monkeypatch) -> None:
    """With Pillow's own guard disabled, ``MAX_DECODE_PIXELS`` is still a hard ceiling."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    oversized = _png_header_claiming(15_000, 15_000)

    with pytest.raises(ValidationError):
        read_image_size(oversized)

    # Just under the ceiling still opens, so the rejection is the bound and not a
    # blanket refusal of large headers.
    assert read_image_size(_png_header_claiming(10_000, 10_000)) == (10_000, 10_000)


def test_concurrent_decodes_do_not_interfere() -> None:
    """An oversized decode on one thread must not let one through on another.

    The previous implementation raised ``Image.MAX_IMAGE_PIXELS`` for the duration of a
    decode, so an interleaved decode inherited the relaxed bound and a restore could
    clobber a concurrent value.
    """
    oversized = _png_header_claiming(15_000, 15_000)
    valid = _png_bytes(6, 3)
    observed_limits: list[object] = []

    def decode(index: int) -> tuple[int, int] | str:
        observed_limits.append(Image.MAX_IMAGE_PIXELS)
        try:
            return read_image_size(oversized if index % 2 else valid)
        except ValidationError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(decode, range(64)))

    assert [result for index, result in enumerate(results) if index % 2] == ["rejected"] * 32
    assert [result for index, result in enumerate(results) if not index % 2] == [(6, 3)] * 32
    assert set(observed_limits) == {PILLOW_DEFAULT_MAX_PIXELS}


def test_encoder_failure_is_not_reported_as_invalid_input(monkeypatch) -> None:
    """A WebP encoder fault is ours, so it must not surface as a 422 about the upload."""

    def boom(self, fp, format=None, **params):  # noqa: A002 - mirrors Image.save
        raise OSError("encoder exploded")

    monkeypatch.setattr(Image.Image, "save", boom)

    with pytest.raises(OSError, match="encoder exploded"):
        encode_part_image_with_size(_png_bytes(4, 4))

    with pytest.raises(OSError, match="encoder exploded"):
        render_part_thumbnail(_png_bytes(8, 8), 4)


def test_decode_rejects_non_image_bytes() -> None:
    with pytest.raises(ValidationError):
        encode_part_image_with_size(b"not an image at all")


def test_encode_reports_dimensions_of_the_source() -> None:
    encoded = encode_part_image_with_size(_png_bytes(23, 7))

    assert (encoded.width, encoded.height) == (23, 7)
    assert read_image_size(encoded.data) == (23, 7)
