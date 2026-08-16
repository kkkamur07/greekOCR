"""Direct-to-storage (presigned) upload flow — service and store unit tests.

The presigned path exists to bypass Vercel's 4.5 MB function-body cap: the API
mints a URL, the browser PUTs bytes straight to object storage, then the API
finalizes the part. These tests pin three things that must not regress:

1. ``create_upload_url`` on the local backend raises — a filesystem cannot presign.
2. ``create_upload_url`` on Supabase delegates to the storage client and returns
   the signed URL plus token verbatim, and refuses when either is missing.
3. ``DocumentPartService.begin_upload``/``finalize_upload`` create the ``pending``
   row, fall back cleanly when the store cannot presign, refuse to finalize twice,
   refuse a key that names any other part's object, discard blobs they reject, and
   re-derive dimensions from the stored blob rather than trusting the client.

No Postgres and no network: the media store and repository are fakes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.core.exceptions import ValidationError
from backend.document.application import part_service as part_service_module
from backend.document.application.document_access import DocumentContext, PartContext
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.media_store import (
    LocalMediaStore,
    PresignUnsupported,
    SupabaseMediaStore,
)
from backend.document.infrastructure.media_store.encoding import encode_part_image_with_size
from backend.document.infrastructure.media_store.keys import part_image_key
from backend.document.infrastructure.orm_models import Document, DocumentPart


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --- create_upload_url: backend capability ---


def test_local_store_cannot_presign(tmp_path) -> None:
    store = LocalMediaStore(root=tmp_path)
    with pytest.raises(PresignUnsupported, match="cannot presign"):
        store.create_upload_url(
            part_image_key(uuid.uuid4()),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )


def test_supabase_store_presigns_and_returns_url_and_token() -> None:
    signed_result = {
        "signedURL": "https://storage.example/object/upload/sign/bucket/path.webp?token=abc",
        "signedUrl": "https://storage.example/object/upload/sign/bucket/path.webp?token=abc",
        "token": "abc",
        "path": "path.webp",
    }
    calls: list[tuple[str, object]] = []

    class _FakeBucket:
        def create_signed_upload_url(self, path, options=None):
            calls.append((path, options))
            return signed_result

    class _FakeStorage:
        def from_(self, bucket):
            return _FakeBucket()

    store = SupabaseMediaStore(client=SimpleNamespace(storage=_FakeStorage()))
    key = part_image_key(uuid.uuid4())
    url, token = store.create_upload_url(key, expires_at=datetime.now(UTC) + timedelta(minutes=5))

    assert url == signed_result["signedUrl"]
    assert token == "abc"
    assert calls == [(key, None)]


def test_supabase_store_refuses_missing_url_or_token() -> None:
    class _FakeBucket:
        def create_signed_upload_url(self, path, options=None):
            return {"token": "abc"}  # no signedUrl

    class _FakeStorage:
        def from_(self, bucket):
            return _FakeBucket()

    store = SupabaseMediaStore(client=SimpleNamespace(storage=_FakeStorage()))
    with pytest.raises(RuntimeError, match="no signed upload URL"):
        store.create_upload_url(
            part_image_key(uuid.uuid4()),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )


# --- begin_upload / finalize_upload service logic ---


class _Repository:
    async def next_part_order(self, _session, _document_id) -> int:
        return 0


class _StubAccess:
    def __init__(self, document: Document) -> None:
        self._document = document
        self._part = DocumentPart(
            id=uuid.uuid4(), document_id=document.id, order=0, image_key="pending"
        )

    async def require_document(self, *_args, **_kwargs) -> DocumentContext:
        return DocumentContext(project=object(), document=self._document)

    async def require_part(self, *args, **_kwargs) -> PartContext:
        # Faithful to the real access object: resolve the part the caller named. The
        # earlier stub returned a fresh row of its own, which structurally hid the
        # key-to-part binding check finalize performs.
        session, _user, _project_id, _document_id, part_id = args
        for item in getattr(session, "added", []):
            if isinstance(item, DocumentPart) and item.id == part_id:
                return PartContext(project=object(), document=self._document, part=item)
        return PartContext(project=object(), document=self._document, part=self._part)


class _Store:
    def __init__(self, *, presign_url: str | None = "https://presigned.example/x") -> None:
        self.presign_url = presign_url
        self.presign_calls: list[str] = []
        self.reads: list[str] = []
        self.blobs: dict[str, bytes] = {}

    def part_image_key(self, part_id, **kwargs) -> str:
        return part_image_key(part_id)

    def create_upload_url(self, image_key, *, expires_at):
        self.presign_calls.append(image_key)
        if self.presign_url is None:
            raise PresignUnsupported("cannot presign")
        return self.presign_url, "tok"

    def read(self, image_key: str) -> bytes:
        self.reads.append(image_key)
        if image_key not in self.blobs:
            raise FileNotFoundError(image_key)
        return self.blobs[image_key]

    def write(self, image_key: str, data: bytes) -> None:
        self.blobs[image_key] = data

    def delete(self, image_key: str) -> None:
        self.blobs.pop(image_key, None)

    def signed_object_url(self, image_key, *, expires_at):
        return f"/media/signed/{image_key}"


class _RecordingSession:
    def __init__(self) -> None:
        self.commits = 0
        self.added: list[object] = []

    def add(self, item: object) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        # SQLAlchemy assigns the UUID primary-key default on flush; the fake mirrors
        # that so ``part.id`` is set before it is used to build the object key.
        for item in self.added:
            if isinstance(item, DocumentPart) and item.id is None:
                item.id = uuid.uuid4()

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _item: object) -> None:
        pass

    async def rollback(self) -> None:
        pass


def _make_service(store: _Store, document: Document) -> DocumentPartService:
    return DocumentPartService(
        documents=_Repository(),
        media=store,
        access=_StubAccess(document),
    )


async def test_begin_upload_creates_pending_row_and_mints_url() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, image_key, url, token = begin

    assert part.image_key == "pending"
    assert part.width is None
    assert part.height is None
    assert image_key == f"parts/{part.id}.webp"
    assert store.presign_calls == [image_key]
    assert url == "https://presigned.example/x"
    assert token == "tok"
    assert session.commits >= 1


async def test_begin_upload_signals_use_multipart_when_store_cannot_presign() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store(presign_url=None)
    service = _make_service(store, document)
    session = _RecordingSession()

    result = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )

    # A filesystem cannot presign, so the part is None (rolled back, no orphaned
    # ``pending`` row) and both URL and token are absent. The caller is expected
    # to fall back to the multipart upload.
    assert result is not None
    part, image_key, url, token = result
    assert part is None
    assert image_key.startswith("parts/")
    assert url is None
    assert token is None
    assert store.presign_calls != []  # the store was asked before it raised
    assert session.commits == 0


async def test_finalize_upload_verifies_blob_and_persists_dimensions() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, image_key, _url, _token = begin

    # The browser has PUT a real WebP to storage; the server must re-derive its size.
    store.blobs[image_key] = encode_part_image_with_size(_png_bytes(51, 13)).data

    finalized = await service.finalize_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        part_id=part.id,
        image_key=image_key,
        width=999,  # liar's dimensions; the stored blob wins
        height=999,
    )

    assert finalized.image_key == image_key
    assert (finalized.width, finalized.height) == (51, 13)


async def test_finalize_upload_rejects_non_image_blob() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, image_key, _url, _token = begin
    store.blobs[image_key] = b"not an image"

    with pytest.raises(ValidationError, match="not a valid image"):
        await service.finalize_upload(
            session,
            user=object(),
            project_id=uuid.uuid4(),
            document_id=document.id,
            part_id=part.id,
            image_key=image_key,
        )

    # A rejected blob must not linger in storage, referenced by nothing.
    assert image_key not in store.blobs


async def test_finalize_upload_rejects_a_foreign_image_key() -> None:
    """A part may only seal a key its own begin could have minted.

    Accepting an arbitrary client-supplied key aliases two rows onto one blob, and
    deleting either part then destroys the other document's image via the shared
    blob's deletion intent.
    """
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, _image_key, _url, _token = begin

    foreign_key = part_image_key(uuid.uuid4())
    store.blobs[foreign_key] = encode_part_image_with_size(_png_bytes(9, 9)).data

    with pytest.raises(ValidationError, match="does not belong"):
        await service.finalize_upload(
            session,
            user=object(),
            project_id=uuid.uuid4(),
            document_id=document.id,
            part_id=part.id,
            image_key=foreign_key,
        )

    # The refusal must come before any read or compensation: the foreign blob is
    # someone else's image and must survive untouched.
    assert store.reads == []
    assert foreign_key in store.blobs
    assert part.image_key == "pending"


async def test_finalize_upload_rejects_and_discards_an_oversized_blob(monkeypatch) -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, image_key, _url, _token = begin

    # The presigned PUT bypasses the API, so the declared size in begin binds nothing;
    # finalize re-checks the stored bytes. A tiny patched cap keeps the test cheap.
    monkeypatch.setattr(part_service_module, "MAX_PART_UPLOAD_BYTES", 64)
    store.blobs[image_key] = b"\x00" * 65

    with pytest.raises(ValidationError, match="maximum allowed size"):
        await service.finalize_upload(
            session,
            user=object(),
            project_id=uuid.uuid4(),
            document_id=document.id,
            part_id=part.id,
            image_key=image_key,
        )

    assert image_key not in store.blobs
    assert part.image_key == "pending"


async def test_finalize_upload_rejects_double_finalize() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    store = _Store()
    service = _make_service(store, document)
    session = _RecordingSession()

    begin = await service.begin_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        filename="page.png",
    )
    assert begin is not None
    part, image_key, _url, _token = begin
    store.blobs[image_key] = encode_part_image_with_size(_png_bytes(7, 5)).data

    await service.finalize_upload(
        session,
        user=object(),
        project_id=uuid.uuid4(),
        document_id=document.id,
        part_id=part.id,
        image_key=image_key,
    )

    # The part row now points at the real key; a second finalize must refuse.
    with pytest.raises(ValidationError, match="already been finalized"):
        await service.finalize_upload(
            session,
            user=object(),
            project_id=uuid.uuid4(),
            document_id=document.id,
            part_id=part.id,
            image_key=image_key,
        )
