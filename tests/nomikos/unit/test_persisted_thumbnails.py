"""Thumbnails are rendered once and kept in the bucket next to their original.

Rendering a thumbnail needs the whole page scan, so before this every thumbnail
request on a process that had not seen the page pulled megabytes out of object
storage to answer with kilobytes. The persisted rendering is read *instead of*
the original; that substitution is what these tests pin down, along with the
deletion that keeps the bucket from accumulating renderings of deleted pages.
"""

from __future__ import annotations

from io import BytesIO
from uuid import UUID, uuid4

import pytest
from PIL import Image

from backend.core.exceptions import NotFoundError
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.media_gc import delete_media_object
from backend.document.infrastructure.media_store import (
    PERSISTED_THUMBNAIL_WIDTHS,
    THUMBNAIL_ENCODER_VERSION,
    clear_thumbnail_cache,
    derived_image_key,
    persisted_thumbnail_key,
    persisted_thumbnail_keys,
    validate_image_key,
)
from backend.document.infrastructure.media_store.keys import part_image_key

PART_ID = UUID("6f1c9a2e-3b4d-4c5e-8f60-71a2b3c4d5e6")


class _FakeStore:
    """A dict-backed media store that counts every read, like a metered bucket."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.deletes: list[str] = []
        self.fail_writes = False
        self.fail_deletes: set[str] = set()

    def read(self, image_key: str) -> bytes:
        self.reads.append(image_key)
        if image_key not in self.objects:
            raise FileNotFoundError(image_key)
        return self.objects[image_key]

    def write(self, image_key: str, data: bytes) -> None:
        if self.fail_writes:
            raise RuntimeError("bucket unavailable")
        self.writes.append(image_key)
        self.objects[image_key] = data

    def delete(self, image_key: str) -> None:
        if image_key in self.fail_deletes:
            raise RuntimeError("bucket unavailable")
        self.deletes.append(image_key)
        self.objects.pop(image_key, None)


class _Part:
    def __init__(self, image_key: str) -> None:
        self.id = uuid4()
        self.image_key = image_key


def _page_bytes(width: int = 1200, height: int = 800) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (200, 180, 150)).save(buffer, format="WEBP", lossless=True)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _fresh_render_cache():
    clear_thumbnail_cache()
    yield
    clear_thumbnail_cache()


@pytest.mark.parametrize(
    ("image_key", "expected"),
    [
        (
            f"parts/{PART_ID}.webp",
            f"parts/{PART_ID}/thumb-w200-webp-q85-v1.webp",
        ),
        (
            f"parts/{PART_ID}/folio_12r.png",
            f"parts/{PART_ID}/folio_12r-thumb-w200-webp-q85-v1.webp",
        ),
    ],
)
def test_derived_keys_live_in_the_parts_folder_and_pass_validation(image_key, expected):
    derived = derived_image_key(image_key, width=200, encoder_version="webp-q85-v1")
    assert derived == expected
    validate_image_key(derived)


def test_derived_keys_differ_by_width_and_encoder_version():
    image_key = part_image_key(PART_ID, filename_stem="page")
    keys = {
        derived_image_key(image_key, width=width, encoder_version=version)
        for width in (200, 400)
        for version in ("webp-q85-v1", "webp-q85-v2")
    }
    assert len(keys) == 4


def test_a_long_filename_stem_still_yields_a_valid_key():
    image_key = part_image_key(PART_ID, filename_stem="x" * 128)
    validate_image_key(derived_image_key(image_key, width=800, encoder_version="webp-q85-v1"))


def test_only_the_closed_width_set_is_persisted():
    image_key = part_image_key(PART_ID)
    assert persisted_thumbnail_key(image_key, 1600) is None
    for width in PERSISTED_THUMBNAIL_WIDTHS:
        assert persisted_thumbnail_key(image_key, width) == derived_image_key(
            image_key, width=width, encoder_version=THUMBNAIL_ENCODER_VERSION
        )
    assert persisted_thumbnail_keys(image_key) == [
        persisted_thumbnail_key(image_key, width) for width in sorted(PERSISTED_THUMBNAIL_WIDTHS)
    ]


def test_first_thumbnail_read_renders_from_the_original_and_persists_it():
    image_key = part_image_key(PART_ID)
    store = _FakeStore({image_key: _page_bytes()})
    service = DocumentPartService(media=store)
    part = _Part(image_key)

    encoded = service._read_part_bytes(part, 200)

    derived = persisted_thumbnail_key(image_key, 200)
    assert store.reads == [derived, image_key], "derived first, original only on the miss"
    assert store.writes == [derived]
    assert store.objects[derived] == encoded
    with Image.open(BytesIO(encoded)) as thumb:
        assert thumb.width == 200


def test_later_thumbnail_reads_never_touch_the_original():
    image_key = part_image_key(PART_ID)
    store = _FakeStore({image_key: _page_bytes()})
    service = DocumentPartService(media=store)
    part = _Part(image_key)
    first = service._read_part_bytes(part, 200)
    store.reads.clear()
    clear_thumbnail_cache()  # a different process: the in-memory render cache is cold

    second = service._read_part_bytes(part, 200)

    assert second == first
    assert store.reads == [persisted_thumbnail_key(image_key, 200)]
    assert image_key not in store.reads


def test_widths_outside_the_set_are_rendered_but_not_persisted():
    image_key = part_image_key(PART_ID)
    store = _FakeStore({image_key: _page_bytes()})
    service = DocumentPartService(media=store)

    encoded = service._read_part_bytes(_Part(image_key), 1600)

    assert store.reads == [image_key]
    assert store.writes == []
    with Image.open(BytesIO(encoded)) as thumb:
        assert thumb.width == 1200, "a thumbnail is never upscaled"


def test_full_size_reads_are_unchanged():
    image_key = part_image_key(PART_ID)
    original = _page_bytes()
    store = _FakeStore({image_key: original})
    service = DocumentPartService(media=store)

    assert service._read_part_bytes(_Part(image_key), None) == original
    assert store.reads == [image_key]
    assert store.writes == []


def test_a_failed_persist_still_answers_the_request():
    image_key = part_image_key(PART_ID)
    store = _FakeStore({image_key: _page_bytes()})
    store.fail_writes = True
    service = DocumentPartService(media=store)

    encoded = service._read_part_bytes(_Part(image_key), 200)

    with Image.open(BytesIO(encoded)) as thumb:
        assert thumb.width == 200
    assert store.writes == []


def test_a_missing_original_is_still_not_found():
    image_key = part_image_key(PART_ID)
    service = DocumentPartService(media=_FakeStore())
    with pytest.raises(NotFoundError):
        service._read_part_bytes(_Part(image_key), 200)


def test_deleting_a_page_image_sweeps_its_persisted_thumbnails():
    image_key = part_image_key(PART_ID, filename_stem="page")
    derived = persisted_thumbnail_keys(image_key)
    store = _FakeStore({image_key: b"page", **{key: b"thumb" for key in derived}})

    delete_media_object(store, image_key)

    assert store.deletes == [image_key, *derived]
    assert store.objects == {}


def test_a_thumbnail_that_will_not_delete_does_not_fail_the_intent():
    image_key = part_image_key(PART_ID)
    derived = persisted_thumbnail_keys(image_key)
    store = _FakeStore({image_key: b"page", derived[0]: b"thumb"})
    store.fail_deletes = {derived[0]}

    delete_media_object(store, image_key)  # does not raise

    assert image_key not in store.objects
    assert store.deletes == [image_key, *derived[1:]]


def test_the_original_failing_to_delete_still_raises_for_the_retry():
    image_key = part_image_key(PART_ID)
    store = _FakeStore({image_key: b"page"})
    store.fail_deletes = {image_key}

    with pytest.raises(RuntimeError):
        delete_media_object(store, image_key)
    assert store.deletes == []
