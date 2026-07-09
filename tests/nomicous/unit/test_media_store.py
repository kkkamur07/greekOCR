"""MediaStore path safety."""

import uuid
from pathlib import Path

import pytest

from backend.document.infrastructure.media_store import LocalMediaStore


@pytest.fixture
def store(tmp_path: Path) -> LocalMediaStore:
    return LocalMediaStore(root=tmp_path)


# --- Read/write roundtrip ---
# Tests part image keys stay under the store root. Does not test HTTP upload.


def test_part_image_key_and_roundtrip(store: LocalMediaStore) -> None:
    part_id = uuid.uuid4()
    key = store.part_image_key(part_id, suffix="webp")
    store.write(key, b"pixels")
    assert store.read(key) == b"pixels"
    assert store.absolute_path(key).is_relative_to(store._root)


# --- Path traversal rejection ---
# Tests malicious keys cannot escape the media root. Does not test symlink attacks.


def test_rejects_traversal_keys(store: LocalMediaStore) -> None:
    with pytest.raises(ValueError, match="Invalid image key"):
        store.absolute_path("../outside.png")
    with pytest.raises(ValueError, match="Invalid image key"):
        store.absolute_path("parts/../../etc/passwd")


# --- Resolved paths stay under root ---
# Tests normal keys resolve inside the configured media directory.


def test_resolved_path_stays_under_root(store: LocalMediaStore) -> None:
    part_id = uuid.uuid4()
    key = store.part_image_key(part_id, suffix="webp")
    resolved = store.absolute_path(key)
    assert resolved.is_relative_to(store._root)
