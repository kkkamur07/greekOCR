"""Local filesystem storage for document part page images."""

from pathlib import Path
from uuid import UUID

from backend.core.settings import get_app_settings
from backend.document.infrastructure.media_store.keys import (
    DEFAULT_PART_IMAGE_SUFFIX,
    part_image_key,
    validate_image_key,
)


class LocalMediaStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or get_app_settings().media_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def part_image_key(
        self,
        part_id: UUID,
        *,
        suffix: str = DEFAULT_PART_IMAGE_SUFFIX,
        filename_stem: str | None = None,
    ) -> str:
        return part_image_key(part_id, suffix=suffix, filename_stem=filename_stem)

    def absolute_path(self, image_key: str) -> Path:
        validate_image_key(image_key)
        path = (self._root / image_key).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("Invalid image key")
        return path

    def write(self, image_key: str, data: bytes) -> None:
        path = self.absolute_path(image_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read(self, image_key: str) -> bytes:
        path = self.absolute_path(image_key)
        if not path.is_file():
            raise FileNotFoundError(image_key)
        return path.read_bytes()

    def delete(self, image_key: str) -> None:
        path = self.absolute_path(image_key)
        if path.is_file():
            path.unlink()
