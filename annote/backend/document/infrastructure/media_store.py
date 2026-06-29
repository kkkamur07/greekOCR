"""Local filesystem storage for DocumentPart images."""

import re
from pathlib import Path
from uuid import UUID

from backend.core.settings import get_app_settings

# Keys are generated as parts/{uuid}/{filename}.{suffix}; older UUID-stemmed
# keys remain valid for already uploaded test/dev media.
_SAFE_IMAGE_KEY = re.compile(
    r"^parts/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?:/[a-z0-9][a-z0-9_-]{0,127})?\.[a-z0-9]{1,16}$"
)


class MediaStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or get_app_settings().media_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def part_image_key(
        self, part_id: UUID, *, suffix: str = "bin", filename_stem: str | None = None
    ) -> str:
        safe = re.sub(r"[^a-z0-9]", "", suffix.lstrip(".").lower())[:16] or "bin"
        if filename_stem is None:
            return f"parts/{part_id}.{safe}"
        stem = re.sub(r"[^a-z0-9_-]", "-", filename_stem.lower()).strip("-_")[:128]
        if not stem:
            stem = str(part_id)
        return f"parts/{part_id}/{stem}.{safe}"

    def _validate_image_key(self, image_key: str) -> None:
        if not image_key or image_key.startswith("/"):
            raise ValueError("Invalid image key")
        if Path(image_key).is_absolute():
            raise ValueError("Invalid image key")
        if ".." in Path(image_key).parts:
            raise ValueError("Invalid image key")
        if not _SAFE_IMAGE_KEY.match(image_key):
            raise ValueError("Invalid image key")

    def absolute_path(self, image_key: str) -> Path:
        self._validate_image_key(image_key)
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
