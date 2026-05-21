"""Local filesystem storage for DocumentPart images."""

from pathlib import Path
from uuid import UUID

from backend.core.settings import get_app_settings


class MediaStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or get_app_settings().media_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def part_image_key(self, part_id: UUID, *, suffix: str = "bin") -> str:
        safe = suffix.lstrip(".") or "bin"
        return f"parts/{part_id}.{safe}"

    def absolute_path(self, image_key: str) -> Path:
        path = (self._root / image_key).resolve()
        if not str(path).startswith(str(self._root)):
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
