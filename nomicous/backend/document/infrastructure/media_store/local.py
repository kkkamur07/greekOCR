"""Local filesystem storage for document part page images."""

from datetime import datetime
from pathlib import Path
from uuid import UUID

from backend.core.settings import get_app_settings
from backend.document.infrastructure.media_store.errors import PresignUnsupported
from backend.document.infrastructure.media_store.keys import (
    DEFAULT_PART_IMAGE_SUFFIX,
    part_image_key,
    validate_image_key,
)
from backend.document.infrastructure.media_store.signing import sign_object_path


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

    def create_upload_url(self, image_key: str, *, expires_at: datetime) -> tuple[str, str]:
        """A filesystem has no presigned upload URL - the browser cannot write it.

        The local backend serves uploads through the API's own byte handling, so a
        presigned upload is impossible by construction. Callers that offer the
        direct-upload path must first check the backend and fall back to the plain
        multipart upload; this method exists only to make the store protocol honest
        and raises if it is ever asked to mint a URL it cannot.
        """
        raise PresignUnsupported("local media store cannot presign uploads")

    def signed_object_url(self, image_key: str, *, expires_at: datetime) -> str:
        """A root-relative signed link to this one object.

        Relative because a filesystem has no public origin of its own: the caller
        resolves it against the platform's own base URL, and the bytes come back
        from the platform's ``/media/signed`` route. That is the one place a
        manuscript scan is served by the API rather than by object storage, and
        it is why the route refuses to answer unless this backend is selected -
        production runs on Supabase, where the signature is checked by storage
        and the bytes never touch a serverless function.
        """
        validate_image_key(image_key)
        return sign_object_path(image_key, expires_at=expires_at)

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
