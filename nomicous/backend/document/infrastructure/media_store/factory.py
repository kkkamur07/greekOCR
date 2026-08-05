"""Select the configured media store backend."""

from datetime import datetime
from functools import lru_cache
from typing import Protocol
from uuid import UUID

from backend.core.settings import get_storage_settings
from backend.document.infrastructure.media_store.local import LocalMediaStore
from backend.document.infrastructure.media_store.supabase import SupabaseMediaStore


class MediaStore(Protocol):
    def part_image_key(
        self,
        part_id: UUID,
        *,
        suffix: str = ...,
        filename_stem: str | None = None,
    ) -> str: ...

    def signed_object_url(self, image_key: str, *, expires_at: datetime) -> str:
        """A link that reaches this one object without any further credential.

        Part of the store rather than of the claim endpoint because only the
        store knows how its objects are reached: Supabase signs a Storage URL,
        the filesystem signs a path the platform serves itself. The claim route
        asks for a link and does not care which.

        The returned link may be absolute or root-relative; a relative one is
        resolved against the platform's own base URL by whoever hands it out.
        """
        ...

    def write(self, image_key: str, data: bytes) -> None: ...

    def read(self, image_key: str) -> bytes: ...

    def delete(self, image_key: str) -> None: ...


@lru_cache
def get_media_store() -> MediaStore:
    if get_storage_settings().storage_backend == "supabase":
        return SupabaseMediaStore()
    return LocalMediaStore()
