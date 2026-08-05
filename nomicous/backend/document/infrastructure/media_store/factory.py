"""Select the configured media store backend."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from backend.core.settings import get_storage_settings
from backend.core.settings._cache import settings_cache
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


@settings_cache
def get_media_store() -> MediaStore:
    """The configured store, built once per process.

    Enrolled in ``reset_settings_caches`` rather than memoized with a bare
    ``lru_cache``: which backend this returns is read from ``STORAGE_BACKEND``,
    so a plain cache keeps serving the backend that was configured the first
    time anything asked - exactly the leak that registry exists to remove.
    """
    if get_storage_settings().storage_backend == "supabase":
        return SupabaseMediaStore()
    return LocalMediaStore()
