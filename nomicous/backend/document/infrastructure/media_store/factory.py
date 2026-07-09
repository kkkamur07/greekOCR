"""Select the configured media store backend."""

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

    def write(self, image_key: str, data: bytes) -> None: ...

    def read(self, image_key: str) -> bytes: ...

    def delete(self, image_key: str) -> None: ...


@lru_cache
def get_media_store() -> MediaStore:
    if get_storage_settings().storage_backend == "supabase":
        return SupabaseMediaStore()
    return LocalMediaStore()
