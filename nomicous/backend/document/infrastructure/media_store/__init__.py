"""Document part page image storage (local filesystem or Supabase Storage)."""

from backend.document.infrastructure.media_store.encoding import encode_part_image, encode_part_thumbnail
from backend.document.infrastructure.media_store.factory import MediaStore, get_media_store
from backend.document.infrastructure.media_store.keys import DEFAULT_PART_IMAGE_SUFFIX
from backend.document.infrastructure.media_store.local import LocalMediaStore
from backend.document.infrastructure.media_store.supabase import SupabaseMediaStore

__all__ = [
    "DEFAULT_PART_IMAGE_SUFFIX",
    "LocalMediaStore",
    "MediaStore",
    "SupabaseMediaStore",
    "encode_part_image",
    "encode_part_thumbnail",
    "get_media_store",
]
