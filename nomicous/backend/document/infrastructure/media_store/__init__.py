"""Document part page image storage (local filesystem or Supabase Storage)."""

from backend.document.infrastructure.media_store.encoding import (
    DecodedPartImage,
    encode_part_image,
    encode_part_image_with_size,
    encode_part_thumbnail,
    read_image_size,
    render_part_thumbnail,
)
from backend.document.infrastructure.media_store.factory import MediaStore, get_media_store
from backend.document.infrastructure.media_store.keys import DEFAULT_PART_IMAGE_SUFFIX
from backend.document.infrastructure.media_store.local import LocalMediaStore
from backend.document.infrastructure.media_store.signing import (
    SIGNED_MEDIA_PREFIX,
    sign_object_path,
    signature_is_valid,
)
from backend.document.infrastructure.media_store.supabase import SupabaseMediaStore
from backend.document.infrastructure.media_store.thumbnail_cache import clear_thumbnail_cache

__all__ = [
    "DEFAULT_PART_IMAGE_SUFFIX",
    "SIGNED_MEDIA_PREFIX",
    "DecodedPartImage",
    "LocalMediaStore",
    "MediaStore",
    "SupabaseMediaStore",
    "clear_thumbnail_cache",
    "encode_part_image",
    "encode_part_image_with_size",
    "encode_part_thumbnail",
    "get_media_store",
    "read_image_size",
    "render_part_thumbnail",
    "sign_object_path",
    "signature_is_valid",
]
