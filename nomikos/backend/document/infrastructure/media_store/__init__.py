"""Document part page image storage (local filesystem or Supabase Storage)."""

from backend.document.infrastructure.media_store.encoding import (
    PERSISTED_THUMBNAIL_WIDTHS,
    THUMBNAIL_ENCODER_VERSION,
    DecodedPartImage,
    encode_part_image,
    encode_part_image_with_size,
    encode_part_thumbnail,
    persisted_thumbnail_key,
    persisted_thumbnail_keys,
    read_image_size,
    render_part_thumbnail,
)
from backend.document.infrastructure.media_store.errors import PresignUnsupported
from backend.document.infrastructure.media_store.factory import MediaStore, get_media_store
from backend.document.infrastructure.media_store.keys import (
    DEFAULT_PART_IMAGE_SUFFIX,
    derived_image_key,
    validate_image_key,
)
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
    "PERSISTED_THUMBNAIL_WIDTHS",
    "SIGNED_MEDIA_PREFIX",
    "THUMBNAIL_ENCODER_VERSION",
    "DecodedPartImage",
    "LocalMediaStore",
    "MediaStore",
    "PresignUnsupported",
    "SupabaseMediaStore",
    "clear_thumbnail_cache",
    "derived_image_key",
    "encode_part_image",
    "encode_part_image_with_size",
    "encode_part_thumbnail",
    "get_media_store",
    "persisted_thumbnail_key",
    "persisted_thumbnail_keys",
    "read_image_size",
    "render_part_thumbnail",
    "sign_object_path",
    "signature_is_valid",
    "validate_image_key",
]
