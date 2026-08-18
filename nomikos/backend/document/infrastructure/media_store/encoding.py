"""Normalize uploaded page images to WebP for storage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from typing import NamedTuple

from PIL import Image, UnidentifiedImageError

from backend.core.exceptions import ValidationError
from backend.core.settings import get_storage_settings
from backend.document.infrastructure.media_store.thumbnail_cache import (
    get_cached_thumbnail,
    store_cached_thumbnail,
    thumbnail_cache_key,
)

INVALID_IMAGE_MESSAGE = "Uploaded file is not a valid image"
# Upper bound on the pixels a single decode may allocate. Enforced by this module from
# the image header, not by mutating ``Image.MAX_IMAGE_PIXELS`` process-wide, so every
# other decoder in the process keeps Pillow's default - which is stricter than this and
# therefore still the first bound to fire for the largest rasters.
MAX_DECODE_PIXELS = 200_000_000


class DecodedPartImage(NamedTuple):
    """Storage-ready bytes plus the source dimensions recovered from the same decode."""

    data: bytes
    width: int
    height: int


@contextmanager
def bounded_image(data: bytes) -> Iterator[Image.Image]:
    """Open an untrusted image and reject oversized rasters before they are decoded.

    ``Image.open`` parses only the header, so ``image.size`` is known before any pixel
    buffer is allocated; comparing it against ``MAX_DECODE_PIXELS`` here is a hard
    ceiling that costs nothing. Pillow's own guard is deliberately not used: it raises
    ``DecompressionBombError`` only past *twice* ``Image.MAX_IMAGE_PIXELS`` and merely
    warns below that, and tightening it means mutating ``Image.MAX_IMAGE_PIXELS`` and
    the warning filters, both of which are process-wide and neither of which is
    thread-safe. Decodes run on ``asyncio.to_thread`` workers, so two of them would
    interleave those mutations - one raising the global bound while the other is mid
    decode, letting an oversized image through and clobbering the restore.

    Pillow's untouched default is *stricter* than ``MAX_DECODE_PIXELS``, so it may reject
    a raster before this check does; either way the caller sees ``ValidationError`` and
    nothing above ``MAX_DECODE_PIXELS`` is ever decoded.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_DECODE_PIXELS:
                raise ValidationError(INVALID_IMAGE_MESSAGE)
            yield image
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ValidationError(INVALID_IMAGE_MESSAGE) from exc


def _webp_ready_image(image: Image.Image) -> Image.Image:
    # ``convert`` always materialises a new image, so the result outlives the decoder's
    # file handle - which is what lets callers encode outside ``bounded_image``.
    if image.mode in ("RGBA", "LA"):
        return image.convert("RGBA")
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def read_image_size(data: bytes) -> tuple[int, int]:
    """Read pixel dimensions from the header without decoding the raster."""
    with bounded_image(data) as image:
        return image.size


def encode_part_image_with_size(data: bytes) -> DecodedPartImage:
    """Validate, convert to WebP, and report the source dimensions from one decode."""
    settings = get_storage_settings()
    with bounded_image(data) as image:
        image.load()
        width, height = image.size
        prepared = _webp_ready_image(image)

    # Encoding sits outside the guard on purpose: ``bounded_image`` turns OSError into
    # ``ValidationError``, and a WebP *encoder* failure is our fault, not the uploader's.
    # Blaming the client with a 422 would hide a server bug behind a bad-input message.
    buffer = BytesIO()
    save_kwargs: dict = {"format": "WEBP", "method": 6}
    if settings.media_webp_lossless:
        save_kwargs["lossless"] = True
    else:
        save_kwargs["quality"] = max(1, min(settings.media_webp_quality, 100))
    prepared.save(buffer, **save_kwargs)
    return DecodedPartImage(data=buffer.getvalue(), width=width, height=height)


def encode_part_image(data: bytes) -> bytes:
    """Convert any supported raster image to WebP (lossless by default)."""
    return encode_part_image_with_size(data).data


def render_part_thumbnail(data: bytes, width: int) -> bytes:
    """Encode a width-bounded, non-upscaled lossy WebP preview."""
    with bounded_image(data) as image:
        image.load()
        target_width = min(width, image.width)
        target_height = max(1, round(image.height * target_width / image.width))
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        prepared = _webp_ready_image(image)

    # Outside the guard for the same reason as the full-size encode: an encoder failure
    # is a server fault and must not be reported as invalid input.
    buffer = BytesIO()
    prepared.save(buffer, format="WEBP", quality=85, method=6)
    return buffer.getvalue()


def encode_part_thumbnail(data: bytes, width: int) -> bytes:
    """Return a thumbnail, rendering it only when it is not already cached.

    The rendering is deliberately not serialised: two concurrent misses for the same
    variant both render and the second simply overwrites an identical entry, which is
    cheaper than making every hit wait behind a lock held across a decode.
    """
    key = thumbnail_cache_key(data, width)
    cached = get_cached_thumbnail(key)
    if cached is not None:
        return cached
    encoded = render_part_thumbnail(data, width)
    store_cached_thumbnail(key, encoded)
    return encoded
