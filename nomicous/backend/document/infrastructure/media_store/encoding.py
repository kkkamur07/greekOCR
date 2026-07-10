"""Normalize uploaded page images to WebP for storage."""

from io import BytesIO

from PIL import Image

from backend.core.settings import get_storage_settings


def _webp_ready_image(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        return image.convert("RGBA")
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def encode_part_image(data: bytes) -> bytes:
    """Convert any supported raster image to WebP (lossless by default)."""
    settings = get_storage_settings()
    with Image.open(BytesIO(data)) as image:
        image.load()
        prepared = _webp_ready_image(image)

        buffer = BytesIO()
        save_kwargs: dict = {"format": "WEBP", "method": 6}
        if settings.media_webp_lossless:
            save_kwargs["lossless"] = True
        else:
            save_kwargs["quality"] = max(1, min(settings.media_webp_quality, 100))
        prepared.save(buffer, **save_kwargs)
        return buffer.getvalue()


def encode_part_thumbnail(data: bytes, width: int) -> bytes:
    """Encode a width-bounded, non-upscaled lossy WebP preview."""
    with Image.open(BytesIO(data)) as image:
        image.load()
        target_width = min(width, image.width)
        target_height = max(1, round(image.height * target_width / image.width))
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        _webp_ready_image(image).save(buffer, format="WEBP", quality=85, method=6)
        return buffer.getvalue()
