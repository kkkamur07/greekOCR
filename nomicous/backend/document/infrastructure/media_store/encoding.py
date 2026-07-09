"""Normalize uploaded page images to WebP for storage."""

from io import BytesIO

from PIL import Image

from backend.core.settings import get_storage_settings


def encode_part_image(data: bytes) -> bytes:
    """Convert any supported raster image to WebP (lossless by default)."""
    settings = get_storage_settings()
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode in ("RGBA", "LA"):
            prepared = image.convert("RGBA")
        elif image.mode == "P" and "transparency" in image.info:
            prepared = image.convert("RGBA")
        else:
            prepared = image.convert("RGB")

        buffer = BytesIO()
        save_kwargs: dict = {"format": "WEBP", "method": 6}
        if settings.media_webp_lossless:
            save_kwargs["lossless"] = True
        else:
            save_kwargs["quality"] = max(1, min(settings.media_webp_quality, 100))
        prepared.save(buffer, **save_kwargs)
        return buffer.getvalue()
