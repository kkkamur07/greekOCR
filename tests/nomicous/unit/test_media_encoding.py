"""WebP encoding for stored page images."""

from io import BytesIO

from PIL import Image

from backend.document.infrastructure.media_store.encoding import encode_part_image


def _sample_png() -> bytes:
    image = Image.new("RGB", (8, 8), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_encode_part_image_produces_webp() -> None:
    encoded = encode_part_image(_sample_png())
    with Image.open(BytesIO(encoded)) as image:
        assert image.format == "WEBP"
        assert image.size == (8, 8)
