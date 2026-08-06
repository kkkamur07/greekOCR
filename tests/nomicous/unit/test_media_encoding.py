"""WebP encoding for stored page images."""

from io import BytesIO

import pytest
from PIL import Image

from backend.document.infrastructure.media_store import encoding
from backend.document.infrastructure.media_store.encoding import (
    encode_part_image,
    encode_part_thumbnail,
    render_part_thumbnail,
)
from backend.document.infrastructure.media_store.thumbnail_cache import clear_thumbnail_cache


def _sample_png(size: tuple[int, int] = (8, 8)) -> bytes:
    image = Image.new("RGB", size, color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _empty_thumbnail_cache():
    clear_thumbnail_cache()
    yield
    clear_thumbnail_cache()


def test_encode_part_image_produces_webp() -> None:
    encoded = encode_part_image(_sample_png())
    with Image.open(BytesIO(encoded)) as image:
        assert image.format == "WEBP"
        assert image.size == (8, 8)


def test_repeated_thumbnail_requests_render_once(monkeypatch) -> None:
    source = _sample_png((40, 20))
    renders: list[int] = []

    def counting_render(data: bytes, width: int) -> bytes:
        renders.append(width)
        return render_part_thumbnail(data, width)

    monkeypatch.setattr(encoding, "render_part_thumbnail", counting_render)

    first = encode_part_thumbnail(source, 20)
    second = encode_part_thumbnail(source, 20)
    other_width = encode_part_thumbnail(source, 10)

    assert first == second
    assert renders == [20, 10]
    with Image.open(BytesIO(other_width)) as image:
        assert image.size == (10, 5)


def test_thumbnail_cache_distinguishes_sources_with_the_same_width() -> None:
    wide = encode_part_thumbnail(_sample_png((40, 20)), 20)
    tall = encode_part_thumbnail(_sample_png((10, 40)), 20)

    assert wide != tall
    with Image.open(BytesIO(tall)) as image:
        assert image.size == (10, 40)
