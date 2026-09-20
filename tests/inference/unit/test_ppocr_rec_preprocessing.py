"""PP-OCR recognition preprocessing: the kraken recipe, and only that.

Torch-free by construction, like everything under ``tests/inference``: the
test never imports kraken or torchvision, it pins the recipe structurally
(exact widths, exact padding columns, exact scale and invert values) so a
regression fails here before it costs an end-to-end parity run.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.ppocr_rec.preprocessing import (
    MIN_PREPROCESSED_WIDTH,
    ensure_minimum_width,
    fixed_resize_to_height,
    open_line_image,
    pad_line_sides,
    preprocess_line_image_bytes_to_ppocr_rec_tensor,
)


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_output_shape_dtype_and_range() -> None:
    tensor = preprocess_line_image_bytes_to_ppocr_rec_tensor(
        _png_bytes(Image.new("RGB", (200, 50), (10, 20, 30))),
        line_height=96,
        pad=16,
        pad_fill=255,
    )
    # int(200 * 96 / 50) = 384, plus 16 white columns on each side.
    assert tensor.shape == (1, 3, 96, 416)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_resize_truncates_the_width_like_kraken() -> None:
    """kraken truncates (``int(w * oh / h)``); rounding would give 14 here."""
    resized = fixed_resize_to_height(Image.new("RGB", (1, 7), "white"), 96)
    assert resized.size == (13, 96)


def test_resize_keeps_aspect_ratio() -> None:
    resized = fixed_resize_to_height(Image.new("RGB", (200, 50), "white"), 96)
    assert resized.size == (384, 96)


def test_side_padding_is_exact_and_white() -> None:
    tensor = preprocess_line_image_bytes_to_ppocr_rec_tensor(
        _png_bytes(Image.new("RGB", (100, 96), (0, 0, 0))),
        line_height=96,
        pad=16,
        pad_fill=255,
    )
    # 100 px of black ink between 16 white columns on each side, inverted.
    assert tensor.shape == (1, 3, 96, 132)
    assert np.all(tensor[:, :, :, :16] == 0.0)
    assert np.all(tensor[:, :, :, -16:] == 0.0)
    assert np.all(tensor[:, :, :, 16:-16] == 1.0)


def test_inversion_maps_white_to_zero_and_black_to_one() -> None:
    image = Image.new("RGB", (4, 96), "black")
    for y in range(96):
        image.putpixel((1, y), (255, 255, 255))
    tensor = preprocess_line_image_bytes_to_ppocr_rec_tensor(
        _png_bytes(image),
        line_height=96,
        pad=0,
        pad_fill=255,
    )
    assert tensor.shape == (1, 3, 96, MIN_PREPROCESSED_WIDTH)
    # Column 1 is white, the rest is black; only the padded tail is unknown.
    assert np.all(tensor[:, :, :, 1] == 0.0)
    assert np.all(tensor[:, :, :, 0] == 1.0)


def test_narrow_line_is_extended_on_the_right_only() -> None:
    """Padded widths below 16 grow on the right with ``pad_fill``, up to 16."""
    tensor = preprocess_line_image_bytes_to_ppocr_rec_tensor(
        _png_bytes(Image.new("RGB", (5, 96), (0, 0, 0))),
        line_height=96,
        pad=0,
        pad_fill=255,
    )
    assert tensor.shape == (1, 3, 96, 16)
    assert np.all(tensor[:, :, :, :5] == 1.0)
    assert np.all(tensor[:, :, :, 5:] == 0.0)


def test_grayscale_input_becomes_three_channels() -> None:
    tensor = preprocess_line_image_bytes_to_ppocr_rec_tensor(
        _png_bytes(Image.new("L", (100, 50), 128)),
        line_height=96,
        pad=16,
        pad_fill=255,
    )
    assert tensor.shape[1] == 3


def test_open_line_image_decodes_to_rgb() -> None:
    output = BytesIO()
    Image.new("L", (10, 10), 200).save(output, format="PNG")
    image = open_line_image(output.getvalue())
    assert image.mode == "RGB"
    assert image.size == (10, 10)


def test_pad_line_sides_places_the_original_at_the_offset() -> None:
    image = Image.new("RGB", (10, 4), (7, 8, 9))
    padded = pad_line_sides(image, 16, 255)
    assert padded.size == (42, 4)
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((16, 0)) == (7, 8, 9)
    assert padded.getpixel((25, 0)) == (7, 8, 9)
    assert padded.getpixel((41, 0)) == (255, 255, 255)


def test_ensure_minimum_width_leaves_wide_images_alone() -> None:
    image = Image.new("RGB", (32, 96), "white")
    assert ensure_minimum_width(image, 255) is image


def test_ensure_minimum_width_grows_right_only() -> None:
    image = Image.new("RGB", (5, 96), (0, 0, 0))
    extended = ensure_minimum_width(image, 255)
    assert extended.size == (16, 96)
    assert extended.getpixel((0, 0)) == (0, 0, 0)
    assert extended.getpixel((5, 0)) == (255, 255, 255)
    assert extended.getpixel((15, 0)) == (255, 255, 255)


def test_invalid_arguments_raise() -> None:
    image = Image.new("RGB", (10, 10), "white")
    with pytest.raises(ValueError, match="line_height must be positive"):
        fixed_resize_to_height(image, 0)
    with pytest.raises(ValueError, match="pad must be non-negative"):
        pad_line_sides(image, -1, 255)
    with pytest.raises(ValueError, match="pad_fill must be a uint8 value"):
        pad_line_sides(image, 16, 256)


def test_undecodable_bytes_raise() -> None:
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        preprocess_line_image_bytes_to_ppocr_rec_tensor(
            b"not an image", line_height=96, pad=16, pad_fill=255
        )
