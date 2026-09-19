"""PP-OCRv6 detection preprocessing: shapes, ratios and normalisation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nomikos_inference.architectures.ppocr_det.preprocessing import (
    PPOCR_DET_MEAN,
    PPOCR_DET_STD,
    preprocess_ppocr_det_image,
)


def test_landscape_above_the_limit_scales_by_the_longer_side() -> None:
    image = Image.new("RGB", (3000, 1000), (10, 20, 30))

    tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=1920)

    assert tensor.shape == (1, 3, 640, 1920)
    assert tensor.dtype == np.float32
    assert (meta.orig_width, meta.orig_height) == (3000, 1000)
    assert meta.ratio_h == 640 / 1000
    assert meta.ratio_w == 1920 / 3000


def test_portrait_above_the_limit_scales_by_the_longer_side() -> None:
    image = Image.new("RGB", (1000, 3000), (10, 20, 30))

    tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=1920)

    assert tensor.shape == (1, 3, 1920, 640)
    assert meta.ratio_h == 1920 / 3000
    assert meta.ratio_w == 640 / 1000


def test_image_below_the_limit_is_not_upscaled() -> None:
    image = Image.new("RGB", (800, 600), (10, 20, 30))

    tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=1920)

    # No limit scaling (the longer side stays 800), but PaddleX still snaps
    # each side to a multiple of 32, so 600 becomes 608.
    assert tensor.shape == (1, 3, 608, 800)
    assert meta.ratio_h == 608 / 600
    assert meta.ratio_w == 1.0


def test_sides_round_to_multiples_of_32_with_a_floor() -> None:
    tensor, meta = preprocess_ppocr_det_image(Image.new("RGB", (100, 100), "white"))

    assert tensor.shape == (1, 3, 96, 96)
    assert meta.ratio_h == 0.96
    assert meta.ratio_w == 0.96

    tiny, _ = preprocess_ppocr_det_image(Image.new("RGB", (10, 10), "white"))

    assert tiny.shape == (1, 3, 32, 32)


def test_normalisation_matches_paddlex_bgr_channel_order() -> None:
    # Solid red in PIL (RGB) is stored BGR by the PaddleX decode, so channel 0
    # (blue, value 0) normalises with the first mean/std and channel 2 (red,
    # value 255) with the last pair. RGB order would swap the two.
    tensor, _ = preprocess_ppocr_det_image(Image.new("RGB", (96, 96), (255, 0, 0)))

    blue = (0 / 255 - PPOCR_DET_MEAN[0]) / PPOCR_DET_STD[0]
    green = (0 / 255 - PPOCR_DET_MEAN[1]) / PPOCR_DET_STD[1]
    red = (255 / 255 - PPOCR_DET_MEAN[2]) / PPOCR_DET_STD[2]
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        np.array([blue, green, red], dtype=np.float32),
        rtol=1e-5,
    )


def test_custom_limit_side_len_is_respected() -> None:
    image = Image.new("RGB", (3000, 1000), (10, 20, 30))

    tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=960)

    assert tensor.shape == (1, 3, 320, 960)
    assert meta.ratio_h == 320 / 1000
    assert meta.ratio_w == 960 / 3000


def test_non_positive_limit_is_rejected() -> None:
    image = Image.new("RGB", (96, 96), "white")

    with pytest.raises(ValueError):
        preprocess_ppocr_det_image(image, limit_side_len=0)
