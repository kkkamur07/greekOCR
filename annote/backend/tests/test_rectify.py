"""Rectify processing step — polygon mask onto axis-aligned rectangle."""

import numpy as np
from PIL import Image

from annote.services.processing.rectify import rectify
from tests.conftest import minimal_jpeg_bytes


def test_rectify_rectangle_produces_axis_aligned_crop():
    jpeg = minimal_jpeg_bytes(200, 100, color="red")
    page = np.array(Image.open(__import__("io").BytesIO(jpeg)))
    segment = {
        "kind": "rectangle",
        "points": [[20, 30], [120, 30], [120, 60], [20, 60]],
    }

    result = rectify(page, segment)

    assert result.ndim == 3
    assert result.shape == (30, 100, 3)


def test_rectify_polygon_produces_output():
    jpeg = minimal_jpeg_bytes(200, 100, color="blue")
    page = np.array(Image.open(__import__("io").BytesIO(jpeg)))
    segment = {
        "kind": "polygon",
        "points": [[10, 10], [80, 15], [75, 45], [15, 40]],
    }

    result = rectify(page, segment)

    assert result.shape[0] > 0 and result.shape[1] > 0


def test_rectify_many_point_polygon_uses_full_bbox_extent():
    """Multi-click polygons crop to the full selection bounding box."""
    jpeg = minimal_jpeg_bytes(400, 120, color="green")
    page = np.array(Image.open(__import__("io").BytesIO(jpeg)))
    segment = {
        "kind": "polygon",
        "points": [
            [50, 30],
            [55, 28],
            [60, 26],
            [65, 25],
            [300, 25],
            [305, 28],
            [310, 30],
            [310, 55],
            [305, 58],
            [300, 60],
            [65, 60],
            [60, 58],
            [55, 56],
            [50, 55],
        ],
    }

    result = rectify(page, segment)

    assert result.shape[1] >= 200
    assert result.shape[0] >= 20


def test_rectify_masks_pixels_outside_polygon():
    """Only ink inside the drawn polygon is kept; the rest is white."""
    page = np.zeros((80, 160, 3), dtype=np.uint8)
    page[:, :80] = [200, 0, 0]
    page[:, 80:] = [0, 0, 200]
    segment = {
        "kind": "polygon",
        "points": [[20, 20], [60, 22], [58, 50], [22, 48], [18, 35]],
    }

    result = rectify(page, segment)

    blue_pixels = (result[:, :, 2] > 150) & (result[:, :, 0] < 100)
    assert not np.any(blue_pixels)
    assert np.all(result[0, 0] == 255)
    assert np.all(result[-1, -1] == 255)
