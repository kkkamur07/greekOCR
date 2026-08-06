"""Mask rasterization behind the simplification quality gate.

The gate compares two line polygons by IoU. It used to do that on two
page-sized masks per iteration, twenty iterations per line, forty lines per
page. These tests pin the answer to what a page-sized rasterization gives, so
the windowed version cannot drift from it.
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.preprocessing import segment_geometry
from inference.preprocessing.segment_geometry import (
    candidate_quality,
    mask_from_polygon,
    mask_iou,
    mask_window,
)

PAGE = (4000, 6000)


def _page_iou(a: list[list[float]], b: list[list[float]]) -> float:
    width, height = PAGE
    return mask_iou(
        mask_from_polygon(a, width=width, height=height),
        mask_from_polygon(b, width=width, height=height),
    )


def _quality_iou(candidate: list[list[float]], reference: list[list[float]]) -> float:
    width, height = PAGE
    _, metrics = candidate_quality(
        candidate,
        reference,
        width=width,
        height=height,
        bbox_tolerance=1.0,
        baseline=None,
    )
    return metrics["simplification_iou"]


CASES = {
    "overlapping": (
        [[1000.0, 2000.0], [1400.0, 2000.0], [1400.0, 2060.0], [1000.0, 2060.0]],
        [[1010.0, 1995.0], [1395.0, 2005.0], [1402.0, 2070.0], [995.0, 2055.0]],
    ),
    "identical": (
        [[10.0, 10.0], [90.0, 12.0], [88.0, 44.0], [12.0, 40.0]],
        [[10.0, 10.0], [90.0, 12.0], [88.0, 44.0], [12.0, 40.0]],
    ),
    "disjoint": (
        [[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]],
        [[900.0, 900.0], [960.0, 900.0], [960.0, 940.0], [900.0, 940.0]],
    ),
    "fractional_vertices": (
        [[100.4, 200.6], [340.5, 199.5], [341.2, 260.9], [99.7, 261.4]],
        [[101.9, 201.1], [339.4, 200.4], [340.6, 259.5], [100.2, 260.6]],
    ),
    "partly_off_the_page": (
        [[-50.0, -20.0], [300.0, -20.0], [300.0, 80.0], [-50.0, 80.0]],
        [[-70.0, -25.0], [290.0, -10.0], [295.0, 75.0], [-60.0, 70.0]],
    ),
    "touching_the_far_edge": (
        [[3900.0, 5900.0], [4200.0, 5900.0], [4200.0, 6100.0], [3900.0, 6100.0]],
        [[3910.0, 5890.0], [4300.0, 5905.0], [4250.0, 6050.0], [3890.0, 6080.0]],
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_windowed_iou_matches_a_page_sized_rasterization(name: str) -> None:
    candidate, reference = CASES[name]

    assert _quality_iou(candidate, reference) == pytest.approx(_page_iou(candidate, reference))


def test_a_polygon_with_no_points_has_no_window() -> None:
    assert mask_window(([],), width=100, height=100) is None


def test_a_polygon_entirely_off_the_page_has_no_window() -> None:
    width, height = PAGE
    off_page = [[-90.0, -90.0], [-10.0, -90.0], [-10.0, -20.0]]

    assert mask_window((off_page,), width=width, height=height) is None


def test_the_gate_allocates_a_line_sized_mask_not_a_page_sized_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the window: a 4000x6000 page must not cost 24 MB per compare."""
    width, height = PAGE
    sizes: list[tuple[int, int]] = []
    real = segment_geometry.mask_from_polygon

    def spy(points, **kwargs):
        sizes.append((kwargs["width"], kwargs["height"]))
        return real(points, **kwargs)

    monkeypatch.setattr(segment_geometry, "mask_from_polygon", spy)
    candidate, reference = CASES["overlapping"]

    candidate_quality(
        candidate,
        reference,
        width=width,
        height=height,
        bbox_tolerance=1.0,
        baseline=None,
    )

    assert sizes
    # The two polygons span roughly 410x80; the page is 4000x6000.
    assert all(mask_width * mask_height < width * height / 100 for mask_width, mask_height in sizes)


def test_an_integer_origin_translates_the_raster_exactly() -> None:
    polygon = [[12.0, 9.0], [40.0, 11.0], [38.0, 30.0], [10.0, 28.0]]
    full = mask_from_polygon(polygon, width=60, height=50)
    windowed = mask_from_polygon(polygon, width=40, height=30, origin=(8, 5))

    np.testing.assert_array_equal(windowed, full[5:35, 8:48])
