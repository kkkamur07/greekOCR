"""Segment refinement — Otsu ink + margin inside Kraken ceiling."""

import numpy as np
from PIL import Image

from annote.schemas.annotation import Segment
from annote.services.segment_refinement import refine_kraken_segments, refine_segment


def _gapped_character_line_fixture() -> tuple[Image.Image, list[list[float]]]:
    """Separate character blobs on one line — reproduces the single-glyph collapse bug."""
    image = Image.new("RGB", (400, 120), "white")
    pixels = image.load()
    for char_x in (40, 70, 100, 140, 180, 220, 260, 300, 340):
        for y in range(45, 72):
            for x in range(char_x, char_x + 18):
                pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 30.0], [380.0, 30.0], [380.0, 90.0], [20.0, 90.0]]
    return image, ceiling


def _synthetic_ink_fixture() -> tuple[Image.Image, list[list[float]]]:
    """White page with a horizontal ink bar and a fat Kraken ceiling around it."""
    image = Image.new("RGB", (200, 100), "white")
    pixels = image.load()
    for y in range(40, 56):
        for x in range(50, 151):
            pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 20.0], [180.0, 20.0], [180.0, 75.0], [20.0, 75.0]]
    return image, ceiling


def _assert_points_inside_ceiling(points: list[list[float]], ceiling: list[list[float]]) -> None:
    import cv2

    poly = np.array(ceiling, dtype=np.float32)
    for x, y in points:
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0:
            raise AssertionError(f"Point ({x}, {y}) is outside the Kraken ceiling")


def test_refine_segment_keeps_points_inside_ceiling_on_synthetic_ink():
    image, ceiling = _synthetic_ink_fixture()

    refined = refine_segment(image, ceiling)

    assert len(refined) >= 3
    _assert_points_inside_ceiling(refined, ceiling)


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def test_refine_segment_spans_gapped_characters_not_one_glyph():
    """Highly gapped ink may fall back to ceiling; output must stay inside Kraken."""
    image, ceiling = _gapped_character_line_fixture()

    refined = refine_segment(image, ceiling)

    _assert_points_inside_ceiling(refined, ceiling)
    assert len(refined) >= 3


def test_refine_segment_wraps_ink_with_outward_margin():
    image, ceiling = _synthetic_ink_fixture()

    refined = refine_segment(image, ceiling)
    _, refined_y0, _, refined_y1 = _bbox(refined)

    # Ink spans y=40..55; 4 px margin should pad above/below but stay inside ceiling.
    assert refined_y0 <= 40.0
    assert refined_y1 >= 55.0
    assert refined_y0 >= 20.0
    assert refined_y1 <= 75.0


def test_refine_segment_envelope_follows_ink_top_and_bottom():
    image = Image.new("RGB", (220, 100), "white")
    pixels = image.load()
    for x in range(30, 60):
        for y in range(50, 70):
            pixels[x, y] = (0, 0, 0)
    for x in range(90, 120):
        for y in range(40, 68):
            pixels[x, y] = (0, 0, 0)
    for x in range(150, 180):
        for y in range(52, 72):
            pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 30.0], [200.0, 30.0], [200.0, 85.0], [20.0, 85.0]]

    refined = refine_segment(image, ceiling)
    ys = [p[1] for p in refined]

    assert min(ys) <= 42
    assert max(ys) >= 68


def test_refine_segment_shrinks_toward_ink_on_synthetic_fixture():
    image, ceiling = _synthetic_ink_fixture()

    refined = refine_segment(image, ceiling)

    assert _polygon_area(refined) < _polygon_area(ceiling) * 0.75


def test_refine_segment_falls_back_to_ceiling_when_no_ink_signal():
    image = Image.new("RGB", (120, 80), "white")
    ceiling = [[10.0, 10.0], [110.0, 10.0], [110.0, 70.0], [10.0, 70.0]]

    refined = refine_segment(image, ceiling)

    assert refined == ceiling


def test_refine_kraken_segments_stamps_source_and_ceiling():
    image, ceiling = _synthetic_ink_fixture()
    segments = [
        Segment(
            id="seg-1",
            number=1,
            kind="polygon",
            points=ceiling,
        )
    ]

    refined = refine_kraken_segments(image, segments)

    assert refined[0].source == "kraken"
    assert refined[0].kraken_ceiling == ceiling
    assert refined[0].points != ceiling


def test_refine_segment_settles_margin_outside_ink_after_inward_shrink():
    image = Image.new("RGB", (120, 80), "white")
    pixels = image.load()
    for y in range(35, 50):
        for x in range(40, 81):
            pixels[x, y] = (0, 0, 0)
    ceiling = [[20.0, 20.0], [100.0, 20.0], [100.0, 65.0], [20.0, 65.0]]

    refined = refine_segment(image, ceiling, margin_px=4.0)
    ys = [p[1] for p in refined]

    assert min(ys) < 35
    assert max(ys) > 50


def test_refine_segment_returns_fewer_vertices_than_dense_kraken_boundary():
    image, ceiling = _synthetic_ink_fixture()
    dense_ceiling: list[list[float]] = []
    for x in range(20, 181, 2):
        dense_ceiling.append([float(x), 20.0])
    for y in range(22, 76, 2):
        dense_ceiling.append([180.0, float(y)])
    for x in range(178, 19, -2):
        dense_ceiling.append([float(x), 75.0])
    for y in range(73, 21, -2):
        dense_ceiling.append([20.0, float(y)])

    refined = refine_segment(image, dense_ceiling)

    assert len(refined) < len(dense_ceiling)
