"""Poly output mode for the PP-OCRv6 detection segmenter.

Synthetic probability maps and hand-built quads only, no model needed. The
quad path stays byte-identical (same quads, scores and responses as quad
mode); the polygon follows the blob, merges union, cuts stay disjoint,
simplification caps at 64 points, failures fall back to the quad, and every
served polygon stays inside the page and about one line tall.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.polygons import (
    MAX_POLYGON_POINTS,
    intersect_outline,
    ring_area,
    simplify_outline,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad, detect_lines
from nomikos_inference.architectures.ppocr_det.ppocr_det import _box_type
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines
from nomikos_inference.architectures.ppocr_det.response import (
    build_ppocr_det_response,
    build_refined_ppocr_det_response,
)


def _banana_map() -> np.ndarray:
    prob = np.zeros((160, 160), dtype=np.float32)
    cv2.ellipse(prob, (80, 80), (60, 30), 0, 10, 170, 1.0, 14)
    return prob


def _rect_map() -> np.ndarray:
    prob = np.zeros((1, 1, 128, 128), dtype=np.float32)
    prob[0, 0, 20:36, 20:110] = 1.0
    return prob


def _detect(prob: np.ndarray, box_type: str, **kwargs) -> list[DetectedQuad]:
    size = prob.shape[-1]
    return detect_lines(
        prob.reshape(prob.shape[-2], prob.shape[-1]),
        orig_width=size,
        orig_height=size,
        ratio_h=1.0,
        ratio_w=1.0,
        box_type=box_type,
        **kwargs,
    )


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return DetectedQuad(points=points, score=score, polygon=[list(p) for p in points])


def _point_to_segment(point: list[float], first: list[float], second: list[float]) -> float:
    point = np.asarray(point, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    leg = second - first
    length = float(np.linalg.norm(leg))
    if length <= 0:
        return float(np.linalg.norm(point - first))
    return abs(float(leg[0] * (first[1] - point[1]) - leg[1] * (first[0] - point[0]))) / length


def test_curved_blob_polygon_hugs_the_blob() -> None:
    quads = _detect(_banana_map(), "poly", box_thresh=0.2)

    assert len(quads) == 1
    quad = quads[0]
    assert quad.polygon is not None
    assert not quad.polygon_fallback
    assert len(quad.polygon) > 4
    ratio = Polygon(quad.polygon).area / Polygon(quad.points).area
    assert ratio <= 0.8


def test_straight_blob_baseline_matches_quad_baseline() -> None:
    quads = _detect(_rect_map(), "poly")
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(quads, layout, box_type="poly", page_width=128, page_height=128)
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    assert len(poly_items) == len(quad_items) == 1
    item, quad_item = poly_items[0], quad_items[0]
    assert item.members == quad_item.members
    assert len(item.baseline) >= 2
    first, second = quad_item.baseline
    for point in item.baseline:
        assert _point_to_segment(point, first, second) <= 1.0


def test_quad_mode_matches_poly_quads_and_is_deterministic() -> None:
    first = _detect(_rect_map(), "quad")
    second = _detect(_rect_map(), "quad")
    poly = _detect(_rect_map(), "poly")

    assert [quad.points for quad in first] == [quad.points for quad in second]
    assert [quad.points for quad in first] == [quad.points for quad in poly]
    assert [quad.score for quad in first] == [quad.score for quad in poly]
    assert all(quad.polygon is None for quad in first)
    assert all(len(quad.points) == 4 for quad in first)

    layout = layout_lines(first, direction="ltr")
    items = refine_to_lines(first, layout, box_type="quad")
    response = build_refined_ppocr_det_response(128, 128, first, items, layout)
    again = build_refined_ppocr_det_response(
        128, 128, first, refine_to_lines(first, layout, box_type="quad"), layout
    )
    assert response.model_dump_json() == again.model_dump_json()
    for line in response.lines:
        assert len(line.points) == 4
        assert len(line.baseline["points"]) == 2
        assert "polygon_fallback" not in line.source_metadata


def test_bad_box_type_raises() -> None:
    with pytest.raises(ValueError):
        _detect(_rect_map(), "circle")
    with pytest.raises(ValueError):
        _box_type({"box_type": "circle"})
    quads = _detect(_rect_map(), "quad")
    layout = layout_lines(quads, direction="ltr")
    with pytest.raises(ValueError):
        refine_to_lines(quads, layout, box_type="circle")
    with pytest.raises(ValueError):
        build_ppocr_det_response(128, 128, quads, box_type="circle")


def test_default_box_type_is_poly() -> None:
    assert _box_type({}) == "poly"


def test_merge_unions_touching_polygons() -> None:
    quads = [
        _quad(0, 0, 200, 20),
        _quad(0, 40, 200, 60),
        _quad(0, 80, 200, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(100, 120, 200, 140, score=0.6),
    ]
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(quads, layout, box_type="poly", page_width=300, page_height=300)
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    merged = [item for item in poly_items if item.merged_from == 2]
    assert len(merged) == 1
    left = Polygon(quads[3].polygon)
    right = Polygon(quads[4].polygon)
    assert Polygon(merged[0].points).area == pytest.approx(left.area + right.area, rel=0.05)
    assert [item.members for item in poly_items] == [item.members for item in quad_items]


def test_merge_hulls_separated_fragments() -> None:
    quads = [
        _quad(0, 0, 200, 20),
        _quad(0, 40, 200, 60),
        _quad(0, 80, 200, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(103, 122, 220, 142, score=0.6),
    ]
    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(quads, layout, box_type="poly", page_width=300, page_height=300)

    merged = [item for item in items if item.merged_from == 2]
    assert len(merged) == 1
    # Simplification may shave sub-epsilon corners, so allow a 1 px slack.
    outline = Polygon(merged[0].points).buffer(1.0)
    assert outline.covers(Polygon(quads[3].polygon))
    assert outline.covers(Polygon(quads[4].polygon))


def test_overlap_cut_leaves_polygons_disjoint() -> None:
    quads = [
        _quad(0, 0, 100, 15),
        _quad(0, 25, 100, 40),
        _quad(0, 50, 100, 65),
        _quad(0, 70, 100, 95),
        _quad(0, 85, 100, 110),
    ]
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(quads, layout, box_type="poly", page_width=200, page_height=200)
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    pair = [item for item in poly_items if set(item.members) & {3, 4}]
    assert len(pair) == 2
    cut = intersect_outline(pair[0].points, pair[1].points)
    assert cut is None or ring_area(cut) == pytest.approx(0.0)
    assert [item.members for item in poly_items] == [item.members for item in quad_items]
    assert [item.role for item in poly_items] == [item.role for item in quad_items]


def test_simplify_caps_dense_rings_at_64_points() -> None:
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = [[100.0 + 80 * float(np.cos(a)), 100.0 + 80 * float(np.sin(a))] for a in angles]

    simplified = simplify_outline(circle, median_height=20.0, page_width=200, page_height=200)

    assert simplified is not None
    assert 4 <= len(simplified) <= MAX_POLYGON_POINTS
    assert all(simplified[i] != simplified[i + 1] for i in range(len(simplified) - 1))


def test_triangle_blob_falls_back_to_the_quad() -> None:
    prob = np.zeros((128, 128), dtype=np.float32)
    cv2.fillPoly(prob, [np.array([[10, 110], [120, 110], [65, 10]])], 1.0)
    quads = detect_lines(
        prob, orig_width=128, orig_height=128, ratio_h=1.0, ratio_w=1.0, box_type="poly"
    )

    assert len(quads) == 1
    assert quads[0].polygon_fallback
    assert quads[0].polygon == quads[0].points

    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(quads, layout, box_type="poly", page_width=128, page_height=128)
    response = build_refined_ppocr_det_response(128, 128, quads, items, layout, box_type="poly")
    assert response.lines[0].source_metadata["polygon_fallback"] is True


def test_polygons_stay_inside_the_page_and_about_one_line_tall() -> None:
    quads = _detect(_banana_map(), "poly", box_thresh=0.2) + _detect(_rect_map(), "poly")
    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(quads, layout, box_type="poly", page_width=160, page_height=160)

    assert len(items) == 2
    quads_by_member = {member: quads[member] for item in items for member in item.members}
    for item in items:
        for x, y in item.points:
            assert 0.0 <= x <= 160.0
            assert 0.0 <= y <= 160.0
        assert 4 <= len(item.points) <= MAX_POLYGON_POINTS
        poly_height = max(p[1] for p in item.points) - min(p[1] for p in item.points)
        quad = quads_by_member[item.members[0]]
        quad_height = max(p[1] for p in quad.points) - min(p[1] for p in quad.points)
        assert poly_height <= quad_height + 2.0
