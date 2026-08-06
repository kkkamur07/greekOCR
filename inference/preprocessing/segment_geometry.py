"""Polygon geometry helpers for segment preprocessing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_VERTEX_SPACING_PX = 3.0


def distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def line_height(points: list[list[float]]) -> float:
    _, y0, _, y1 = bbox(points)
    return max(y1 - y0, 1.0)


def bottom_edge_baseline(points: list[list[float]]) -> list[list[float]]:
    """Derive a text baseline from the bottom edge of a polygon.

    Deliberately the same rule as the platform's own
    ``annotation.application.line_geometry.default_baseline_from_polygon``,
    because the two produce baselines an annotator sees side by side. A segment
    through the bounding-box mid-height is not a baseline - it is a
    strike-through - and a transcriber who accepts a split line inherits it.
    """
    if len(points) < 2:
        return [[float(point[0]), float(point[1])] for point in points]

    max_y = max(point[1] for point in points)
    bottom = sorted(
        ([float(point[0]), float(point[1])] for point in points if point[1] >= max_y - 1e-6),
        key=lambda point: point[0],
    )
    if len(bottom) >= 2:
        return [bottom[0], bottom[-1]]
    x0, _, x1, _ = bbox(points)
    return [[float(x0), float(max_y)], [float(x1), float(max_y)]]


def _clip_segment_to_x_span(
    start: list[float],
    end: list[float],
    x0: float,
    x1: float,
) -> list[list[float]]:
    start_x, start_y = float(start[0]), float(start[1])
    end_x, end_y = float(end[0]), float(end[1])
    if max(start_x, end_x) < x0 or min(start_x, end_x) > x1:
        return []
    if abs(end_x - start_x) <= 1e-9:
        return [[start_x, start_y], [end_x, end_y]]

    def at(x: float) -> list[float]:
        ratio = (x - start_x) / (end_x - start_x)
        return [x, start_y + ratio * (end_y - start_y)]

    return [at(min(max(start_x, x0), x1)), at(min(max(end_x, x0), x1))]


def clip_baseline_to_x_span(
    baseline: list[list[float]] | None,
    x0: float,
    x1: float,
) -> list[list[float]]:
    """The part of a baseline polyline that falls inside ``[x0, x1]``.

    Endpoints are interpolated rather than snapped, so the clipped baseline
    still follows the slope of the original instead of flattening at the cut.
    Returns an empty list when nothing of the baseline reaches the span, which
    the caller reads as "this piece of the line has no baseline of its own".
    """
    if not baseline or len(baseline) < 2 or x1 < x0:
        return []

    clipped: list[list[float]] = []
    # Deliberately not strict: this is the pairwise-window idiom, where the two
    # operands differ in length by exactly one and the last point of `baseline`
    # has no successor to pair with. `strict=True` would raise on every call.
    for start, end in zip(baseline, baseline[1:], strict=False):
        if len(start) != 2 or len(end) != 2:
            continue
        for point in _clip_segment_to_x_span(start, end, x0, x1):
            if not clipped or distance(clipped[-1], point) > 1e-9:
                clipped.append(point)
    return clipped if len(clipped) >= 2 else []


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def clean_polygon(
    points: list[list[float]],
    *,
    min_distance: float,
    min_vertices: int = 3,
) -> list[list[float]]:
    if len(points) < 2:
        return points

    working = [[float(point[0]), float(point[1])] for point in points]
    if len(working) >= 2 and distance(working[0], working[-1]) <= min_distance:
        working.pop()

    if len(working) < 2:
        return points

    cleaned: list[list[float]] = [working[0]]
    for point in working[1:]:
        if distance(point, cleaned[-1]) > min_distance:
            cleaned.append(point)

    if len(cleaned) >= 2 and distance(cleaned[0], cleaned[-1]) <= min_distance:
        cleaned.pop()

    return cleaned if len(cleaned) >= min_vertices else points


def mask_from_polygon(
    points: list[list[float]],
    *,
    width: int,
    height: int,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Rasterize a polygon into a ``height x width`` mask.

    ``origin`` shifts the polygon so the mask can cover a window of the page
    rather than the page. It must be integral: vertices are rounded to whole
    pixels before the shift, so an integer offset translates the raster exactly
    and a fractional one would not.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    offset = np.array(origin, dtype=np.int32).reshape(1, 1, 2)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32) - offset], 255)
    return mask


def mask_window(
    polygons: tuple[list[list[float]], ...],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """The page-clipped bounding window that holds every given polygon.

    Comparing two line polygons only needs the pixels either of them can
    occupy. Everything outside this window is zero in both masks, so it can
    neither add to the intersection nor to the union. Returns ``None`` when the
    window is empty - no points, or nothing of them on the page.
    """
    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    if not xs or not ys:
        return None

    x0 = max(0, int(math.floor(min(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    x1 = min(width, int(math.ceil(max(xs))) + 1)
    y1 = min(height, int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_on = a > 0
    b_on = b > 0
    union = np.logical_or(a_on, b_on).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_on, b_on).sum() / union)


def baseline_inside_polygon(
    polygon: list[list[float]],
    baseline: list[list[float]] | None,
) -> bool:
    if not baseline:
        return True

    import cv2

    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return all(
        cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
        for point in baseline
        if len(point) == 2
    )


def approx_polygon(points: list[list[float]], *, epsilon: float) -> list[list[float]]:
    import cv2

    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    if simplified is None or len(simplified) < 3:
        return points
    return [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]


def candidate_quality(
    candidate: list[list[float]],
    reference: list[list[float]],
    *,
    width: int,
    height: int,
    bbox_tolerance: float,
    baseline: list[list[float]] | None,
) -> tuple[bool, dict[str, float]]:
    candidate_area = polygon_area(candidate)
    reference_area = polygon_area(reference)
    area_ratio = candidate_area / reference_area if reference_area > 0 else 0.0

    # Rasterize into the window the two polygons share, not the page. This runs
    # up to twenty times per line and forty-odd times per page: on a 4000x6000
    # scan, page-sized masks meant sixteen hundred allocations of a 24 MB array
    # to compare two shapes a few thousand pixels across. The window is
    # page-clipped, so ``fillPoly``'s own clipping still decides what counts.
    window = mask_window((candidate, reference), width=width, height=height)
    if window is None:
        iou = 0.0
    else:
        x0, y0, x1, y1 = window
        origin = (x0, y0)
        candidate_mask = mask_from_polygon(candidate, width=x1 - x0, height=y1 - y0, origin=origin)
        reference_mask = mask_from_polygon(reference, width=x1 - x0, height=y1 - y0, origin=origin)
        iou = mask_iou(candidate_mask, reference_mask)

    cx0, cy0, cx1, cy1 = bbox(candidate)
    rx0, ry0, rx1, ry1 = bbox(reference)
    contains_bbox = (
        cx0 <= rx0 + bbox_tolerance
        and cy0 <= ry0 + bbox_tolerance
        and cx1 >= rx1 - bbox_tolerance
        and cy1 >= ry1 - bbox_tolerance
    )
    baseline_inside = baseline_inside_polygon(candidate, baseline)

    return contains_bbox and baseline_inside, {
        "simplification_iou": iou,
        "area_ratio": area_ratio,
        "bbox_tolerance": bbox_tolerance,
    }


def simplify_with_quality_gate(
    contour: list[list[float]],
    *,
    width: int,
    height: int,
    target_max_points: int,
    min_iou: float,
    min_area_ratio: float,
    baseline: list[list[float]] | None,
) -> tuple[list[list[float]], dict[str, Any]]:
    height_px = line_height(contour)
    min_spacing = max(MIN_VERTEX_SPACING_PX, 0.02 * height_px)
    reference = clean_polygon(contour, min_distance=min_spacing)
    epsilon = max(1.5, 0.02 * height_px)
    last_valid = reference
    last_metrics: dict[str, Any] = {
        "simplification_iou": 1.0,
        "area_ratio": 1.0,
        "epsilon": 0.0,
    }
    status = "unchanged"

    for _ in range(20):
        candidate = approx_polygon(reference, epsilon=epsilon)
        candidate = clean_polygon(candidate, min_distance=min_spacing)
        shape_ok, metrics = candidate_quality(
            candidate,
            reference,
            width=width,
            height=height,
            bbox_tolerance=max(1.0, epsilon),
            baseline=baseline,
        )
        passes = (
            metrics["simplification_iou"] >= min_iou
            and metrics["area_ratio"] >= min_area_ratio
            and shape_ok
        )
        if not passes:
            status = "quality_gate_stopped"
            break

        last_valid = candidate
        last_metrics = {**metrics, "epsilon": epsilon}
        status = "simplified"
        if len(candidate) <= target_max_points:
            break
        epsilon *= 1.35

    if len(last_valid) > target_max_points and status == "simplified":
        status = "max_points_not_reached"

    return last_valid, {
        **last_metrics,
        "min_vertex_spacing": min_spacing,
        "simplification_status": status,
    }


def simplify_blla_boundary(
    points: list[list[float]],
    *,
    min_distance: float = MIN_VERTEX_SPACING_PX,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Lightweight Douglas-Peucker + spacing cleanup for raw BLLA boundaries."""
    if len(points) < 4:
        return points, {
            "simplification_status": "unchanged",
            "raw_point_count": len(points),
            "simplified_point_count": len(points),
        }

    height_px = line_height(points)
    spacing = max(min_distance, 0.02 * height_px)
    reference = clean_polygon(points, min_distance=spacing)
    epsilon = max(2.0, 0.015 * height_px)
    simplified = approx_polygon(reference, epsilon=epsilon)
    simplified = clean_polygon(simplified, min_distance=spacing)
    if len(simplified) < 4:
        simplified = reference if len(reference) >= 4 else points
    status = "blla_boundary_simplified" if simplified != points else "blla_boundary_unchanged"

    return simplified, {
        "simplification_status": status,
        "raw_point_count": len(points),
        "simplified_point_count": len(simplified),
        "min_vertex_spacing": spacing,
        "epsilon": epsilon,
    }
