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


def default_baseline(points: list[list[float]]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox(points)
    y = (y0 + y1) / 2.0
    return [[x0, y], [x1, y]]


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
) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255)
    return mask


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

    candidate_mask = mask_from_polygon(candidate, width=width, height=height)
    reference_mask = mask_from_polygon(reference, width=width, height=height)
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


def simplify_kraken_boundary(
    points: list[list[float]],
    *,
    min_distance: float = MIN_VERTEX_SPACING_PX,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Lightweight Douglas-Peucker + spacing cleanup for raw Kraken boundaries."""
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
    status = (
        "kraken_boundary_simplified"
        if simplified != points
        else "kraken_boundary_unchanged"
    )

    return simplified, {
        "simplification_status": status,
        "raw_point_count": len(points),
        "simplified_point_count": len(simplified),
        "min_vertex_spacing": spacing,
        "epsilon": epsilon,
    }
