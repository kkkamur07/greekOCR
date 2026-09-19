"""DB postprocess for the PP-OCRv6 detection graph, quad mode.

Mirrors PaddleX 3.7.0
``paddlex/inference/models/text_detection/processors.py::DBPostProcess`` in
quad mode (``box_type="quad"``, the default this model's config leaves
unset), which is ``boxes_from_bitmap``: binarise at ``thresh``,
``cv2.findContours`` over at most ``max_candidates`` contours,
``get_mini_boxes`` with a minimum side of 3, ``box_score_fast`` gated at
``box_thresh``, unclip expansion, a second ``get_mini_boxes`` with a minimum
side of 5, then scaling back to source coordinates with rounding and
clipping. ``score_mode`` is the default ``"fast"`` and ``use_dilation`` is
unset (False), so neither branch is reproduced here.

The ``poly`` branch (``polygons_from_bitmap`` with ``approxPolyDP``) is
deliberately not implemented: the config selects quad mode and the segment
contract carries four corner quads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

# PaddleX's ``min_size`` (3) and ``min_size + 2`` (5): the pre-expansion floor
# drops specks, the post-expansion floor drops boxes the expansion failed to
# grow (degenerate contours whose offset collapses).
PPOCR_MIN_SIDE = 3
PPOCR_MIN_EXPANDED_SIDE = PPOCR_MIN_SIDE + 2


@dataclass(frozen=True)
class DetectedQuad:
    """One detection in original image coordinates."""

    points: list[list[float]]
    score: float


def unclip(points: np.ndarray, ratio: float) -> np.ndarray | None:
    """Expand a contour by ``area * ratio / perimeter`` with round joins.

    PaddleX implements this with pyclipper (``JT_ROUND``); this uses
    ``shapely``'s ``buffer`` with round joins instead, so no new dependency
    is needed. The function is kept isolated (single input, single output)
    so it can be swapped for pyclipper if the parity measurement ever shows
    more than 2 px of corner error. Returns ``None`` when the offset
    collapses or splits, which the caller treats as a dropped candidate.
    """

    contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(contour) < 3:
        return None
    # Area and perimeter come from OpenCV exactly as in PaddleX, so the
    # offset distance matches even though the offsetting itself is shapely.
    # (OpenCV needs float32 here, while the buffering below uses float64.)
    area = float(cv2.contourArea(contour))
    length = float(cv2.arcLength(contour, True))
    if length <= 0:
        return None
    distance = area * ratio / length
    if distance <= 0:
        return None
    polygon = Polygon(np.asarray(contour, dtype=np.float64))
    if not polygon.is_valid or polygon.area <= 0:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
    expanded = polygon.buffer(distance, join_style="round")
    if expanded.is_empty or isinstance(expanded, MultiPolygon):
        return None
    coords = np.asarray(expanded.exterior.coords, dtype=np.float64)
    if len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    if len(coords) < 3:
        return None
    return coords


def get_mini_boxes(contour: np.ndarray) -> tuple[list[list[float]], float]:
    """Minimum-area rectangle of a contour, ordered like PaddleX.

    Returns the four corners (left pair then right pair, top first within
    each pair) and the shorter side of the rectangle.
    """

    bounding = cv2.minAreaRect(np.asarray(contour, dtype=np.float32))
    corners = sorted((list(point) for point in cv2.boxPoints(bounding)), key=lambda point: point[0])
    if corners[1][1] > corners[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if corners[3][1] > corners[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [corners[index_1], corners[index_2], corners[index_3], corners[index_4]]
    return box, float(min(bounding[1]))


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """Mean probability inside a quad, via a filled mask over its bbox."""

    values = np.asarray(box, dtype=np.float64).reshape(-1, 2).copy()
    height, width = bitmap.shape[:2]
    xmin = max(0, min(math.floor(float(values[:, 0].min())), width - 1))
    xmax = max(0, min(math.ceil(float(values[:, 0].max())), width - 1))
    ymin = max(0, min(math.floor(float(values[:, 1].min())), height - 1))
    ymax = max(0, min(math.ceil(float(values[:, 1].max())), height - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    values[:, 0] -= xmin
    values[:, 1] -= ymin
    cv2.fillPoly(mask, [values.astype(np.int32)], (1,))
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def order_quad_clockwise(points: np.ndarray) -> np.ndarray:
    """Order four corners clockwise starting from the top left.

    PaddleX returns ``get_mini_boxes`` order directly; the segment contract
    wants a canonical start corner, so the sums and differences method picks
    top left (smallest x plus y), bottom right, top right and bottom left,
    with a shoelace guard for the orientation.
    """

    corners = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    sums = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 0] - corners[:, 1]
    top_left = int(np.argmin(sums))
    bottom_right = int(np.argmax(sums))
    top_right = int(np.argmax(diffs))
    bottom_left = int(np.argmin(diffs))
    ordered = corners[[top_left, top_right, bottom_right, bottom_left]]
    # In image coordinates (y down) a visually clockwise ring has positive
    # signed area; a mirrored pick is repaired by reversing past the start.
    area = float(
        np.sum(
            ordered[:, 0] * np.roll(ordered[:, 1], -1) - np.roll(ordered[:, 0], -1) * ordered[:, 1]
        )
    )
    if area < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def detect_lines(
    prob_map: np.ndarray,
    *,
    orig_width: int,
    orig_height: int,
    ratio_h: float,
    ratio_w: float,
    thresh: float = 0.2,
    box_thresh: float = 0.45,
    unclip_ratio: float = 1.4,
    max_candidates: int = 3000,
) -> list[DetectedQuad]:
    """Run the DB quad postprocess over one probability map.

    ``prob_map`` is the ``(H, W)`` detector output in resized-image space;
    the returned quads are in original image coordinates, clockwise from the
    top left, each with its ``box_score_fast`` score.
    """

    pred = np.asarray(prob_map, dtype=np.float32)
    if pred.ndim != 2:
        raise ValueError("PP-OCRv6 det probability map must have shape (H, W)")
    height, width = pred.shape
    if height <= 0 or width <= 0:
        raise ValueError("PP-OCRv6 det probability map must not be empty")
    if ratio_h <= 0 or ratio_w <= 0:
        raise ValueError("PP-OCRv6 det resize ratios must be positive")

    bitmap = pred > thresh
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    width_scale = orig_width / width
    height_scale = orig_height / height

    quads: list[DetectedQuad] = []
    for contour in contours[:max_candidates]:
        box, short_side = get_mini_boxes(contour)
        if short_side < PPOCR_MIN_SIDE:
            continue
        score = box_score_fast(pred, np.asarray(box, dtype=np.float64))
        if score < box_thresh:
            continue
        expanded = unclip(np.asarray(box, dtype=np.float64), unclip_ratio)
        if expanded is None:
            continue
        grown, grown_side = get_mini_boxes(expanded)
        if grown_side < PPOCR_MIN_EXPANDED_SIDE:
            continue
        ordered = order_quad_clockwise(np.asarray(grown, dtype=np.float64))
        mapped = [
            [
                float(min(max(round(float(x) * width_scale), 0), orig_width)),
                float(min(max(round(float(y) * height_scale), 0), orig_height)),
            ]
            for x, y in ordered
        ]
        quads.append(DetectedQuad(points=mapped, score=score))
    return quads


__all__ = [
    "DetectedQuad",
    "box_score_fast",
    "detect_lines",
    "get_mini_boxes",
    "order_quad_clockwise",
    "unclip",
]
