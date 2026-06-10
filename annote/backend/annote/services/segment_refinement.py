"""Segment refinement — shrink Kraken polygons with scikit-image active contours."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from annote.services.processing.polygon import merge_close_polygon_points

REFINEMENT_MARGIN_PX = 4.0
SIMPLIFY_TOLERANCE_PX = 2.0
MIN_INK_AREA_FRACTION = 0.12
MIN_LINE_WIDTH_FRACTION = 0.35
SNAKE_SAMPLE_SPACING_PX = 5.0
SNAKE_MAX_ITERATIONS = 250


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _points_inside_polygon(points: list[list[float]], polygon: list[list[float]]) -> bool:
    import cv2

    poly = np.array(polygon, dtype=np.float32)
    for x, y in points:
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0:
            return False
    return True


def _simplify_polygon(points: list[list[float]], *, tolerance: float) -> list[list[float]]:
    import cv2

    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, tolerance, closed=True)
    if simplified is None or len(simplified) < 3:
        return points
    return [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]


def _resample_closed_polygon(
    points: list[list[float]],
    *,
    spacing: float = SNAKE_SAMPLE_SPACING_PX,
) -> np.ndarray:
    if len(points) < 3:
        return np.array(points, dtype=np.float64)

    closed = [list(p) for p in points]
    if math.hypot(closed[0][0] - closed[-1][0], closed[0][1] - closed[-1][1]) > 1e-3:
        closed.append(closed[0])

    cumulative = [0.0]
    for i in range(1, len(closed)):
        cumulative.append(
            cumulative[-1]
            + math.hypot(closed[i][0] - closed[i - 1][0], closed[i][1] - closed[i - 1][1])
        )

    perimeter = cumulative[-1]
    if perimeter <= spacing:
        return np.array(points, dtype=np.float64)

    sample_count = max(int(math.ceil(perimeter / spacing)), 12)
    targets = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    sampled: list[list[float]] = []
    segment_index = 1
    for target in targets:
        while segment_index < len(cumulative) - 1 and cumulative[segment_index] < target:
            segment_index += 1
        segment_length = cumulative[segment_index] - cumulative[segment_index - 1]
        if segment_length <= 1e-6:
            sampled.append(closed[segment_index])
            continue
        ratio = (target - cumulative[segment_index - 1]) / segment_length
        x0, y0 = closed[segment_index - 1]
        x1, y1 = closed[segment_index]
        sampled.append([x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio])
    return np.array(sampled, dtype=np.float64)


def _contour_to_points(contour: np.ndarray, *, x0: int, y0: int) -> list[list[float]]:
    return [[float(pt[0][0] + x0), float(pt[0][1] + y0)] for pt in contour]


def _largest_contour_polygon(
    mask: np.ndarray,
    *,
    x0: int,
    y0: int,
) -> list[list[float]] | None:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 8:
        return None
    return _contour_to_points(best, x0=x0, y0=y0)


def _line_close_kernel_width(mask: np.ndarray) -> int:
    import cv2

    cols = [x for x in range(mask.shape[1]) if cv2.countNonZero(mask[:, x : x + 1]) > 0]
    if not cols:
        return 5
    width = cols[-1] - cols[0] + 1
    return max(min(width // 12, 31), 5)


def _snake_points_inside_mask(points: list[list[float]], mask: np.ndarray, *, x0: int, y0: int) -> bool:
    h, w = mask.shape
    for x, y in points:
        xi = int(round(x - x0))
        yi = int(round(y - y0))
        if xi < 0 or yi < 0 or xi >= w or yi >= h or mask[yi, xi] == 0:
            return False
    return True


def _project_points_to_mask(
    points: list[list[float]],
    mask: np.ndarray,
    *,
    x0: int,
    y0: int,
) -> list[list[float]] | None:
    import cv2

    valid_pixels = cv2.findNonZero(mask)
    if valid_pixels is None:
        return None

    valid_xy = valid_pixels.reshape(-1, 2).astype(np.float64)
    h, w = mask.shape
    projected: list[list[float]] = []
    for x, y in points:
        local_x = float(x - x0)
        local_y = float(y - y0)
        xi = int(round(local_x))
        yi = int(round(local_y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
            projected.append([x, y])
            continue

        deltas = valid_xy - np.array([local_x, local_y], dtype=np.float64)
        nearest = valid_xy[int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))]
        projected.append([float(nearest[0] + x0), float(nearest[1] + y0)])

    return projected


def _active_contour_refine(
    edge_image: np.ndarray,
    seed_points: list[list[float]],
    valid_mask: np.ndarray,
    *,
    x0: int,
    y0: int,
) -> list[list[float]] | None:
    try:
        from skimage.segmentation import active_contour
    except ImportError:
        return None

    sampled = _resample_closed_polygon([[p[0] - x0, p[1] - y0] for p in seed_points])
    if len(sampled) < 3:
        return None

    # skimage snakes use (row, col) coordinates, while annote stores (x, y).
    snake = np.column_stack((sampled[:, 1], sampled[:, 0]))
    evolved = active_contour(
        edge_image,
        snake,
        alpha=0.01,
        beta=1.0,
        w_line=0.0,
        w_edge=1.0,
        gamma=0.01,
        max_px_move=1.0,
        max_num_iter=SNAKE_MAX_ITERATIONS,
        convergence=0.05,
        boundary_condition="periodic",
    )
    points = [[float(col + x0), float(row + y0)] for row, col in evolved]
    if _snake_points_inside_mask(points, valid_mask, x0=x0, y0=y0):
        return points
    return _project_points_to_mask(points, valid_mask, x0=x0, y0=y0)


def refine_segment(
    image: Image.Image,
    ceiling: list[list[float]],
    *,
    margin_px: float = REFINEMENT_MARGIN_PX,
) -> list[list[float]]:
    """Shrink a Kraken ceiling polygon with a scikit-image active contour.

    Refinement seeds a snake from the ink envelope plus margin_px, evolves it
    against the page edge signal, and keeps it inside the eroded Kraken ceiling.
    Falls back to the seed or merged ceiling on failure.
    """
    if len(ceiling) < 3:
        return ceiling

    fallback = merge_close_polygon_points(ceiling)

    try:
        import cv2
    except ImportError:
        return fallback

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    ceiling_arr = np.array(ceiling, dtype=np.float32)
    xs, ys = ceiling_arr[:, 0], ceiling_arr[:, 1]
    pad = int(math.ceil(margin_px)) + 4
    x0 = max(int(math.floor(xs.min())) - pad, 0)
    y0 = max(int(math.floor(ys.min())) - pad, 0)
    x1 = min(int(math.ceil(xs.max())) + pad, w)
    y1 = min(int(math.ceil(ys.max())) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return fallback

    crop = rgb[y0:y1, x0:x1]
    shifted = ceiling_arr.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0

    ceiling_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(ceiling_mask, [shifted.astype(np.int32)], 255)
    margin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(int(margin_px * 2) + 1, 3), max(int(margin_px * 2) + 1, 3)),
    )
    inner_mask = cv2.erode(ceiling_mask, margin_kernel, iterations=1)
    inner_area = cv2.countNonZero(inner_mask)
    if inner_area == 0:
        return fallback

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_masked = cv2.bitwise_and(ink, ceiling_mask)

    bridge_w = _line_close_kernel_width(ceiling_mask)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 3))
    ink_bridged = cv2.morphologyEx(ink_masked, cv2.MORPH_CLOSE, bridge_kernel)
    ink_bridged = cv2.bitwise_and(ink_bridged, ceiling_mask)

    ink_area = cv2.countNonZero(ink_bridged)
    if ink_area < inner_area * MIN_INK_AREA_FRACTION:
        return fallback

    # Contract from the Kraken ceiling to ink, then expand back out by the
    # refinement margin. This seeds skimage's active contour close to the
    # desired stroke envelope instead of asking the snake to travel from Kraken.
    refined_mask = cv2.dilate(ink_bridged, margin_kernel, iterations=1)
    refined_mask = cv2.bitwise_and(refined_mask, inner_mask)
    seed_points = _largest_contour_polygon(refined_mask, x0=x0, y0=y0)
    if seed_points is None:
        return fallback

    edge_image = cv2.Canny(blurred, 40, 120).astype(np.float64) / 255.0
    edge_image = cv2.GaussianBlur(edge_image, (0, 0), 1.5)
    edge_image[inner_mask == 0] = 0.0
    page_points = (
        _active_contour_refine(edge_image, seed_points, inner_mask, x0=x0, y0=y0)
        or seed_points
    )
    page_points = merge_close_polygon_points(page_points)
    page_points = _simplify_polygon(page_points, tolerance=SIMPLIFY_TOLERANCE_PX)

    if len(page_points) < 3 or not _points_inside_polygon(page_points, ceiling):
        return fallback

    ceiling_w = float(xs.max() - xs.min())
    ceiling_h = float(ys.max() - ys.min())
    rx0, _, rx1, _ = _bbox(page_points)
    refined_w = rx1 - rx0
    if ceiling_w > ceiling_h * 1.5 and refined_w < ceiling_w * MIN_LINE_WIDTH_FRACTION:
        return fallback

    if ink_area > 0 and _polygon_area(page_points) < ink_area * 0.25:
        return fallback

    if _polygon_area(page_points) < _polygon_area(fallback) * 0.08:
        return fallback

    return page_points


def refine_kraken_segments(image: Image.Image, segments: list) -> list:
    """Auto-refine Kraken segments and stamp source / kraken_ceiling metadata."""
    refined_segments = []
    for segment in segments:
        ceiling = [list(point) for point in segment.points]
        points = refine_segment(image, ceiling)
        refined_segments.append(
            segment.model_copy(
                update={
                    "source": "kraken",
                    "kraken_ceiling": ceiling,
                    "points": points,
                }
            )
        )
    return refined_segments
