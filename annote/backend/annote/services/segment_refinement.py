"""Segment refinement — Kraken ceiling → per-segment Otsu ink + margin."""

from __future__ import annotations

import numpy as np
from PIL import Image

from annote.services.processing.polygon import merge_close_polygon_points

REFINEMENT_MARGIN_PX = 4.0
SIMPLIFY_TOLERANCE_PX = 2.0


def _simplify_polygon(points: list[list[float]], *, tolerance: float) -> list[list[float]]:
    import cv2

    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, tolerance, closed=True)
    if simplified is None or len(simplified) < 3:
        return points
    return [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]


def _segment_contour(
    gray: np.ndarray,
    boundary: list[list[float]],
    margin_px: float,
) -> list[list[float]] | None:
    """Per-segment Otsu + margin — matches experiments/active_countour.ipynb."""
    import cv2

    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(boundary, dtype=np.int32)], 255)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    y0, y1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, h)
    x0, x1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, w)
    crop = gray[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1]
    crop[m == 0] = 255

    _, ink = cv2.threshold(
        cv2.GaussianBlur(crop, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(margin_px * 2) + 1, int(margin_px * 2) + 1))
    band = cv2.bitwise_and(cv2.dilate(cv2.bitwise_and(ink, m), k), m)
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    c[:, 0] += x0
    c[:, 1] += y0
    return [[float(x), float(y)] for x, y in c]


def refine_segment(
    image: Image.Image,
    ceiling: list[list[float]],
    *,
    margin_px: float = REFINEMENT_MARGIN_PX,
) -> list[list[float]]:
    """Kraken polygon → Otsu ink + margin → simplify. Falls back to ceiling if no ink contour."""
    if len(ceiling) < 3:
        return ceiling

    fallback = merge_close_polygon_points(ceiling)

    try:
        import cv2
    except ImportError:
        return fallback

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    page_points = _segment_contour(gray, ceiling, margin_px)
    if page_points is None:
        return fallback

    page_points = merge_close_polygon_points(page_points)
    page_points = _simplify_polygon(page_points, tolerance=SIMPLIFY_TOLERANCE_PX)
    return page_points if len(page_points) >= 3 else fallback


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
