"""Otsu contour extraction inside BLLA ceiling polygons."""

from __future__ import annotations

import math

import numpy as np

from inference.preprocessing.segment_geometry import bbox, mask_from_polygon


def combine_contours(contours: list[list[list[float]]]) -> list[list[float]]:
    import cv2

    if len(contours) == 1:
        contour = np.array(contours[0], dtype=np.float32)
    else:
        contour = cv2.convexHull(
            np.array([point for contour in contours for point in contour], dtype=np.float32)
        )
    return [[float(x), float(y)] for x, y in contour.reshape(-1, 2)]


def cluster_contours_by_vertical_gap(
    contours: list[list[list[float]]],
    *,
    gap_px: float,
) -> list[list[list[list[float]]]]:
    if len(contours) <= 1:
        return [contours] if contours else []

    sorted_contours = sorted(contours, key=lambda contour: bbox(contour)[1])
    clusters: list[list[list[list[float]]]] = []
    current: list[list[list[float]]] = []
    current_y1: float | None = None

    for contour in sorted_contours:
        _, y0, _, y1 = bbox(contour)
        if current and current_y1 is not None and y0 > current_y1 + gap_px:
            clusters.append(current)
            current = []
            current_y1 = None
        current.append(contour)
        current_y1 = y1 if current_y1 is None else max(current_y1, y1)

    if current:
        clusters.append(current)
    return clusters


def otsu_band_contours(
    gray: np.ndarray,
    ceiling: list[list[float]],
    *,
    margin_px: float,
) -> list[list[list[float]]]:
    import cv2

    height, width = gray.shape
    # Rasterize the ceiling into its own page-clipped window rather than a
    # full-page mask: a line ceiling spans a few thousand pixels on a page of
    # millions, and a full-page mask was allocated and scanned once per line.
    xs = [point[0] for point in ceiling]
    ys = [point[1] for point in ceiling]
    if not xs or not ys:
        return []

    x0 = max(0, int(math.floor(min(xs))) - 2)
    y0 = max(0, int(math.floor(min(ys))) - 2)
    x1 = min(width, int(math.ceil(max(xs))) + 3)
    y1 = min(height, int(math.ceil(max(ys))) + 3)
    if x1 <= x0 or y1 <= y0:
        return []

    crop_mask = mask_from_polygon(ceiling, width=x1 - x0, height=y1 - y0, origin=(x0, y0))
    if cv2.countNonZero(crop_mask) == 0:
        return []

    crop = gray[y0:y1, x0:x1].copy()
    crop[crop_mask == 0] = 255

    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.bitwise_and(ink, crop_mask)
    if cv2.countNonZero(ink) == 0:
        return []

    kernel_size = max(1, int(round(margin_px * 2)) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    band = cv2.bitwise_and(cv2.dilate(ink, kernel), crop_mask)
    contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    page_contours: list[list[list[float]]] = []
    for contour in contours:
        shifted = contour.reshape(-1, 2).astype(float)
        shifted[:, 0] += x0
        shifted[:, 1] += y0
        page_contours.append([[float(x), float(y)] for x, y in shifted])
    return page_contours
