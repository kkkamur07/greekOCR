"""Small synthetic-input BLLA decoder used by focused unit tests."""

from __future__ import annotations

import cv2
import numpy as np

from inference.architectures.blla.blla_decoder.common import as_heatmaps
from inference.architectures.blla.blla_decoder.types import DecodedBLLALine


def _resize_heatmaps(heatmaps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.stack(
        [
            cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
            for channel in heatmaps
        ]
    )


def _baseline_points(
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    component_id: int,
) -> list[list[float]]:
    ys, xs = np.where(labels == component_id)
    if len(xs) == 0:
        return []

    points: list[list[float]] = []
    for x in range(int(xs.min()), int(xs.max()) + 1):
        column_ys = np.where(labels[:, x] == component_id)[0]
        if len(column_ys) == 0:
            continue
        weights = np.maximum(baseline_probability[column_ys, x], 1e-6)
        y = float(np.average(column_ys, weights=weights))
        points.append([float(x), y])

    if len(points) < 2:
        return []
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    epsilon = max(1.0, 0.01 * (xs.max() - xs.min() + 1))
    simplified = cv2.approxPolyDP(contour, epsilon, closed=False)
    return [[float(x), float(y)] for x, y in simplified.reshape(-1, 2)]


def _orient_baseline(
    baseline: list[list[float]],
    start_probability: np.ndarray,
    end_probability: np.ndarray,
) -> list[list[float]]:
    if len(baseline) < 2:
        return baseline
    first = tuple(np.rint(baseline[0]).astype(int))
    last = tuple(np.rint(baseline[-1]).astype(int))
    height, width = start_probability.shape

    def marker(probability: np.ndarray, point: tuple[int, int]) -> float:
        x, y = point
        x0, x1 = max(0, x - 10), min(width, x + 11)
        y0, y1 = max(0, y - 10), min(height, y + 11)
        return float(probability[y0:y1, x0:x1].max(initial=0.0))

    first_delta = marker(start_probability, first) - marker(end_probability, first)
    last_delta = marker(start_probability, last) - marker(end_probability, last)
    if first_delta < -0.2 and last_delta > 0.2:
        return list(reversed(baseline))
    if abs(first_delta) <= 0.2 and abs(last_delta) <= 0.2 and baseline[0][0] > baseline[-1][0]:
        return list(reversed(baseline))
    return baseline


def _fallback_polygon(
    baseline: list[list[float]],
    *,
    width: int,
    height: int,
    half_height: float,
) -> list[list[float]]:
    top = [[point[0], max(0.0, point[1] - half_height)] for point in baseline]
    bottom = [
        [point[0], min(float(height - 1), point[1] + half_height)] for point in reversed(baseline)
    ]
    return top + bottom


def _polygon_for_baseline(
    baseline: list[list[float]],
    region_probability: np.ndarray,
    *,
    baseline_ys: list[float],
) -> list[list[float]]:
    height, width = region_probability.shape
    positive = region_probability >= 0.5
    if baseline_ys:
        sorted_ys = sorted(baseline_ys)
        gaps = np.diff(sorted_ys)
        line_gap = float(np.median(gaps[gaps > 2])) if np.any(gaps > 2) else height * 0.12
    else:
        line_gap = height * 0.12
    half_height = max(6.0, min(80.0, line_gap * 0.42))

    top: list[list[float]] = []
    bottom: list[list[float]] = []
    for x_value, y_value in baseline:
        x = min(max(int(round(x_value)), 0), width - 1)
        y = min(max(int(round(y_value)), 0), height - 1)
        column = positive[:, x]
        runs = np.flatnonzero(np.diff(np.r_[False, column, False])).reshape(-1, 2)
        containing = [run for run in runs if run[0] <= y < run[1]]
        if containing:
            y0, y1 = containing[0]
            if y1 - y0 <= max(12, int(half_height * 4)):
                top.append([float(x), float(y0)])
                bottom.append([float(x), float(y1 - 1)])
                continue
        top.append([float(x), max(0.0, y_value - half_height)])
        bottom.append([float(x), min(float(height - 1), y_value + half_height)])

    return top + list(reversed(bottom))


def decode_simple_heatmaps(
    heatmaps: np.ndarray,
    *,
    image_size: tuple[int, int],
    threshold: float,
    min_length: float,
) -> list[DecodedBLLALine]:
    """Decode synthetic probability maps without image-dependent polygonization."""

    width, height = image_size
    resized = _resize_heatmaps(as_heatmaps(heatmaps), (width, height))
    start, end, baseline_probability, region = resized
    ridge = (baseline_probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ridge, connectivity=8)
    candidates: list[list[list[float]]] = []
    for component_id in range(1, count):
        component_width = float(stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = float(stats[component_id, cv2.CC_STAT_HEIGHT])
        if max(component_width, component_height) < min_length:
            continue
        points = _baseline_points(labels, baseline_probability, component_id)
        if len(points) >= 2:
            candidates.append(_orient_baseline(points, start, end))

    candidates.sort(
        key=lambda points: (sum(point[1] for point in points) / len(points), points[0][0])
    )
    baseline_ys = [point[1] for points in candidates for point in points]
    decoded: list[DecodedBLLALine] = []
    for points in candidates:
        length = float(
            sum(
                np.linalg.norm(np.subtract(points[index + 1], points[index]))
                for index in range(len(points) - 1)
            )
        )
        if length < min_length:
            continue
        polygon = _polygon_for_baseline(points, region, baseline_ys=baseline_ys)
        if len(polygon) < 4:
            polygon = _fallback_polygon(
                points,
                width=width,
                height=height,
                half_height=max(6.0, height * 0.03),
            )
        decoded.append(DecodedBLLALine(baseline=points, polygon=polygon))
    return decoded
