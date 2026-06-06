"""Polygon point utilities."""

from __future__ import annotations

import math

# Empirical default from manuscript page analysis (Grec_1360…_6.json, 31 segments):
# consecutive edge lengths p10 ≈ 6.7 px; 13% of edges ≤ 8 px.
DUPLICATE_VERTEX_SPACING = 6.0


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def merge_close_polygon_points(
    points: list[list[float]],
    *,
    min_distance: float = DUPLICATE_VERTEX_SPACING,
    min_vertices: int = 3,
) -> list[list[float]]:
    """Drop consecutive polygon vertices closer than *min_distance* (image pixels).

    Kraken line boundaries often contain dense runs of nearly coincident points;
    merging them keeps polygons editable without changing the overall shape much.
    """
    if len(points) < 2:
        return points

    working = [list(p) for p in points]

    # Closed polygons sometimes repeat the first vertex at the end.
    if len(working) >= 2 and _distance(working[0], working[-1]) <= min_distance:
        working.pop()

    if len(working) < 2:
        return points

    merged: list[list[float]] = [working[0]]
    for pt in working[1:]:
        if _distance(pt, merged[-1]) > min_distance:
            merged.append(pt)

    if len(merged) >= 2 and _distance(merged[0], merged[-1]) <= min_distance:
        merged.pop()

    if len(merged) < min_vertices:
        return points

    return merged
