"""Helpers for line baseline and mask geometry on document lines."""

from __future__ import annotations


def geometry_points(geometry: dict | None) -> list[list[float]]:
    if not geometry:
        return []
    raw = geometry.get("points")
    if isinstance(raw, list):
        return _normalize_point_list(raw)
    raw = geometry.get("coordinates")
    if isinstance(raw, list):
        return _normalize_point_list(raw)
    return []


def _normalize_point_list(raw: list[object]) -> list[list[float]]:
    points: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def _points_close(a: list[float], b: list[float], *, tolerance: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def baseline_matches_polygon(
    baseline_points: list[list[float]], polygon_points: list[list[float]]
) -> bool:
    """True when baseline is just the segment polygon copied verbatim (not a text line)."""
    if len(baseline_points) != len(polygon_points):
        return False
    return all(
        _points_close(baseline_points[index], polygon_points[index])
        for index in range(len(polygon_points))
    )


def default_baseline_from_polygon(points: list[list[float]]) -> list[list[float]]:
    """Derive a horizontal text baseline from the bottom edge of a segment polygon."""
    if len(points) < 2:
        return points
    max_y = max(point[1] for point in points)
    bottom = [point for point in points if point[1] >= max_y - 1e-6]
    bottom.sort(key=lambda point: point[0])
    if len(bottom) >= 2:
        return [bottom[0], bottom[-1]]
    ordered = sorted(points, key=lambda point: (-point[1], point[0]))
    left = ordered[0]
    right = max(ordered[:2], key=lambda point: point[0])
    return [left, right]


def resolve_line_baseline_and_mask(
    *,
    points: list[list[float]],
    payload_baseline: dict | None,
    payload_mask: dict | None,
    existing_baseline: dict | None,
    existing_mask: dict | None,
) -> tuple[dict, dict]:
    if payload_baseline is not None:
        baseline = payload_baseline
    else:
        existing_baseline_points = geometry_points(existing_baseline)
        if len(existing_baseline_points) >= 2 and not baseline_matches_polygon(
            existing_baseline_points, points
        ):
            baseline = existing_baseline or {"points": default_baseline_from_polygon(points)}
        else:
            baseline = {"points": default_baseline_from_polygon(points)}

    if payload_mask is not None:
        mask = payload_mask
    elif existing_mask is not None and geometry_points(existing_mask):
        mask = existing_mask
    else:
        mask = {"points": points}

    return baseline, mask
