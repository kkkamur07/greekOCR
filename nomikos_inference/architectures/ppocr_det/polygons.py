"""Polygon algebra for the PP-OCRv6 poly output mode.

Everything here runs on plain rings (lists of ``[x, y]``) with OpenCV and
pyclipper only, so the poly carriage in refinement and response never needs
a second geometry engine. The quad decision path in ``refinement.py`` is
untouched; these helpers only combine, cut, simplify and measure the
polygons that the served masks are drawn from.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyclipper

#: Hard cap on served polygon points; the simplifier raises epsilon until
#: the ring fits.
MAX_POLYGON_POINTS = 64

#: Tolerance for dropping collinear baseline samples, in pixels.
BASELINE_COLLINEAR_PX = 0.5


def _ring(points: object) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def ring_area(points: object) -> float:
    """Absolute area of a ring, 0 for degenerate input."""
    ring = _ring(points)
    if len(ring) < 3:
        return 0.0
    return abs(float(cv2.contourArea(ring.astype(np.float32))))


def dedup_ring(points: object) -> list[list[float]]:
    """Drop repeated consecutive points and the closing duplicate."""
    ring = _ring(points).tolist()
    out: list[list[float]] = []
    for point in ring:
        pair = [float(point[0]), float(point[1])]
        if not out or pair != out[-1]:
            out.append(pair)
    while len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def clip_ring_to_page(
    points: object, page_width: float | None, page_height: float | None
) -> list[list[float]]:
    """Clamp a ring inside the page; unclamped axes pass through."""
    ring = dedup_ring(points)
    if page_width is None and page_height is None:
        return ring
    clipped = []
    for x, y in ring:
        if page_width is not None:
            x = min(max(x, 0.0), float(page_width))
        if page_height is not None:
            y = min(max(y, 0.0), float(page_height))
        clipped.append([x, y])
    return dedup_ring(clipped)


def union_outlines(outlines: list[object]) -> list[list[float]] | None:
    """pyclipper union of member polygons; one path or ``None``.

    Touching members fuse into a single path, which the caller keeps.
    Separate fragments stay several paths (``None``) so the caller can use
    the convex hull instead.
    """
    clipper = pyclipper.Pyclipper()
    for outline in outlines:
        ring = _ring(outline)
        if len(ring) >= 3 and ring_area(ring) > 0:
            clipper.AddPath(ring, pyclipper.PT_SUBJECT, True)
    try:
        solution = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    except pyclipper.ClipperException:
        return None
    paths = [_ring(path) for path in solution if len(_ring(path)) >= 3 and ring_area(path) > 0]
    if len(paths) != 1:
        return None
    return dedup_ring(paths[0])


def intersect_outline(outline: object, rect: list[list[float]]) -> list[list[float]] | None:
    """Clip a polygon with a rectangle; largest piece or ``None``."""
    ring = _ring(outline)
    if len(ring) < 3 or ring_area(ring) <= 0:
        return None
    clipper = pyclipper.Pyclipper()
    clipper.AddPath(ring, pyclipper.PT_SUBJECT, True)
    clipper.AddPath(_ring(rect), pyclipper.PT_CLIP, True)
    try:
        solution = clipper.Execute(
            pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO
        )
    except pyclipper.ClipperException:
        return None
    paths = [_ring(path) for path in solution if len(_ring(path)) >= 3 and ring_area(path) > 0]
    if not paths:
        return None
    best = max(paths, key=ring_area)
    return dedup_ring(best)


def convex_hull_of(points: object) -> list[list[float]]:
    """Convex hull of a point cloud, as a deduped ring."""
    cloud = _ring(points).astype(np.float32)
    if len(cloud) < 3:
        return dedup_ring(cloud)
    hull = cv2.convexHull(cloud, clockwise=False).reshape(-1, 2)
    return dedup_ring(hull)


def simplify_outline(
    outline: object,
    *,
    median_height: float,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[list[float]] | None:
    """Douglas-Peucker a polygon for serving; ``None`` when unusable.

    Epsilon is ``max(1.0 px, 0.01 * median page line height)``, raised until
    the ring fits ``MAX_POLYGON_POINTS``. The result is deduped, clipped to
    the page, and checked with ``SimplifyPolygon`` keeping the largest
    piece; anything under 4 points is ``None`` so the caller falls back to
    the quad.
    """
    ring = _ring(outline).astype(np.float32)
    if len(ring) < 3:
        return None
    epsilon = max(1.0, 0.01 * float(median_height))
    approx = ring
    for _ in range(25):
        approx = cv2.approxPolyDP(ring, epsilon, True).reshape(-1, 2)
        if len(approx) <= MAX_POLYGON_POINTS:
            break
        epsilon *= 1.5
    cleaned = clip_ring_to_page(approx, page_width, page_height)
    if len(cleaned) < 3:
        return None
    try:
        pieces = pyclipper.SimplifyPolygon(cleaned, True)
    except pyclipper.ClipperException:
        return None
    candidates = [
        _ring(piece) for piece in pieces if len(_ring(piece)) >= 3 and ring_area(piece) > 0
    ]
    if not candidates:
        return None
    best = dedup_ring(max(candidates, key=ring_area))
    if len(best) < 4:
        return None
    return best


def _long_axis(quad_ring: object) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Principal axis of a quad ring: origin, unit vector, min and max u."""
    ring = _ring(quad_ring)
    origin = ring.mean(axis=0)
    centred = ring - origin
    cov = (centred.T @ centred) / max(len(ring), 1)
    _, vecs = np.linalg.eigh(cov)
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0]) if norm <= 0 else axis / norm
    proj = (ring - origin) @ axis
    return origin, axis, float(proj.min()), float(proj.max())


def polyline_baseline(
    outline: object,
    quad_ring: object,
    fraction: float,
    *,
    samples: int | None = None,
) -> list[list[float]] | None:
    """Baseline polyline across a polygon along the quad long axis.

    ``samples`` (8 to 16) x positions span the quad axis; at each, the
    polygon's upper and lower boundary give a point at ``fraction`` between
    them. Collinear runs collapse, so a straight line reduces to two points
    on the quad baseline. Returns ``None`` when the polygon has no usable
    crossings.
    """
    ring = _ring(outline)
    if len(ring) < 3 or ring_area(ring) <= 0:
        return None
    origin, axis, _, _ = _long_axis(quad_ring)
    normal = np.array([-axis[1], axis[0]])
    if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
        normal = -normal
    rel = ring - origin
    outline_u = rel @ axis
    outline_v = rel @ normal
    quad_u = (_ring(quad_ring) - origin) @ axis
    low, high = float(quad_u.min()), float(quad_u.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    count = samples or min(16, max(8, int(round((high - low) / 16.0))))
    count = min(16, max(8, count))
    edges = [
        (outline_u[i], outline_v[i], outline_u[(i + 1) % len(ring)], outline_v[(i + 1) % len(ring)])
        for i in range(len(ring))
    ]
    points: list[list[float]] = []
    for step in range(count):
        pos = low + (high - low) * (step + 0.5) / count
        crossings: list[float] = []
        for u1, v1, u2, v2 in edges:
            if (u1 <= pos < u2) or (u2 <= pos < u1):
                crossings.append(float(v1 + (pos - u1) * (v2 - v1) / (u2 - u1)))
            elif u1 == pos and u2 == pos:
                crossings.extend([float(v1), float(v2)])
        if len(crossings) < 2:
            continue
        top, bottom = min(crossings), max(crossings)
        at = top + float(fraction) * (bottom - top)
        point = origin + pos * axis + at * normal
        points.append([float(point[0]), float(point[1])])
    if len(points) < 2:
        return None
    return _drop_collinear(points)


def _drop_collinear(points: list[list[float]]) -> list[list[float]]:
    """Greedily drop points within half a pixel of their chord."""
    kept = [points[0]]
    for point in points[1:]:
        kept.append(point)
        while len(kept) >= 3:
            first = np.asarray(kept[-3], dtype=np.float64)
            middle = np.asarray(kept[-2], dtype=np.float64)
            last = np.asarray(kept[-1], dtype=np.float64)
            leg = last - first
            length = float(np.linalg.norm(leg))
            if length <= 0:
                kept.pop(-2)
                continue
            distance = (
                abs(float(leg[0] * (first[1] - middle[1]) - leg[1] * (first[0] - middle[0])))
                / length
            )
            if distance <= BASELINE_COLLINEAR_PX:
                kept.pop(-2)
            else:
                break
    return kept


__all__ = [
    "BASELINE_COLLINEAR_PX",
    "MAX_POLYGON_POINTS",
    "clip_ring_to_page",
    "convex_hull_of",
    "dedup_ring",
    "intersect_outline",
    "polyline_baseline",
    "ring_area",
    "simplify_outline",
    "union_outlines",
]
