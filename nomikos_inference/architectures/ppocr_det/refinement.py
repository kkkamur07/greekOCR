"""Refine raw PP-OCRv6 detection quads into clean manuscript lines.

Pipeline order inside the adapter: detect quads, merge row fragments,
resolve overlaps, classify suspects, order, build the response. Everything
here is pure geometry relative to the page's own medians and the column
bands that ``reading_order`` builds; no absolute pixel threshold transfers
between manuscripts. Overlap is shared area divided by the smaller
polygon's area: 5 to 7 percent is normal (ascenders, descenders), 20
percent or more is treated as a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout
from nomikos_inference.architectures.ppocr_det.response import synthetic_baseline_points

DEFAULT_MERGE_GAP_RATIO = 1.5
DEFAULT_MERGE_MAX_HEIGHT_RATIO = 2.0
DEFAULT_OVERLAP_CUT_THRESHOLD = 0.20
DUPLICATE_SHARED_RATIO = 0.50
_MAX_OVERLAP_PASSES = 100


@dataclass
class RefinedLine:
    """One manuscript line after refinement, before ordering."""

    points: list[list[float]]
    baseline: list[list[float]]
    score: float
    members: tuple[int, ...] = ()
    role: str = "line"
    suspect: bool = False
    suspect_reason: str = ""
    merged_from: int = 1
    overlap_unresolved: bool = False


@dataclass
class _Work:
    """Mutable geometry while merging and cutting."""

    poly: Polygon
    baseline: tuple[tuple[float, float], tuple[float, float]]
    score: float
    members: tuple[int, ...]
    role: str = "line"
    merged_from: int = 1
    overlap_unresolved: bool = False
    dropped: bool = False
    suspect: bool = False
    suspect_reason: str = ""
    area: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = self.poly.area


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _median(values: list[float], fallback: float) -> float:
    if not values:
        return fallback
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _work_of_quad(index: int, quad: DetectedQuad, baseline_fraction: float) -> _Work:
    points = [[float(x), float(y)] for x, y in quad.points]
    baseline = synthetic_baseline_points(points, baseline_fraction)
    return _Work(
        poly=Polygon(points),
        baseline=((baseline[0][0], baseline[0][1]), (baseline[1][0], baseline[1][1])),
        score=float(quad.score),
        members=(index,),
    )


def _bridge(prev: _Work, nxt: _Work) -> Polygon | None:
    """Rectangle over the x gap inside the pair's shared y range."""
    _, prev_ymin, prev_xmax, prev_ymax = prev.poly.bounds
    _, nxt_ymin, nxt_xmax, nxt_ymax = nxt.poly.bounds
    low, high = max(prev_ymin, nxt_ymin), min(prev_ymax, nxt_ymax)
    if high <= low:
        return None
    return Polygon([(prev_xmax, low), (nxt_xmax, low), (nxt_xmax, high), (prev_xmax, high)])


def _merged_polygon(parts: list[Polygon]) -> Polygon:
    """Union member polygons and gap bridges into one valid simple polygon."""
    merged = unary_union(parts)
    if isinstance(merged, Polygon) and merged.is_valid and not merged.is_empty:
        return merged
    hull = unary_union(parts).convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        raise ValueError("merge produced no polygon")
    return hull


def _merged_baseline(
    works: list[_Work], baseline_fraction: float, quad_points: list[list[list[float]]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Baseline from the outer ends of the outer members' baselines."""
    left = min(works, key=lambda work: (work.poly.bounds[0], work.poly.bounds[1]))
    right = max(works, key=lambda work: (work.poly.bounds[2], work.poly.bounds[1]))
    left_base = synthetic_baseline_points(quad_points[left.members[0]], baseline_fraction)
    right_base = synthetic_baseline_points(quad_points[right.members[0]], baseline_fraction)
    start = min(left_base, key=lambda point: point[0])
    end = max(right_base, key=lambda point: point[0])
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def merge_row_fragments(
    quads: list[DetectedQuad],
    layout: PageLayout,
    *,
    gap_ratio: float = DEFAULT_MERGE_GAP_RATIO,
    height_ratio: float = DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    baseline_fraction: float = 0.75,
) -> list[_Work]:
    """Merge row mates split into fragments, one visual line per run.

    Inside one column, quads that the row grouping puts in the same row are
    one visual line: runs along x merge when the gap is at most
    ``gap_ratio`` times the page median quad height. A member taller than
    ``height_ratio`` times a run mate never merges: a tall multi-line
    initial stays its own line with ``role`` ``"initial"``. Merging never
    crosses columns; spanning quads pass through untouched.
    """
    heights = [_bbox(quad.points)[3] - _bbox(quad.points)[1] for quad in quads]
    median_height = _median(heights, 1.0) if heights else 1.0
    quad_points = [[[float(x), float(y)] for x, y in quad.points] for quad in quads]
    builder_heights = [
        _bbox(quads[i].points)[3] - _bbox(quads[i].points)[1]
        for column in layout.columns
        for i in column.builders
    ]
    builder_median = _median(builder_heights, median_height)

    in_column = {i for column in layout.columns for i in column.members}
    works: list[_Work] = []
    for column in layout.columns:
        for row in column.rows:
            row_heights = {i: heights[i] for i in row}
            ordered = sorted(
                row,
                key=lambda i: (
                    _bbox(quads[i].points)[0],
                    _bbox(quads[i].points)[1],
                    tuple(round(c, 6) for point in quads[i].points for c in point),
                ),
            )
            run: list[int] = []
            run_max_h = 0.0
            run_min_h = 0.0
            run_xmax = 0.0

            def close_run(run: list[int], row_heights: dict[int, float]) -> None:
                if len(run) > 1:
                    works.append(_merge_run(run, quads, quad_points, baseline_fraction))
                elif run:
                    index = run[0]
                    work = _work_of_quad(index, quads[index], baseline_fraction)
                    mates = [h for j, h in row_heights.items() if j != index]
                    reference = _median(mates, builder_median)
                    if reference > 0 and heights[index] > height_ratio * reference:
                        work.role = "initial"
                    works.append(work)

            for index in ordered:
                xmin, ymin, xmax, ymax = _bbox(quads[index].points)
                height = ymax - ymin
                if not run:
                    run, run_max_h, run_min_h, run_xmax = [index], height, height, xmax
                    continue
                gap = xmin - run_xmax
                new_max = max(run_max_h, height)
                new_min = min(run_min_h, height)
                if gap <= gap_ratio * median_height and new_max <= height_ratio * new_min:
                    run.append(index)
                    run_max_h, run_min_h, run_xmax = new_max, new_min, max(run_xmax, xmax)
                else:
                    close_run(run, row_heights)
                    run, run_max_h, run_min_h, run_xmax = [index], height, height, xmax
            close_run(run, row_heights)
    spanning = sorted(set(range(len(quads))) - in_column)
    for index in spanning:
        works.append(_work_of_quad(index, quads[index], baseline_fraction))
    return works


def _merge_run(
    run: list[int],
    quads: list[DetectedQuad],
    quad_points: list[list[list[float]]],
    baseline_fraction: float,
) -> _Work:
    """Union one x-run of row mates plus gap bridges into a single line."""
    ordered = sorted(run, key=lambda i: _bbox(quads[i].points)[0])
    parts: list[Polygon] = [Polygon(quad_points[i]) for i in ordered]
    members = [_work_of_quad(i, quads[i], baseline_fraction) for i in ordered]
    for prev, nxt in zip(members, members[1:], strict=False):
        bridge = _bridge(prev, nxt)
        if bridge is not None:
            parts.append(bridge)
    merged = _merged_polygon(parts)
    ring = [list(point) for point in merged.exterior.coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    total_area = sum(work.area for work in members)
    score = sum(work.score * work.area for work in members) / total_area if total_area > 0 else 0.0
    baseline = _merged_baseline(members, baseline_fraction, quad_points)
    return _Work(
        poly=merged,
        baseline=baseline,
        score=score,
        members=tuple(ordered),
        merged_from=len(ordered),
    )


def _shared_ratio(left: _Work, right: _Work) -> float:
    smaller = min(left.area, right.area)
    if smaller <= 0:
        return 0.0
    return left.poly.intersection(right.poly).area / smaller


def _baseline_distance(left: _Work, right: _Work) -> float:
    """Perpendicular distance between the two baseline midpoints."""
    first = np.asarray(left.baseline, dtype=float)
    second = np.asarray(right.baseline, dtype=float)
    first_dir = first[1] - first[0]
    second_dir = second[1] - second[0]
    if float(np.dot(first_dir, second_dir)) < 0:
        second_dir = -second_dir
    mean = first_dir + second_dir
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return float(np.linalg.norm((first.mean(axis=0) + second.mean(axis=0)) / 2))
    normal = np.array([-mean[1], mean[0]]) / norm
    return float(abs(np.dot(second.mean(axis=0) - first.mean(axis=0), normal)))


def _median_line_spacing(layout: PageLayout, fallback: float) -> float:
    """Median vertical pitch between consecutive row groups in a column."""
    gaps: list[float] = []
    for column in layout.columns:
        centres = sorted(
            sum(layout.centre_y[i] for i in row) / len(row) for row in column.rows if row
        )
        gaps.extend(b - a for a, b in zip(centres, centres[1:], strict=False) if b > a)
    return _median(gaps, fallback)


def _clip_baseline(
    baseline: tuple[tuple[float, float], tuple[float, float]], kept: Polygon
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a baseline segment to the kept half after a cut."""
    line = LineString([baseline[0], baseline[1]])
    clipped = line.intersection(kept)
    if clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        coords = list(clipped.coords)
        return (tuple(coords[0]), tuple(coords[-1]))
    longest = None
    longest_length = 0.0
    for part in getattr(clipped, "geoms", []):
        if isinstance(part, LineString) and part.length > longest_length:
            longest, longest_length = part, part.length
    if longest is None:
        return None
    coords = list(longest.coords)
    return (tuple(coords[0]), tuple(coords[-1]))


def _cut_pair(left: _Work, right: _Work) -> bool:
    """Clip a stacked pair apart at the mid-baseline line. True when cut."""
    first = np.asarray(left.baseline, dtype=float)
    second = np.asarray(right.baseline, dtype=float)
    first_dir = first[1] - first[0]
    second_dir = second[1] - second[0]
    if float(np.dot(first_dir, second_dir)) < 0:
        second_dir = -second_dir
    mean = first_dir + second_dir
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return False
    direction = mean / norm
    normal = np.array([-direction[1], direction[0]])
    mid = (first.mean(axis=0) + second.mean(axis=0)) / 2
    arm, reach = direction * 1e5, normal * 1e5
    centre = mid.tolist()
    left_sign = float(np.dot(np.asarray(left.poly.centroid.coords[0]) - mid, normal))
    left_sign = 1.0 if left_sign >= 0 else -1.0
    kept: list[Polygon] = []
    for work, sign in ((left, left_sign), (right, -left_sign)):
        corners = [
            (centre - arm + sign * reach).tolist(),
            (centre + arm + sign * reach).tolist(),
            (centre + arm).tolist(),
            (centre - arm).tolist(),
        ]
        half = Polygon(corners)
        piece = work.poly.intersection(half)
        if piece.is_empty or piece.area <= 0.5 * work.area:
            return False
        if not isinstance(piece, Polygon):
            return False
        baseline = _clip_baseline(work.baseline, piece)
        if baseline is None:
            return False
        kept.append((piece, baseline))
    old_areas = (left.area, right.area)
    for work, (piece, baseline) in zip((left, right), kept, strict=True):
        work.poly = piece
        work.baseline = baseline
        work.area = piece.area
    return abs(left.area - old_areas[0]) > 1e-9 or abs(right.area - old_areas[1]) > 1e-9


def resolve_overlaps(
    works: list[_Work],
    layout: PageLayout,
    *,
    cut_threshold: float = DEFAULT_OVERLAP_CUT_THRESHOLD,
    median_height: float = 1.0,
) -> list[_Work]:
    """Cut stacked neighbours apart, drop duplicates, flag the rest.

    Pairs sharing at least ``cut_threshold`` of the smaller polygon: stacked
    pairs (baseline distance at least half the page median line spacing) are
    clipped apart at the mid-baseline line until they share zero area, unless
    the cut would remove more than half of either polygon; coinciding
    baselines with more than half shared are duplicates where only the higher
    score survives; anything else is marked ``overlap_unresolved`` and left.
    """
    spacing = _median_line_spacing(layout, median_height)
    for _ in range(_MAX_OVERLAP_PASSES):
        pairs: list[tuple[float, int, int]] = []
        live = [i for i, work in enumerate(works) if not work.dropped]
        for pos_a, a in enumerate(live):
            for b in live[pos_a + 1 :]:
                ratio = _shared_ratio(works[a], works[b])
                if ratio >= cut_threshold:
                    pairs.append((-ratio, a, b))
        pairs.sort()
        changed = False
        for _, a, b in pairs:
            left, right = works[a], works[b]
            if left.dropped or right.dropped:
                continue
            ratio = _shared_ratio(left, right)
            if ratio < cut_threshold:
                continue
            distance = _baseline_distance(left, right)
            if ratio > DUPLICATE_SHARED_RATIO and distance < spacing / 2:
                if (right.score, right.members) > (left.score, left.members):
                    left.dropped = True
                else:
                    right.dropped = True
                changed = True
            elif distance >= spacing / 2:
                if _cut_pair(left, right):
                    changed = True
                else:
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
            else:
                left.overlap_unresolved = True
                right.overlap_unresolved = True
        if not changed:
            break
    live = [work for work in works if not work.dropped]
    for pos_a, left in enumerate(live):
        for right in live[pos_a + 1 :]:
            if _shared_ratio(left, right) >= cut_threshold:
                left.overlap_unresolved = True
                right.overlap_unresolved = True
    return [work for work in works if not work.dropped]


__all__ = [
    "DEFAULT_MERGE_GAP_RATIO",
    "DEFAULT_MERGE_MAX_HEIGHT_RATIO",
    "DEFAULT_OVERLAP_CUT_THRESHOLD",
    "RefinedLine",
    "merge_row_fragments",
    "resolve_overlaps",
]
