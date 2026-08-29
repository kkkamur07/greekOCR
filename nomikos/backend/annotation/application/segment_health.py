"""Find the three mistakes kraken makes on a page, and build the fix for each.

Everything here is pure geometry: no database, no shapely, no numpy. That is a
deliberate constraint rather than an accident. The platform API ships to Vercel
from the `platform-prod` dependency group, which carries neither shapely nor
numpy, and adding GEOS to a serverless bundle to cut a polygon at a vertical
line would be a poor trade. The two operations actually needed are a half-plane
clip and a join of two pieces of one text line, and both are exact in a hundred
lines of arithmetic.

The other constraint is that every threshold is relative to the page it is
measured on. The same rule in absolute pixels was tried across Syriac, Greek
and two Armenian manuscripts and did not transfer: what counts as a narrow
segment on a 2479px scan is not what counts on a photographed folio. So widths
and heights are compared against that page's own medians, and horizontal
position is compared against column bands derived from that page's own layout.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

Point = list[float]
Polygon = list[Point]
# (min x, max x, min y, max y), the order the reference scripts used.
Box = tuple[float, float, float, float]

# A segment has to be this much wider than the page before it is taken as
# evidence of where a column runs. Narrow things are what we are trying to
# classify, so they cannot also be what defines the layout.
BAND_MIN_WIDTH_FRACTION = 0.15
# How finely the page is sliced when reading columns off the coverage profile.
# 200 slices is about 12px on a 2479px scan, comfortably narrower than a gutter.
COVERAGE_SLICES = 200
# A slice belongs to a column when this many of the page's text-width segments
# cover it, as a fraction of the busiest slice. The reference gutters sit near
# 25% of peak and the columns at 100%, so a floor between the two separates
# them without needing to know how many columns the page has.
COLUMN_COVERAGE_RATIO = 0.35
# Narrow enough to be a candidate for noise. On its own this catches real
# gutter numerals too, which is exactly why it is never the whole rule.
NARROW_WIDTH_FRACTION = 0.06
# A segment with no more corners than this, and squat, is a sliver: a dot shaken
# off an initial, a speck of dirt read as ink.
TINY_MAX_POINTS = 6
TINY_HEIGHT_RATIO = 0.6
# Two baselines belong to one text line if they sit this close, measured
# against the spacing between lines on that page rather than in pixels.
COLLINEAR_SPACING_RATIO = 0.4
# ... and if the horizontal gap between them is smaller than this, measured
# against the height of the line itself.
ADJACENT_GAP_HEIGHT_RATIO = 1.5
# A fragment is the smaller piece by a wide margin. Two halves of similar size
# are more likely two real lines that kraken split correctly.
FRAGMENT_MAX_WIDTH_RATIO = 0.6
# A spanning segment has to reach this far into each band before the band counts
# as covered, so that a polygon merely leaning into its neighbour is left alone.
BAND_COVERAGE_RATIO = 0.2


def bbox(points: Polygon) -> Box:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), max(xs), min(ys), max(ys))


def width_of(points: Polygon) -> float:
    box = bbox(points)
    return box[1] - box[0]


def height_of(points: Polygon) -> float:
    box = bbox(points)
    return box[3] - box[2]


@dataclass(frozen=True)
class Segment:
    """One line as this module needs it, detached from the ORM."""

    id: str
    points: Polygon
    baseline: Polygon
    manual_geometry: bool = False
    # True when something downstream depends on this row surviving: a human
    # typed text against it, or the pairing points at it.
    has_text: bool = False
    is_paired: bool = False


@dataclass(frozen=True)
class PageStats:
    """What a page says about itself, before any segment is judged."""

    width: float
    height: float
    bands: list[tuple[float, float]]
    median_width: float
    median_height: float
    line_spacing: float


@dataclass(frozen=True)
class Suspect:
    line_id: str
    # Plain-language reasons, shown to whoever decides. A flag nobody can check
    # is a flag nobody should act on.
    reasons: list[str]


@dataclass(frozen=True)
class SpanningSplit:
    line_id: str
    cuts: list[float]
    pieces: list[tuple[Polygon, Polygon]] = field(default_factory=list)


@dataclass(frozen=True)
class FragmentMerge:
    primary_id: str
    fragment_id: str
    points: Polygon
    baseline: Polygon


def _median(values: list[float], fallback: float) -> float:
    return statistics.median(values) if values else fallback


def column_bands(segments: list[Segment], page_width: float) -> list[tuple[float, float]]:
    """The x-ranges the page's columns actually occupy.

    Derived from the page rather than configured, because the same document can
    hold one-column and two-column pages and nobody wants to declare which is
    which. Only segments wide enough to be running text vote, so the numerals
    and specks we are about to classify do not get to define the layout that
    classifies them.

    The columns are read off a coverage profile: how many text-width segments
    cover each slice of the page. A gutter is a deep, narrow dip in that
    profile, and on the reference pages it is unmissable, coverage falling from
    24 segments to 6 in a single slice.

    Merging overlapping x-ranges instead would be simpler and wrong. The
    segments that span two columns are precisely the ones this module exists to
    find, and each of them bridges the gutter; a few of those are enough to
    weld the two columns into one band, at which point the spanning segments
    are inside a single column and invisible. Counting coverage rather than
    merging extents keeps a handful of bad segments from outvoting a hundred
    good ones.
    """
    spans = [
        (bbox(segment.points)[0], bbox(segment.points)[1])
        for segment in segments
        if segment.points and width_of(segment.points) >= BAND_MIN_WIDTH_FRACTION * page_width
    ]
    if not spans or page_width <= 0:
        return []
    slice_width = page_width / COVERAGE_SLICES
    coverage = [0] * COVERAGE_SLICES
    for low, high in spans:
        first = max(0, int(low / slice_width))
        last = min(COVERAGE_SLICES - 1, int(high / slice_width))
        for index in range(first, last + 1):
            coverage[index] += 1
    peak = max(coverage)
    if peak == 0:
        return []
    floor = COLUMN_COVERAGE_RATIO * peak
    bands: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, value in enumerate(coverage):
        if value >= floor and run_start is None:
            run_start = index
        elif value < floor and run_start is not None:
            bands.append(_band_extent(spans, run_start, index - 1, slice_width))
            run_start = None
    if run_start is not None:
        bands.append(_band_extent(spans, run_start, COVERAGE_SLICES - 1, slice_width))
    return [band for band in bands if band[1] - band[0] >= BAND_MIN_WIDTH_FRACTION * page_width]


def _band_extent(
    spans: list[tuple[float, float]], first_slice: int, last_slice: int, slice_width: float
) -> tuple[float, float]:
    """Snap a run of covered slices to the segments that actually fill it.

    The slice boundaries are only as fine as the profile, so the band is
    tightened onto the segments lying wholly inside it. Segments that overrun
    the run are ignored here for the same reason they are downweighted above:
    a segment crossing the gutter would drag the band across it.
    """
    low, high = first_slice * slice_width, (last_slice + 1) * slice_width
    inside = [
        span for span in spans if span[0] >= low - slice_width and span[1] <= high + slice_width
    ]
    if not inside:
        return (low, high)
    return (min(span[0] for span in inside), max(span[1] for span in inside))


def _baseline_mean_y(baseline: Polygon) -> float:
    return statistics.mean(point[1] for point in baseline)


def line_spacing(segments: list[Segment], bands: list[tuple[float, float]]) -> float:
    """Median gap between one text line and the next, measured within a column.

    Measured per band, because on a two-column page the baselines of the left
    and right columns interleave in y, and mixing them would report half the
    real spacing and make every threshold built on it too tight.
    """
    gaps: list[float] = []
    for low, high in bands:
        ys = sorted(
            _baseline_mean_y(segment.baseline)
            for segment in segments
            if len(segment.baseline) >= 2 and _band_index(segment, [(low, high)]) == 0
        )
        gaps.extend(
            second - first for first, second in zip(ys, ys[1:], strict=False) if second - first > 0
        )
    if not gaps:
        return 0.0
    return statistics.median(gaps)


def page_stats(segments: list[Segment], page_width: float, page_height: float) -> PageStats:
    usable = [segment for segment in segments if len(segment.points) >= 3]
    bands = column_bands(usable, page_width)
    widths = [width_of(segment.points) for segment in usable]
    heights = [height_of(segment.points) for segment in usable]
    return PageStats(
        width=page_width,
        height=page_height,
        bands=bands,
        median_width=_median(widths, 0.0),
        median_height=_median(heights, 0.0),
        line_spacing=line_spacing(usable, bands),
    )


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _band_index(segment: Segment, bands: list[tuple[float, float]]) -> int | None:
    """Which band this segment sits in, or None when it sits between them."""
    if not segment.points:
        return None
    box = bbox(segment.points)
    span = (box[0], box[1])
    best: int | None = None
    best_overlap = 0.0
    for index, band in enumerate(bands):
        overlap = _overlap(span, band)
        if overlap > best_overlap:
            best, best_overlap = index, overlap
    if best is None:
        return None
    # More than half of the segment has to be inside before it counts as living
    # there; a numeral clipping the edge of a column is still in the gutter.
    return best if best_overlap >= 0.5 * (span[1] - span[0]) else None


def find_suspects(segments: list[Segment], stats: PageStats) -> list[Suspect]:
    """Segments that look like noise, flagged and never deleted.

    The rule is a conjunction on purpose. Width alone was measured on four
    manuscripts and fails on all of them: on one it deletes 46 real gutter
    numerals a page, on another the noise is not narrow at all but squat and
    four-cornered, sitting inside the columns. What survived every page was
    narrow AND (living outside every column OR being barely a polygon).

    Nothing here decides to delete. Some of what this flags is real ink that
    nobody has transcribed yet, and no amount of geometry can tell that from
    a smudge. A human can, in one look, which is what the flag is for.
    """
    found: list[Suspect] = []
    for segment in segments:
        if segment.manual_geometry or len(segment.points) < 3:
            continue
        if segment.has_text or segment.is_paired:
            continue
        width = width_of(segment.points)
        if width >= NARROW_WIDTH_FRACTION * stats.width:
            continue
        reasons = [f"narrower than {NARROW_WIDTH_FRACTION:.0%} of the page"]
        outside = stats.bands and _band_index(segment, stats.bands) is None
        tiny = len(segment.points) <= TINY_MAX_POINTS and (
            stats.median_height > 0
            and height_of(segment.points) < TINY_HEIGHT_RATIO * stats.median_height
        )
        if outside:
            reasons.append("outside every column")
        if tiny:
            reasons.append(
                f"only {len(segment.points)} corners and shorter than "
                f"{TINY_HEIGHT_RATIO:.0%} of a typical line"
            )
        if outside or tiny:
            found.append(Suspect(line_id=segment.id, reasons=reasons))
    return found


def y_at(polyline: Polygon, x: float) -> float:
    """Height of a polyline at an x, clamped to its ends."""
    ordered = sorted(polyline)
    if x <= ordered[0][0]:
        return ordered[0][1]
    if x >= ordered[-1][0]:
        return ordered[-1][1]
    for (ax, ay), (bx, by) in zip(ordered, ordered[1:], strict=False):
        if ax <= x <= bx:
            return ay if bx == ax else ay + (by - ay) * (x - ax) / (bx - ax)
    return ordered[-1][1]


def _intersect_at(first: Point, second: Point, x_cut: float) -> Point:
    (ax, ay), (bx, by) = first, second
    t = (x_cut - ax) / (bx - ax) if bx != ax else 0.0
    return [round(x_cut, 1), round(ay + t * (by - ay), 1)]


def clip_half(polygon: Polygon, x_cut: float, *, keep_left: bool) -> Polygon:
    """Sutherland-Hodgman against the vertical line x = x_cut.

    A half-plane is convex, which is what this algorithm needs of its clip
    region; the polygon being clipped may be as concave as kraken likes.
    """
    inside = (lambda p: p[0] <= x_cut) if keep_left else (lambda p: p[0] >= x_cut)
    out: Polygon = []
    for index in range(len(polygon)):
        current, previous = polygon[index], polygon[index - 1]
        if inside(current):
            if not inside(previous):
                out.append(_intersect_at(previous, current, x_cut))
            out.append([float(current[0]), float(current[1])])
        elif inside(previous):
            out.append(_intersect_at(previous, current, x_cut))
    return out


def clip_polygon(polygon: Polygon, low: float | None, high: float | None) -> Polygon:
    clipped = [[float(p[0]), float(p[1])] for p in polygon]
    if low is not None:
        clipped = clip_half(clipped, low, keep_left=False)
    if high is not None and clipped:
        clipped = clip_half(clipped, high, keep_left=True)
    return clipped


def clip_baseline(baseline: Polygon, low: float | None, high: float | None) -> Polygon:
    if len(baseline) < 2:
        return []
    ordered = sorted([[float(p[0]), float(p[1])] for p in baseline])
    low = ordered[0][0] if low is None else max(low, ordered[0][0])
    high = ordered[-1][0] if high is None else min(high, ordered[-1][0])
    if high - low < 10:
        return []
    inner = [point for point in ordered if low < point[0] < high]
    return (
        [[round(low, 1), round(y_at(ordered, low), 1)]]
        + inner
        + [[round(high, 1), round(y_at(ordered, high), 1)]]
    )


def find_spanning(segments: list[Segment], stats: PageStats) -> list[SpanningSplit]:
    """Segments that swallowed two columns and the gutter between them.

    Kraken emits one block for the whole page, so nothing stops a line box from
    running left column, gutter, right column in a single polygon. Every one of
    these found in the reference data crossed the gutter, which is what makes
    the cut well defined: the midpoint of the gap between the two bands.
    """
    if len(stats.bands) < 2:
        return []
    found: list[SpanningSplit] = []
    for segment in segments:
        if segment.manual_geometry or len(segment.points) < 3:
            continue
        box = bbox(segment.points)
        span = (box[0], box[1])
        covered = [
            index
            for index, band in enumerate(stats.bands)
            if _overlap(span, band)
            >= BAND_COVERAGE_RATIO * min(band[1] - band[0], span[1] - span[0])
        ]
        if len(covered) < 2:
            continue
        cuts = [
            (stats.bands[first][1] + stats.bands[second][0]) / 2
            for first, second in zip(covered, covered[1:], strict=False)
        ]
        bounds: list[float | None] = [None, *cuts, None]
        pieces: list[tuple[Polygon, Polygon]] = []
        degenerate = False
        for index in range(len(covered)):
            piece = clip_polygon(segment.points, bounds[index], bounds[index + 1])
            baseline = clip_baseline(segment.baseline, bounds[index], bounds[index + 1])
            if len(piece) < 3 or len(baseline) < 2:
                # A cut that produces a sliver is a cut in the wrong place.
                # Better to leave the segment whole and say nothing than to
                # offer a fix that loses part of the line.
                degenerate = True
                break
            pieces.append((piece, baseline))
        if degenerate:
            continue
        found.append(SpanningSplit(line_id=segment.id, cuts=cuts, pieces=pieces))
    return found


def baseline_distance(first: Polygon, second: Polygon) -> float:
    """How far apart two baselines run vertically, across the x they share.

    Vertical offset at a common x, not distance between the polylines. The
    difference matters: two halves of one text line are separated mostly
    horizontally, by the gap where the missing text sits, so a straight
    point-to-polyline distance measures that gap and reports two pieces of the
    same line as far apart. What actually distinguishes one line from the next
    is how much higher or lower it runs, which is what this measures.

    Where the two do not overlap in x, each baseline is extended flat from its
    nearest end, so a fragment beyond the end of its line is compared against
    where that line was still going.
    """
    if len(first) < 2 or len(second) < 2:
        return math.inf
    low = min(point[0] for point in first + second)
    high = max(point[0] for point in first + second)
    if high <= low:
        return abs(_baseline_mean_y(first) - _baseline_mean_y(second))
    samples = [low + (high - low) * index / 10 for index in range(11)]
    return statistics.mean(abs(y_at(first, x) - y_at(second, x)) for x in samples)


def _chains(polygon: Polygon) -> tuple[Polygon, Polygon]:
    """The top and bottom edge of a polygon, each left to right.

    Taken as the highest and lowest vertex at every distinct x rather than by
    walking the ring. Walking is the obvious approach and produces a wrong
    outline here: the path along the bottom of a rectangle also carries its two
    end verticals, so joining two such paths re-traverses the inner ends and
    folds a notch back into the gap the merge was supposed to bridge.

    Reading an envelope instead cannot do that. It costs the fine detail of a
    concave top edge, which for the shape of a text line is not detail worth
    keeping.
    """
    by_x: dict[float, tuple[float, float]] = {}
    for x, y in ((float(p[0]), float(p[1])) for p in polygon):
        low, high = by_x.get(x, (y, y))
        by_x[x] = (min(low, y), max(high, y))
    xs = sorted(by_x)
    upper = [[x, by_x[x][0]] for x in xs]
    lower = [[x, by_x[x][1]] for x in xs]
    return upper, lower


def merge_polygons(primary: Polygon, fragment: Polygon) -> Polygon:
    """One outline around two pieces of the same text line.

    Not a general polygon union, and it does not need to be. The two inputs are
    already known to sit on one baseline with a gap between them, and the gap is
    the line's own text, so the merged outline is the upper edge of both pieces
    followed by the lower edge of both, walked back. Joining the chains bridges
    the gap by construction, which is the part that matters: the reference
    implementation had to add an explicit bridging band because a buffer round
    trip on its own silently dropped the fragment.

    Where the two pieces overlap in x they are first cut apart at the middle of
    the overlap, so the chains never double back and the result stays simple.
    """
    left, right = sorted([primary, fragment], key=lambda poly: bbox(poly)[0])
    left_box, right_box = bbox(left), bbox(right)
    if left_box[1] > right_box[0]:
        middle = (left_box[1] + right_box[0]) / 2
        clipped_left = clip_half(left, middle, keep_left=True)
        clipped_right = clip_half(right, middle, keep_left=False)
        if len(clipped_left) >= 3 and len(clipped_right) >= 3:
            left, right = clipped_left, clipped_right
    left_upper, left_lower = _chains(left)
    right_upper, right_lower = _chains(right)
    merged = left_upper + right_upper + list(reversed(right_lower)) + list(reversed(left_lower))
    return _dedupe([[round(p[0], 1), round(p[1], 1)] for p in merged])


def _dedupe(points: Polygon) -> Polygon:
    out: Polygon = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def merge_baselines(primary: Polygon, fragment: Polygon) -> Polygon:
    """Both baselines as one, left to right, without near-duplicate points."""
    points = sorted(
        [[float(p[0]), float(p[1])] for p in primary]
        + [[float(p[0]), float(p[1])] for p in fragment]
    )
    if not points:
        return []
    out = [points[0]]
    for point in points[1:]:
        if abs(point[0] - out[-1][0]) >= 4:
            out.append(point)
    if len(out) < 2:
        out = [points[0], points[-1]]
    return [[round(p[0], 1), round(p[1], 1)] for p in out]


def find_fragments(segments: list[Segment], stats: PageStats) -> list[FragmentMerge]:
    """Pairs where kraken cut one text line into two.

    A line end, a piece shaken off a decorated initial, a stray dot. The pair
    has to share a baseline, sit side by side with a small gap, and be lopsided:
    two halves of similar width are more likely two real lines that were split
    correctly than one line that was split wrongly.

    The larger piece is the primary and keeps its id, so whatever is already
    attached to it, a transcription or a pairing, survives the merge. A
    fragment that carries text or a pairing of its own is never offered for
    merging, because merging it would destroy that.
    """
    if stats.line_spacing <= 0:
        return []
    usable = [
        segment
        for segment in segments
        if not segment.manual_geometry and len(segment.points) >= 3 and len(segment.baseline) >= 2
    ]
    collinear_limit = COLLINEAR_SPACING_RATIO * stats.line_spacing
    found: list[FragmentMerge] = []
    claimed: set[str] = set()
    for index, first in enumerate(usable):
        for second in usable[index + 1 :]:
            if first.id in claimed or second.id in claimed:
                continue
            if _band_index(first, stats.bands) != _band_index(second, stats.bands):
                continue
            if baseline_distance(first.baseline, second.baseline) >= collinear_limit:
                continue
            first_box, second_box = bbox(first.points), bbox(second.points)
            gap = max(first_box[0], second_box[0]) - min(first_box[1], second_box[1])
            line_height = max(height_of(first.points), height_of(second.points))
            if gap > ADJACENT_GAP_HEIGHT_RATIO * line_height:
                continue
            wide, narrow = sorted(
                (first, second), key=lambda seg: width_of(seg.points), reverse=True
            )
            if width_of(narrow.points) > FRAGMENT_MAX_WIDTH_RATIO * width_of(wide.points):
                continue
            if narrow.has_text or narrow.is_paired:
                continue
            found.append(
                FragmentMerge(
                    primary_id=wide.id,
                    fragment_id=narrow.id,
                    points=merge_polygons(wide.points, narrow.points),
                    baseline=merge_baselines(wide.baseline, narrow.baseline),
                )
            )
            claimed.add(narrow.id)
    return found
