"""Geometry behind the segment-health suggestions.

Several of these guard a mistake that was actually made while writing the
module and caught by measuring against two real manuscripts, rather than a
hazard imagined afterwards. Those are called out where they sit.
"""

from __future__ import annotations

from backend.annotation.application.segment_health import (
    Segment,
    baseline_distance,
    bbox,
    clip_baseline,
    clip_polygon,
    column_bands,
    find_fragments,
    find_spanning,
    find_suspects,
    merge_polygons,
    page_stats,
)

PAGE_WIDTH = 2400.0
PAGE_HEIGHT = 3400.0
LEFT_COLUMN = (300.0, 1100.0)
RIGHT_COLUMN = (1300.0, 2100.0)


def line(
    identifier: str,
    x0: float,
    x1: float,
    y: float,
    *,
    height: float = 100.0,
    corners: int = 4,
    manual: bool = False,
    has_text: bool = False,
    is_paired: bool = False,
) -> Segment:
    """A rectangular text line, optionally with extra corners along its top."""
    top, bottom = y - height / 2, y + height / 2
    points = [[x0, top]]
    for index in range(1, max(1, corners - 3)):
        points.append([x0 + (x1 - x0) * index / max(1, corners - 3), top])
    points += [[x1, top], [x1, bottom], [x0, bottom]]
    return Segment(
        id=identifier,
        points=points,
        baseline=[[x0, bottom], [x1, bottom]],
        manual_geometry=manual,
        has_text=has_text,
        is_paired=is_paired,
    )


def tapered(x0: float, x1: float, y: float, *, height: float = 100.0) -> list[list[float]]:
    """A polygon shaped the way kraken draws one, rather than a rectangle.

    One vertex at each end, so the outline is a single point tall there, and the
    top and bottom vertices at different x in between.
    """
    top, bottom = y - height / 2, y + height / 2
    span = x1 - x0
    return (
        [[x0, y - height / 8]]
        + [[x0 + span * f, top + height / 12 * i] for i, f in enumerate((0.2, 0.5, 0.8))]
        + [[x1, y + height / 8]]
        + [[x0 + span * f, bottom - height / 12 * i] for i, f in enumerate((0.75, 0.45, 0.15))]
    )


def two_column_page(spanning: list[Segment] | None = None) -> list[Segment]:
    """Twenty lines a side, at a 120px pitch, with an empty gutter between."""
    segments: list[Segment] = []
    for row in range(20):
        y = 400.0 + row * 120.0
        segments.append(line(f"L{row}", *LEFT_COLUMN, y))
        segments.append(line(f"R{row}", *RIGHT_COLUMN, y))
    return segments + list(spanning or [])


class TestColumnBands:
    def test_finds_both_columns_and_the_gutter_between_them(self) -> None:
        bands = column_bands(two_column_page(), PAGE_WIDTH)
        assert len(bands) == 2
        assert bands[0][1] < bands[1][0]

    def test_a_few_segments_crossing_the_gutter_do_not_weld_the_columns(self) -> None:
        # The failure this catches, and it is the one the first implementation
        # shipped with. Merging overlapping x-ranges makes the segments that
        # span the gutter define a single band covering the whole text area.
        # The spanning segments are then inside one column, so nothing detects
        # them, and line spacing is measured across two interleaved columns and
        # comes out at half its real value, which silently disables the
        # fragment detector too. One bad segment must not outvote forty good
        # ones.
        crossers = [
            line(f"X{index}", LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0 + index * 120.0)
            for index in range(3)
        ]
        bands = column_bands(two_column_page(crossers), PAGE_WIDTH)
        assert len(bands) == 2, f"gutter lost to {len(crossers)} spanning segments: {bands}"

    def test_a_single_column_page_yields_one_band(self) -> None:
        segments = [line(f"L{row}", 300.0, 2100.0, 400.0 + row * 120.0) for row in range(20)]
        assert len(column_bands(segments, PAGE_WIDTH)) == 1

    def test_a_page_of_nothing_but_specks_has_no_bands(self) -> None:
        specks = [
            line(f"S{index}", 100.0 + index * 30, 120.0 + index * 30, 200.0) for index in range(10)
        ]
        assert column_bands(specks, PAGE_WIDTH) == []


class TestBaselineDistance:
    def test_two_pieces_of_one_line_read_as_close_despite_the_gap(self) -> None:
        # The failure this catches. Measuring point-to-polyline distance makes
        # the horizontal gap between two fragments dominate the answer, so two
        # halves of one text line report as further apart than two different
        # lines stacked on top of each other, and no fragment is ever found.
        left = [[300.0, 500.0], [800.0, 500.0]]
        right = [[1000.0, 502.0], [1100.0, 502.0]]
        stacked = [[300.0, 620.0], [1100.0, 620.0]]
        assert baseline_distance(left, right) < baseline_distance(left, stacked)
        assert baseline_distance(left, right) < 10

    def test_a_line_and_its_neighbour_read_as_a_line_apart(self) -> None:
        first = [[300.0, 500.0], [1100.0, 500.0]]
        second = [[300.0, 620.0], [1100.0, 620.0]]
        assert baseline_distance(first, second) == 120.0


class TestClipping:
    def test_cutting_a_rectangle_keeps_each_side_whole(self) -> None:
        rectangle = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
        left = clip_polygon(rectangle, None, 60.0)
        right = clip_polygon(rectangle, 60.0, None)
        assert bbox(left) == (0.0, 60.0, 0.0, 50.0)
        assert bbox(right) == (60.0, 100.0, 0.0, 50.0)

    def test_a_baseline_cut_too_short_is_refused_rather_than_returned_tiny(self) -> None:
        assert clip_baseline([[0.0, 10.0], [100.0, 10.0]], 0.0, 5.0) == []


class TestSpanning:
    def test_a_segment_across_the_gutter_is_cut_between_the_columns(self) -> None:
        crosser = line("X", LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0)
        segments = two_column_page([crosser])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        found = [split for split in find_spanning(segments, stats) if split.line_id == "X"]
        assert len(found) == 1
        (cut,) = found[0].cuts
        assert LEFT_COLUMN[1] < cut < RIGHT_COLUMN[0]
        assert len(found[0].pieces) == 2

    def test_an_ordinary_line_inside_one_column_is_left_alone(self) -> None:
        segments = two_column_page()
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_spanning(segments, stats) == []


class TestFragments:
    def test_a_line_end_is_offered_for_merging_into_its_line(self) -> None:
        segments = two_column_page()
        primary = line("P", 300.0, 900.0, 2900.0)
        fragment = line("F", 960.0, 1060.0, 2900.0)
        segments += [primary, fragment]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        merges = [merge for merge in find_fragments(segments, stats) if merge.fragment_id == "F"]
        assert len(merges) == 1
        assert merges[0].primary_id == "P"

    def test_a_fragment_carrying_transcribed_text_is_never_offered(self) -> None:
        segments = two_column_page()
        segments += [
            line("P", 300.0, 900.0, 2900.0),
            line("F", 960.0, 1060.0, 2900.0, has_text=True),
        ]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [m for m in find_fragments(segments, stats) if m.fragment_id == "F"] == []

    def test_two_halves_of_similar_width_are_taken_as_two_real_lines(self) -> None:
        segments = two_column_page()
        segments += [line("A", 300.0, 700.0, 2900.0), line("B", 740.0, 1100.0, 2900.0)]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [m for m in find_fragments(segments, stats) if m.fragment_id in {"A", "B"}] == []


class TestMergePolygons:
    def test_the_merged_outline_spans_the_gap_instead_of_dropping_the_fragment(self) -> None:
        # The failure this catches is the one the reference implementation
        # documents: bridging by a buffer round trip alone silently drops the
        # fragment, leaving the primary unchanged and the merge a no-op that
        # still reports success.
        primary = [[300.0, 450.0], [900.0, 450.0], [900.0, 550.0], [300.0, 550.0]]
        fragment = [[960.0, 450.0], [1060.0, 450.0], [1060.0, 550.0], [960.0, 550.0]]
        merged = merge_polygons(primary, fragment)
        box = bbox(merged)
        assert box[0] == 300.0
        assert box[1] == 1060.0
        # A point in the gap has to be inside the outline, which is what makes
        # this one line rather than two pieces sharing a bounding box.
        assert _contains(merged, (930.0, 500.0))

    def test_the_outline_of_two_kraken_shaped_pieces_does_not_cross_itself(self) -> None:
        # The failure this catches, and it is the one that shipped twice,
        # because every fixture above is a rectangle. A rectangle puts its top
        # and bottom vertices at the same x and is a full line height tall at
        # both ends. A kraken mask does neither: it tapers to one vertex at each
        # end, and its top and bottom vertices sit at different x. The end
        # vertex then belongs to the top edge and the bottom edge at once, so
        # the outline arrived at it twice and closed against itself, and the
        # attempt at walking around it instead sent the return path back across
        # the outward one. Both are outlines no consumer of geometry accepts,
        # and both scored well on bounding box and point count, which is all the
        # rectangles could ever have checked.
        merged = merge_polygons(tapered(300.0, 900.0, 500.0), tapered(960.0, 1160.0, 504.0))
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        assert _contains(merged, (930.0, 500.0)), "the gap between the pieces is not enclosed"

    def test_a_fragment_sitting_inside_the_primary_still_gives_a_simple_outline(self) -> None:
        # Not every pair offered is side by side. A blot under a line overlaps
        # it in x completely, and cutting the two apart at the middle of that
        # overlap leaves nothing of one of them, which used to leave the two
        # edges interleaved and crossing.
        primary = tapered(1381.0, 1446.0, 2840.0, height=120.0)
        fragment = tapered(1399.0, 1414.0, 2985.0, height=280.0)
        merged = merge_polygons(primary, fragment)
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        assert bbox(merged)[1] >= 1446.0, "the wider piece was clipped away"

    def test_pieces_overlapping_in_x_still_produce_a_simple_outline(self) -> None:
        primary = [[300.0, 450.0], [900.0, 450.0], [900.0, 550.0], [300.0, 550.0]]
        fragment = [[850.0, 450.0], [1000.0, 450.0], [1000.0, 550.0], [850.0, 550.0]]
        merged = merge_polygons(primary, fragment)
        assert bbox(merged) == (300.0, 1000.0, 450.0, 550.0)
        assert len(merged) >= 4


class TestSuspects:
    def test_running_text_is_never_suspect(self) -> None:
        segments = two_column_page()
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []

    def test_a_speck_in_the_outer_margin_is_flagged(self) -> None:
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [s.line_id for s in find_suspects(segments, stats)] == ["N"]

    def test_a_hand_drawn_segment_is_never_suspect(self) -> None:
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0, manual=True)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []

    def test_a_narrow_segment_someone_has_transcribed_is_never_suspect(self) -> None:
        # Untranscribed marginalia is exactly what this whole feature must not
        # delete on its own judgement; once it carries text, it is not even a
        # candidate.
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0, has_text=True)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []


def _contains(polygon: list[list[float]], point: tuple[float, float]) -> bool:
    """Ray casting, so the merge test asserts containment rather than extent."""
    x, y = point
    inside = False
    for index in range(len(polygon)):
        ax, ay = polygon[index]
        bx, by = polygon[index - 1]
        if (ay > y) != (by > y) and x < ax + (bx - ax) * (y - ay) / (by - ay):
            inside = not inside
    return inside


def _is_simple(polygon: list[list[float]]) -> bool:
    """No vertex used twice, and no two edges that are not neighbours crossing."""
    if len({(p[0], p[1]) for p in polygon}) != len(polygon):
        return False
    count = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % count]) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _crosses(*edges[i], *edges[j]):
                return False
    return True


def _crosses(a, b, c, d) -> bool:
    def side(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(value) < 1e-9 else (1 if value > 0 else -1)

    def on(p, q, r):
        return (
            side(p, q, r) == 0
            and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and (min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
        )

    d1, d2, d3, d4 = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    if d1 != d2 and d3 != d4:
        return True
    return any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))
