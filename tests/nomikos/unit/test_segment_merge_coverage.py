"""When a freshly segmented line is the same ink as a line the merge kept."""

from __future__ import annotations

from shapely.geometry import Polygon

from backend.document.application.segment_merge_service import _mostly_covered


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _kept(*boxes: list[list[float]]) -> list[Polygon]:
    return [Polygon(box) for box in boxes]


def test_a_near_identical_redraw_is_covered():
    kept = _kept(_box(100, 100, 900, 160))
    assert _mostly_covered(_box(96, 104, 905, 158), kept)


def test_a_kept_numeral_inside_a_fresh_text_line_does_not_cover_it():
    # The gutter numeral someone boxed by hand sits inside the sweep of a fresh kraken
    # text line. The numeral claims a sliver of that line's ink, not the line.
    kept = _kept(_box(880, 110, 940, 150))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_neighbours_touching_at_ascenders_do_not_add_up_to_coverage():
    # The line above and the line below each dip a few pixels into this one. Polygons,
    # not bounding boxes, so two shallow overlaps stay two shallow overlaps.
    kept = _kept(_box(100, 40, 900, 106), _box(100, 154, 900, 220))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_a_fresh_line_spanning_two_kept_lines_is_already_claimed():
    # Kraken draws one box across the gutter over two approved lines, or one line over
    # two approved fragments. No single kept line matches it, and it is still not new
    # ink: what it covers is claimed twice over, so it is skipped.
    kept = _kept(_box(100, 100, 480, 160), _box(520, 100, 900, 160))
    assert _mostly_covered(_box(100, 100, 900, 160), kept)


def test_half_inside_is_the_same_ink():
    kept = _kept(_box(100, 100, 500, 160))
    assert _mostly_covered(_box(100, 100, 900, 160), kept)
    assert not _mostly_covered(_box(100, 100, 901, 160), kept)


def test_nothing_kept_or_degenerate_geometry_covers_nothing():
    assert not _mostly_covered(_box(0, 0, 10, 10), [])
    assert not _mostly_covered([[0, 0], [10, 0]], _kept(_box(0, 0, 10, 10)))
    assert not _mostly_covered([], _kept(_box(0, 0, 10, 10)))
