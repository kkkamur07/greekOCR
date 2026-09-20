"""Column-aware reading order for PP-OCRv6 detection quads.

Manuscript pages are often spreads: two text columns side by side that must
read left page first, then right page, each top to bottom. Sorting by y alone
interleaves the columns, so quads are first grouped into columns by
horizontal overlap, columns are ordered across the page, and inside a column
quads are grouped into rows by vertical overlap and read row by row. A quad
that spans two or more columns (a running head or caption) reads as its own
band between the rows above and below it.
"""

from __future__ import annotations

from dataclasses import dataclass

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad

# A quad counts as narrow (a marginal note, page number or initial) when its
# width is below half the page's median quad width. Such quads must not split
# or reorder a column: they attach to the nearest column instead.
NARROW_WIDTH_FRACTION = 0.5
# Two quads share a column when their x ranges overlap by at least half of
# the narrower quad's width, merged transitively.
COLUMN_OVERLAP_FRACTION = 0.5
# A non-narrow quad wider than this multiple of the median width never founds
# a column: it joins the one column it overlaps, or reads as its own band
# when it spans two or more columns.
WIDE_WIDTH_FRACTION = 1.5
# A wide quad overlaps a column when their x ranges share more than this
# fraction of the column's width.
SPANNING_OVERLAP_FRACTION = 0.10
# Two quads in a column share a row when their y ranges overlap by at least
# half of the shorter height and they sit side by side (x overlap below half
# of the narrower width). Stacked quads in the same x range are different
# lines however much they overlap vertically.
ROW_OVERLAP_FRACTION = 0.5
# A quad taller than this multiple of its column's median line height (an
# enlarged initial or figure) joins the first row it overlaps without
# extending that row, so it cannot chain stacked lines into one row.
TALL_HEIGHT_FRACTION = 1.8


def _extents(quads: list[DetectedQuad]) -> list[tuple[float, ...]]:
    """Per quad (xmin, xmax, cx, cy, width, ymin, ymax, height)."""
    extents: list[tuple[float, ...]] = []
    for quad in quads:
        xs = [point[0] for point in quad.points]
        ys = [point[1] for point in quad.points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        extents.append(
            (
                xmin,
                xmax,
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                xmax - xmin,
                ymin,
                ymax,
                ymax - ymin,
            )
        )
    return extents


def _find(roots: list[int], index: int) -> int:
    while roots[index] != index:
        roots[index] = roots[roots[index]]
        index = roots[index]
    return index


def _overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return min(max_a, max_b) - max(min_a, min_b)


def _nearest_column(ranges: list[tuple[float, float]], xmin: float, xmax: float) -> int:
    best, best_gap = 0, None
    for position, (column_xmin, column_xmax) in enumerate(ranges):
        gap = max(0.0, column_xmin - xmax, xmin - column_xmax)
        if best_gap is None or gap < best_gap:
            best, best_gap = position, gap
    return best


def _shares_row(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    height: float,
    row_xmin: float,
    row_xmax: float,
    row_ymin: float,
    row_ymax: float,
) -> bool:
    """Whether a quad belongs to a row with the given x and y ranges."""
    shorter_h = min(height, row_ymax - row_ymin)
    if _overlap(ymin, ymax, row_ymin, row_ymax) < ROW_OVERLAP_FRACTION * shorter_h:
        return False
    # The side-by-side rule reuses the half-width threshold of the column
    # rule: stacked quads sharing an x range are separate lines.
    narrower_w = min(xmax - xmin, row_xmax - row_xmin)
    return _overlap(xmin, xmax, row_xmin, row_xmax) < COLUMN_OVERLAP_FRACTION * narrower_w


def _row_groups(
    members: list[int],
    extents: list[tuple[float, ...]],
    geoms: list[tuple[float, ...]],
    column_builders: set[int],
    direction: str,
) -> list[list[int]]:
    """Group one column's members into rows, top to bottom.

    Rows group transitively along the centre-y sequence: each quad is
    compared with the current row's ranges, so a tall initial met midway
    cannot chain the lines above and below it into one row. Tall quads join
    the first row they overlap and never extend it. Every tie-break is
    geometric (centre, then coordinates), never the input index, so the
    result does not depend on input order.
    """
    by_y = sorted(members, key=lambda i: (extents[i][3], extents[i][2], geoms[i]))
    heights = sorted(extents[i][7] for i in members if i in column_builders)
    median_height = heights[len(heights) // 2]
    rows: list[list[int]] = []
    row_ranges: list[list[float]] = []
    for index in by_y:
        xmin, xmax = extents[index][0], extents[index][1]
        ymin, ymax, height = extents[index][5], extents[index][6], extents[index][7]
        if median_height > 0 and height > TALL_HEIGHT_FRACTION * median_height:
            target = None
            for position, row_range in enumerate(row_ranges):
                if _shares_row(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    height,
                    row_range[2],
                    row_range[3],
                    row_range[0],
                    row_range[1],
                ):
                    target = position
                    break
            if target is None:
                rows.append([index])
                row_ranges.append([ymin, ymax, xmin, xmax])
            else:
                rows[target].append(index)
            continue
        if row_ranges:
            row_ymin, row_ymax, row_xmin, row_xmax = row_ranges[-1]
            if _shares_row(xmin, xmax, ymin, ymax, height, row_xmin, row_xmax, row_ymin, row_ymax):
                rows[-1].append(index)
                row_ranges[-1][0] = min(row_ymin, ymin)
                row_ranges[-1][1] = max(row_ymax, ymax)
                row_ranges[-1][2] = min(row_xmin, xmin)
                row_ranges[-1][3] = max(row_xmax, xmax)
                continue
        rows.append([index])
        row_ranges.append([ymin, ymax, xmin, xmax])
    # Rows read by their top-most centre y, ties broken geometrically.
    ranked = sorted(
        range(len(rows)),
        key=lambda position: (
            min(extents[i][3] for i in rows[position]),
            row_ranges[position][0],
            row_ranges[position][1],
            tuple(sorted(geoms[i] for i in rows[position])),
        ),
    )
    grouped = []
    for position in ranked:
        if direction == "ltr":
            grouped.append(sorted(rows[position], key=lambda i: (extents[i][2], geoms[i])))
        else:
            grouped.append(sorted(rows[position], key=lambda i: (-extents[i][2], geoms[i])))
    return grouped


def _row_order(
    members: list[int],
    extents: list[tuple[float, ...]],
    geoms: list[tuple[float, ...]],
    column_builders: set[int],
    direction: str,
) -> list[int]:
    """Order one column's members top to bottom, row by row."""
    return [
        index
        for row in _row_groups(members, extents, geoms, column_builders, direction)
        for index in row
    ]


@dataclass(frozen=True)
class ColumnLayout:
    """One column: its members, row groups and reading sequence.

    ``builders`` are the full-width members that found the column; the
    text band is their x range, while attached narrow or wide members may
    stick out past it.
    """

    members: tuple[int, ...]
    builders: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    sequence: tuple[int, ...]
    xmin: float
    xmax: float


@dataclass(frozen=True)
class PageLayout:
    """Column and row structure refinement shares with ordering."""

    columns: tuple[ColumnLayout, ...]
    column_order: tuple[int, ...]
    spanning: tuple[int, ...]
    direction: str
    centre_y: tuple[float, ...]


def layout_lines(quads: list[DetectedQuad], *, direction: str = "ltr") -> PageLayout:
    """Group quads into columns and rows, shared by ordering and refinement.

    Columns are in canonical left to right order with ``column_order`` giving
    the reading sequence; each column carries its row groups top to bottom
    and its flat reading ``sequence``. Spanning quads are sorted top to
    bottom. Every tie-break is geometric, so equal inputs give equal outputs
    whatever order they arrive in.
    """

    if direction not in ("ltr", "rtl"):
        raise ValueError('reading direction must be "ltr" or "rtl"')
    count = len(quads)
    if count == 0:
        return PageLayout(
            columns=(), column_order=(), spanning=(), direction=direction, centre_y=()
        )
    extents = _extents(quads)
    geoms = [tuple(float(c) for point in quad.points for c in point) for quad in quads]
    widths = sorted(extent[4] for extent in extents)
    median_width = widths[count // 2]
    # The median quad itself can never be narrow or wide, so at least one
    # column builder always exists for the rest to attach to.
    narrow = {
        index
        for index, extent in enumerate(extents)
        if median_width > 0 and extent[4] < NARROW_WIDTH_FRACTION * median_width
    }
    wide = {
        index
        for index, extent in enumerate(extents)
        if median_width > 0 and extent[4] > WIDE_WIDTH_FRACTION * median_width
    }
    builders = [index for index in range(count) if index not in narrow and index not in wide]

    roots = list(range(count))
    for position, left in enumerate(builders):
        for right in builders[position + 1 :]:
            overlap = _overlap(
                extents[left][0], extents[left][1], extents[right][0], extents[right][1]
            )
            narrower = min(extents[left][4], extents[right][4])
            if overlap >= COLUMN_OVERLAP_FRACTION * narrower:
                left_root, right_root = _find(roots, left), _find(roots, right)
                if left_root != right_root:
                    roots[max(left_root, right_root)] = min(left_root, right_root)

    columns: dict[int, list[int]] = {}
    for index in builders:
        columns.setdefault(_find(roots, index), []).append(index)

    def _column_key(
        members: list[int], xmin: float, xmax: float
    ) -> tuple[float, float, tuple[tuple[float, float], ...]]:
        # Member centres break range ties without depending on input order.
        return (
            xmin,
            xmax,
            tuple(sorted((extents[i][2], extents[i][3]) for i in members)),
        )

    keyed = sorted(
        columns.items(),
        key=lambda item: _column_key(
            item[1],
            min(extents[i][0] for i in item[1]),
            max(extents[i][1] for i in item[1]),
        ),
    )
    ordered_columns = [members for _, members in keyed]
    builder_sets = [set(members) for members in ordered_columns]

    # Wide quads never merge columns: one overlapped column means membership,
    # two or more means a band of its own.
    spanning: list[int] = []
    for index in sorted(wide):
        xmin, xmax = extents[index][0], extents[index][1]
        qualifying = [
            position
            for position, members in enumerate(ordered_columns)
            if _overlap(
                xmin,
                xmax,
                min(extents[i][0] for i in members),
                max(extents[i][1] for i in members),
            )
            > SPANNING_OVERLAP_FRACTION
            * (max(extents[i][1] for i in members) - min(extents[i][0] for i in members))
        ]
        if len(qualifying) >= 2:
            spanning.append(index)
        elif len(qualifying) == 1:
            ordered_columns[qualifying[0]].append(index)
        else:
            ranges = [
                (
                    min(extents[i][0] for i in members),
                    max(extents[i][1] for i in members),
                )
                for members in ordered_columns
            ]
            ordered_columns[_nearest_column(ranges, xmin, xmax)].append(index)

    ranges = [
        (
            min(extents[i][0] for i in members),
            max(extents[i][1] for i in members),
        )
        for members in ordered_columns
    ]
    for index in sorted(narrow):
        ordered_columns[_nearest_column(ranges, extents[index][0], extents[index][1])].append(index)

    column_order = sorted(
        range(len(ordered_columns)),
        key=lambda position: _column_key(
            ordered_columns[position], ranges[position][0], ranges[position][1]
        ),
        reverse=(direction == "rtl"),
    )
    sequenced = [
        _row_order(members, extents, geoms, builder_sets[position], direction)
        for position, members in enumerate(ordered_columns)
    ]
    spanning_sorted = sorted(spanning, key=lambda i: (extents[i][3], extents[i][2], geoms[i]))
    built_columns = tuple(
        ColumnLayout(
            members=tuple(members),
            builders=tuple(sorted(builder_sets[position])),
            rows=tuple(
                tuple(row)
                for row in _row_groups(members, extents, geoms, builder_sets[position], direction)
            ),
            sequence=tuple(sequenced[position]),
            xmin=ranges[position][0],
            xmax=ranges[position][1],
        )
        for position, members in enumerate(ordered_columns)
    )
    return PageLayout(
        columns=built_columns,
        column_order=tuple(column_order),
        spanning=tuple(spanning_sorted),
        direction=direction,
        centre_y=tuple(extent[3] for extent in extents),
    )


def order_lines(quads: list[DetectedQuad], *, direction: str = "ltr") -> list[int]:
    """Return quad indices in reading order.

    Columns group transitively by x overlap; narrow quads attach to the
    nearest column without splitting it, and wide quads either join their
    one overlapped column or, when spanning two or more, read as bands of
    their own. Columns run left to right for ``"ltr"`` and right to left for
    ``"rtl"``; inside a column quads run row by row, each row across the
    page. Every tie-break is geometric, so equal inputs give equal outputs
    whatever order they arrive in.
    """

    layout = layout_lines(quads, direction=direction)
    sequenced = [list(column.sequence) for column in layout.columns]
    if not layout.spanning:
        return [index for position in layout.column_order for index in sequenced[position]]

    # Spanning quads split the page into horizontal bands: everything above a
    # band's centre reads first, then the band, then everything below.
    span_cys = [layout.centre_y[i] for i in layout.spanning]

    def _segment(index: int) -> int:
        band = 0
        for centre_y in span_cys:
            if centre_y < layout.centre_y[index]:
                band += 1
            else:
                break
        return band

    reading: list[int] = []
    for band, span in enumerate(layout.spanning):
        for position in layout.column_order:
            reading.extend(index for index in sequenced[position] if _segment(index) == band)
        reading.append(span)
    for position in layout.column_order:
        reading.extend(
            index for index in sequenced[position] if _segment(index) == len(layout.spanning)
        )
    return reading


__all__ = ["ColumnLayout", "PageLayout", "layout_lines", "order_lines"]
