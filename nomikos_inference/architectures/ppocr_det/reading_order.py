"""Column-aware reading order for PP-OCRv6 detection quads.

Manuscript pages are often spreads: two text columns side by side that must
read left page first, then right page, each top to bottom. Sorting by y alone
interleaves the columns, so quads are first grouped into columns by
horizontal overlap, columns are ordered across the page, and quads are
ordered down each column.
"""

from __future__ import annotations

from .postprocessing import DetectedQuad

# A quad counts as narrow (a marginal note, page number or initial) when its
# width is below half the page's median quad width. Such quads must not split
# or reorder a column: they attach to the nearest column instead.
NARROW_WIDTH_FRACTION = 0.5
# Two quads share a column when their x ranges overlap by at least half of
# the narrower quad's width, merged transitively.
COLUMN_OVERLAP_FRACTION = 0.5


def _extents(quads: list[DetectedQuad]) -> list[tuple[float, float, float, float, float]]:
    extents = []
    for quad in quads:
        xs = [point[0] for point in quad.points]
        ys = [point[1] for point in quad.points]
        xmin, xmax = min(xs), max(xs)
        extents.append((xmin, xmax, (xmin + xmax) / 2.0, (min(ys) + max(ys)) / 2.0, xmax - xmin))
    return extents


def _find(roots: list[int], index: int) -> int:
    while roots[index] != index:
        roots[index] = roots[roots[index]]
        index = roots[index]
    return index


def order_lines(quads: list[DetectedQuad], *, direction: str = "ltr") -> list[int]:
    """Return quad indices in reading order.

    Columns group transitively by x overlap; narrow quads attach to the
    nearest column without splitting it. Columns run left to right for
    ``"ltr"`` and right to left for ``"rtl"``; inside a column quads run by
    centre y, ties by centre x. Equal inputs give equal outputs.
    """

    if direction not in ("ltr", "rtl"):
        raise ValueError('reading direction must be "ltr" or "rtl"')
    count = len(quads)
    if count == 0:
        return []
    extents = _extents(quads)
    widths = sorted(extent[4] for extent in extents)
    median_width = widths[count // 2]
    # The median quad itself can never be narrow, so at least one column
    # always exists for the narrow quads to attach to.
    narrow = {
        index
        for index, extent in enumerate(extents)
        if median_width > 0 and extent[4] < NARROW_WIDTH_FRACTION * median_width
    }

    roots = list(range(count))
    full = [index for index in range(count) if index not in narrow]
    for position, left in enumerate(full):
        for right in full[position + 1 :]:
            overlap = min(extents[left][1], extents[right][1]) - max(
                extents[left][0], extents[right][0]
            )
            narrower = min(extents[left][4], extents[right][4])
            if overlap >= COLUMN_OVERLAP_FRACTION * narrower:
                left_root, right_root = _find(roots, left), _find(roots, right)
                if left_root != right_root:
                    roots[max(left_root, right_root)] = min(left_root, right_root)

    columns: dict[int, list[int]] = {}
    for index in full:
        columns.setdefault(_find(roots, index), []).append(index)
    column_ranges = {
        root: (
            min(extents[index][0] for index in members),
            max(extents[index][1] for index in members),
        )
        for root, members in columns.items()
    }
    # Column identity is the insertion order of first membership, which
    # depends on input order; re-key by extent so the result is deterministic.
    keyed = sorted(
        columns.items(),
        key=lambda item: (
            column_ranges[item[0]][0],
            column_ranges[item[0]][1],
            item[0],
        ),
    )
    ordered_columns = [members for _, members in keyed]
    keyed_ranges = [column_ranges[root] for root, _ in keyed]

    for index in sorted(narrow):
        xmin, xmax = extents[index][0], extents[index][1]
        best, best_gap = 0, None
        for position, (column_xmin, column_xmax) in enumerate(keyed_ranges):
            gap = max(0.0, column_xmin - xmax, xmin - column_xmax)
            if best_gap is None or gap < best_gap:
                best, best_gap = position, gap
        ordered_columns[best].append(index)

    column_order = sorted(
        range(len(ordered_columns)),
        key=lambda position: (
            keyed_ranges[position][0],
            keyed_ranges[position][1],
        ),
        reverse=(direction == "rtl"),
    )
    reading: list[int] = []
    for position in column_order:
        members = ordered_columns[position]
        if direction == "ltr":
            members = sorted(members, key=lambda i: (extents[i][3], extents[i][2], i))
        else:
            members = sorted(members, key=lambda i: (extents[i][3], -extents[i][2], i))
        reading.extend(members)
    return reading


__all__ = ["order_lines"]
