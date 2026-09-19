"""Column-aware reading order for PP-OCRv6 detection quads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import order_lines

DETECTION_DIR = Path(
    "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env/server/"
    "ppocrv6-results-paddle322-20260919"
)


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _ids_in_order(quads: list[DetectedQuad], direction: str = "ltr") -> list[int]:
    return order_lines(quads, direction=direction)


def test_single_column_orders_top_to_bottom() -> None:
    quads = [
        _quad(100, 300, 500, 330),
        _quad(100, 100, 500, 130),
        _quad(100, 200, 500, 230),
    ]

    assert _ids_in_order(quads) == [1, 2, 0]


def test_two_columns_read_left_page_then_right_page() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
    ]

    assert _ids_in_order(quads, direction="ltr") == [3, 1, 0, 2]
    assert _ids_in_order(quads, direction="rtl") == [0, 2, 3, 1]


def test_narrow_marginal_quad_keeps_its_place_in_the_column() -> None:
    quads = [
        _quad(100, 100, 500, 130),
        _quad(520, 150, 560, 180),
        _quad(100, 200, 500, 230),
    ]

    assert _ids_in_order(quads) == [0, 1, 2]


def test_narrow_quad_beside_two_columns_joins_the_nearest() -> None:
    quads = [
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
        _quad(920, 150, 950, 180),
        _quad(100, 200, 400, 230),
        _quad(600, 100, 900, 130),
    ]

    assert _ids_in_order(quads, direction="ltr") == [1, 3, 4, 2, 0]


def test_shuffled_input_gives_the_same_order() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(920, 150, 950, 180),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
        _quad(100, 300, 400, 330),
        _quad(600, 300, 900, 330),
    ]
    reference = [quads[i] for i in _ids_in_order(quads)]

    # A fixed non-identity permutation: reversing is enough to prove the
    # order does not follow the input sequence.
    shuffled = quads[::-1]
    rerun = [shuffled[i] for i in _ids_in_order(shuffled)]

    assert [id(quad) for quad in rerun] == [id(quad) for quad in reference]


def test_bad_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        order_lines([_quad(0, 0, 10, 10)], direction="diagonal")


def _load_quads(slug: str) -> list[DetectedQuad] | None:
    path = DETECTION_DIR / f"medium-1920-{slug}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        DetectedQuad(points=[[float(x), float(y)] for x, y in quad], score=float(score))
        for quad, score in zip(payload["polygons"], payload["scores"], strict=True)
    ]


def _assert_left_column_precedes_right(quads: list[DetectedQuad]) -> None:
    centres = [sum(point[0] for point in quad.points) / 4 for quad in quads]
    middle = max(point[0] for quad in quads for point in quad.points) / 2
    left = {i for i, cx in enumerate(centres) if cx < middle}
    right = {i for i, cx in enumerate(centres) if cx >= middle}
    positions = {quad_index: rank for rank, quad_index in enumerate(order_lines(quads))}
    if left and right:
        assert max(positions[i] for i in left) < min(positions[i] for i in right)


def test_real_spread_reads_left_column_before_right() -> None:
    """Left-before-right on the measured medium-1920 detector output.

    ``grec-p1`` is the file the brief names. In the measured JSON all 29 of
    its quads sit on the right folio (the left folio is blank, as the overlay
    beside the JSON shows), so the claim holds vacuously there; ``grec-p4``
    carries text on both folios and exercises the claim for real.
    """
    quads = _load_quads("grec-p1")
    if quads is None:
        pytest.skip("measured detector output is not present")
    _assert_left_column_precedes_right(quads)

    both = _load_quads("grec-p4")
    if both is None:
        pytest.skip("measured detector output is not present")
    middle = max(point[0] for quad in both for point in quad.points) / 2
    centres = [sum(point[0] for point in quad.points) / 4 for quad in both]
    assert any(cx < middle for cx in centres)
    assert any(cx >= middle for cx in centres)
    _assert_left_column_precedes_right(both)
