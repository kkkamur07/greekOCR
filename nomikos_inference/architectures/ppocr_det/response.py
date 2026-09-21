"""SegmentRunResponse construction for PP-OCRv6 detection quads and polygons."""

from __future__ import annotations

from typing import Any

from nomikos_inference.architectures.ppocr_det.polygons import (
    DEFAULT_OUTLINE_TOLERANCE_PX,
    DEFAULT_VERTICAL_GROWTH,
    grow_outlines_vertical,
    polyline_baseline,
    simplify_outline,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout, order_lines
from nomikos_inference.contracts.common import MAX_SEGMENT_LINES
from nomikos_inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse

#: Where across the quad the synthetic baseline sits, from the top edge (0)
#: to the bottom edge (1). PP-OCRv6 emits boxes, not baselines, and kraken
#: baselines sit near the bottom of the letter bodies, so the default puts
#: the polyline three quarters down the quad sides.
DEFAULT_BASELINE_FRACTION = 0.75


def synthetic_baseline_points(quad: list[list[float]], fraction: float) -> list[list[float]]:
    """Two-point baseline across a clockwise quad at ``fraction`` down."""
    top_left, top_right, bottom_right, bottom_left = quad
    left = [
        top_left[0] + fraction * (bottom_left[0] - top_left[0]),
        top_left[1] + fraction * (bottom_left[1] - top_left[1]),
    ]
    right = [
        top_right[0] + fraction * (bottom_right[0] - top_right[0]),
        top_right[1] + fraction * (bottom_right[1] - top_right[1]),
    ]
    return [left, right]


def _baseline_points(quad: list[list[float]], fraction: float) -> list[list[float]]:
    return synthetic_baseline_points(quad, fraction)


def _poly_points_for(
    quad: DetectedQuad,
    baseline_fraction: float,
    image_width: int,
    image_height: int,
    outline_tolerance_px: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], bool]:
    """Ungrown polygon, quad ring and baseline polyline for one detection.

    Returns the simplified outline, the quad ring and baseline the growth
    step needs for direction, the baseline polyline sampled from the ungrown
    outline, and whether the quad stood in as the polygon.
    """
    quad_points = [[float(x), float(y)] for x, y in quad.points]
    outline = quad.polygon if quad.polygon is not None else quad_points
    fallback = bool(quad.polygon_fallback) or quad.polygon is None
    quad_baseline = _baseline_points(quad_points, baseline_fraction)
    simplified = simplify_outline(
        outline,
        outline_tolerance_px=outline_tolerance_px,
        page_width=image_width,
        page_height=image_height,
    )
    if simplified is None:
        return quad_points, quad_points, quad_baseline, True
    baseline = polyline_baseline(simplified, quad_points, baseline_fraction, baseline=quad_baseline)
    if baseline is None:
        baseline = quad_baseline
    return simplified, quad_points, baseline, fallback


def build_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
    box_type: str = "quad",
    outline_tolerance_px: float = DEFAULT_OUTLINE_TOLERANCE_PX,
    vertical_growth: float = DEFAULT_VERTICAL_GROWTH,
) -> SegmentRunResponse:
    """Number detection quads in reading order under one full-page block.

    Past ``MAX_SEGMENT_LINES`` the highest scoring quads survive (score ties
    keep input order), then the survivors are ordered; numbering always
    follows reading order from 1. With ``box_type="poly"`` each line serves
    its polygon grown across the line and a baseline polyline sampled from
    the ungrown outline; the quad path is unchanged.
    """

    if box_type not in ("quad", "poly"):
        raise ValueError('box_type must be "poly" or "quad"')
    ranked = sorted(range(len(quads)), key=lambda i: (-quads[i].score, i))
    survivors = [quads[i] for i in ranked[:MAX_SEGMENT_LINES]]
    reading = order_lines(survivors, direction=reading_direction)
    per_survivor: list[tuple] = []
    grown_points: list[list[list[float]]] = []
    if box_type == "poly":
        for quad in survivors:
            per_survivor.append(
                _poly_points_for(
                    quad, baseline_fraction, image_width, image_height, outline_tolerance_px
                )
            )
        grown_points = grow_outlines_vertical(
            [simplified for simplified, _, _, _ in per_survivor],
            [ring for _, ring, _, _ in per_survivor],
            factor=float(vertical_growth),
            baselines=[_baseline_points(quad.points, baseline_fraction) for quad in survivors],
            page_width=image_width,
            page_height=image_height,
            outline_tolerance_px=outline_tolerance_px,
        )

    block = SegmentBlock(
        external_id="ppocr-det-block-1",
        order=0,
        box={
            "points": [
                [0.0, 0.0],
                [float(image_width), 0.0],
                [float(image_width), float(image_height)],
                [0.0, float(image_height)],
            ]
        },
    )

    lines = []
    for position, quad_index in enumerate(reading):
        quad = survivors[quad_index]
        source_metadata: dict[str, Any] = {
            "detector": "pp-ocrv6-det",
            "score": float(quad.score),
            "baseline_source": "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
        if box_type == "poly":
            _, _, served_baseline, fallback = per_survivor[quad_index]
            if fallback:
                source_metadata["polygon_fallback"] = True
            lines.append(
                SegmentLine(
                    external_id=f"ppocr-det-line-{position + 1}",
                    order=position,
                    block_external_id=block.external_id,
                    baseline={"points": served_baseline},
                    mask=None,
                    points=grown_points[quad_index],
                    kraken_ceiling=None,
                    source_metadata=source_metadata,
                )
            )
            continue
        lines.append(
            SegmentLine(
                external_id=f"ppocr-det-line-{position + 1}",
                order=position,
                block_external_id=block.external_id,
                baseline={"points": _baseline_points(quad.points, baseline_fraction)},
                mask=None,
                points=[[float(x), float(y)] for x, y in quad.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)


def build_refined_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    items: list,
    layout: PageLayout,
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
    noise_policy: str = "flag",
    box_type: str = "quad",
) -> SegmentRunResponse:
    """Number refined lines in reading order under one full-page block.

    ``items`` are refinement output whose ``members`` name the original quad
    indices, so the body order follows ``order_lines`` over the raw quads:
    each refined line is emitted at its first member's position, initials
    first within their row group, and suspects (unless dropped) after every
    body line, top to bottom. Past ``MAX_SEGMENT_LINES`` the highest
    scoring items survive, ties broken geometrically.
    """

    if box_type not in ("quad", "poly"):
        raise ValueError('box_type must be "poly" or "quad"')
    scoped = [item for item in items if not (noise_policy == "drop" and item.suspect)]

    def _mid(item: Any) -> tuple[float, float]:
        first, second = item.baseline[0], item.baseline[-1]
        return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)

    ranked = sorted(
        range(len(scoped)),
        key=lambda i: (-scoped[i].score, _mid(scoped[i])[1], _mid(scoped[i])[0]),
    )
    survivors = [scoped[i] for i in ranked[:MAX_SEGMENT_LINES]]
    owner: dict[int, Any] = {}
    for item in survivors:
        for member in item.members:
            owner.setdefault(member, item)
    emission: list[Any] = []
    seen: set[int] = set()
    for quad_index in order_lines(quads, direction=reading_direction):
        item = owner.get(quad_index)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        emission.append(item)

    group_of: dict[int, tuple[int, int]] = {}
    for column_index, column in enumerate(layout.columns):
        for row_index, row in enumerate(column.rows):
            for member in row:
                group_of[member] = (column_index, row_index)
    slots: dict[tuple[int, int] | None, list[int]] = {}
    for position, item in enumerate(emission):
        groups = {group_of.get(member) for member in item.members} - {None}
        key = next(iter(groups)) if len(groups) == 1 else None
        slots.setdefault(key, []).append(position)
    for key, positions in slots.items():
        if key is None:
            continue
        first = [position for position in positions if emission[position].role == "initial"]
        rest = [position for position in positions if emission[position].role != "initial"]
        ordered = [emission[position] for position in first + rest]
        for position, item in zip(positions, ordered, strict=True):
            emission[position] = item

    if noise_policy == "flag":
        body = [item for item in emission if not item.suspect]
        tail = sorted(
            (item for item in emission if item.suspect),
            key=lambda item: (_mid(item)[1], _mid(item)[0]),
        )
        emission = body + tail

    block = SegmentBlock(
        external_id="ppocr-det-block-1",
        order=0,
        box={
            "points": [
                [0.0, 0.0],
                [float(image_width), 0.0],
                [float(image_width), float(image_height)],
                [0.0, float(image_height)],
            ]
        },
    )

    lines = []
    for position, item in enumerate(emission):
        source_metadata: dict[str, Any] = {
            "detector": "pp-ocrv6-det",
            "score": float(item.score),
            "baseline_source": "merged_axis" if item.merged_from > 1 else "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
        if item.role == "initial":
            source_metadata["role"] = "initial"
        if item.merged_from > 1:
            source_metadata["merged_from"] = item.merged_from
        if item.suspect:
            source_metadata["suspect"] = True
            source_metadata["suspect_reason"] = item.suspect_reason
        if item.overlap_unresolved:
            source_metadata["overlap_unresolved"] = True
        if box_type == "poly" and getattr(item, "polygon_fallback", False):
            source_metadata["polygon_fallback"] = True
        lines.append(
            SegmentLine(
                external_id=f"ppocr-det-line-{position + 1}",
                order=position,
                block_external_id=block.external_id,
                baseline={"points": [[float(x), float(y)] for x, y in item.baseline]},
                mask=None,
                points=[[float(x), float(y)] for x, y in item.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)


__all__ = [
    "build_ppocr_det_response",
    "build_refined_ppocr_det_response",
    "synthetic_baseline_points",
]
