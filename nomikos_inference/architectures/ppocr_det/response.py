"""SegmentRunResponse construction for PP-OCRv6 detection quads."""

from __future__ import annotations

from typing import Any

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import order_lines
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


def build_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
) -> SegmentRunResponse:
    """Number detection quads in reading order under one full-page block.

    Past ``MAX_SEGMENT_LINES`` the highest scoring quads survive (score ties
    keep input order), then the survivors are ordered; numbering always
    follows reading order from 1.
    """

    ranked = sorted(range(len(quads)), key=lambda i: (-quads[i].score, i))
    survivors = [quads[i] for i in ranked[:MAX_SEGMENT_LINES]]
    reading = order_lines(survivors, direction=reading_direction)

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


__all__ = ["build_ppocr_det_response", "synthetic_baseline_points"]
