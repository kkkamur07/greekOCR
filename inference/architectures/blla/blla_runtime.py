"""Torch-free BLLA decoding and segment-contract conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from inference.architectures.blla.blla_decoder import decode_blla_heatmaps
from inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse
from inference.preprocessing.segment_geometry import simplify_blla_boundary
from inference.preprocessing.segment_refinement import (
    MIN_AREA_RATIO,
    MIN_IOU,
    SPLIT_VERTICAL_GAP_PX,
    TARGET_MAX_POINTS,
    SegmentRefinementResult,
    refine_segment_candidates,
)
from inference.architectures.blla.blla_preprocessing import BLLAInput, BLLANumpyInput


def _bool_param(params: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_float_param(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_int_param(params: Mapping[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def build_blla_segment_response(
    image: Image.Image,
    logits: np.ndarray,
    prepared: BLLAInput | BLLANumpyInput,
    *,
    params: Mapping[str, Any] | None = None,
    torch_free: bool = False,
) -> SegmentRunResponse:
    """Decode logits and preserve the native BLLA response contract."""

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("BLLA logits must have shape (4, height, width)")

    params = params or {}
    use_otsu_refinement = _bool_param(params, "use_otsu_refinement")
    otsu_sphere_radius = _positive_float_param(params, "otsu_sphere_radius", 4.0)
    target_max_points = _positive_int_param(params, "target_max_points", TARGET_MAX_POINTS)
    min_iou = _positive_float_param(params, "min_iou", MIN_IOU)
    min_area_ratio = _positive_float_param(params, "min_area_ratio", MIN_AREA_RATIO)
    split_large_lines = _bool_param(params, "split_large_lines", True)
    split_vertical_gap_px = _positive_float_param(
        params,
        "split_vertical_gap_px",
        SPLIT_VERTICAL_GAP_PX,
    )
    threshold = _positive_float_param(params, "heatmap_threshold", 0.17)
    threshold = min(threshold, 0.99)

    width, height = image.size
    decoded_lines = decode_blla_heatmaps(
        values,
        image_size=(width, height),
        threshold=threshold,
        raw_logits=True,
        scaled_gray=prepared.scaled_gray,
        torch_free=torch_free,
    )
    refinement_image = image.copy()

    block = SegmentBlock(
        external_id="blla-block-1",
        order=0,
        box={
            "points": [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ]
        },
    )

    lines: list[SegmentLine] = []
    for order, decoded in enumerate(decoded_lines):
        baseline = decoded.baseline
        ceiling = decoded.polygon
        if len(ceiling) < 4 or len(baseline) < 2:
            continue

        source_metadata: dict[str, Any] = {
            "adapter": "blla",
            "decoder": "native",
            "raw_order": order,
        }
        if use_otsu_refinement:
            refinements = refine_segment_candidates(
                refinement_image,
                ceiling,
                baseline=baseline,
                margin_px=otsu_sphere_radius,
                target_max_points=target_max_points,
                min_iou=min_iou,
                min_area_ratio=min_area_ratio,
                split_large_lines=split_large_lines,
                split_vertical_gap_px=split_vertical_gap_px,
            )
        else:
            simplified_points, simplify_metrics = simplify_blla_boundary(ceiling)
            source_metadata.update(simplify_metrics)
            refinements = [
                SegmentRefinementResult(
                    points=simplified_points,
                    baseline=baseline,
                    metadata=source_metadata,
                )
            ]

        for split_index, refinement in enumerate(refinements):
            refined_points = refinement.points
            line_baseline = refinement.baseline or baseline
            line_metadata = {
                **source_metadata,
                **refinement.metadata,
            }
            lines.append(
                SegmentLine(
                    external_id=f"blla-line-{order + 1}-{split_index + 1}",
                    order=len(lines),
                    block_external_id=block.external_id,
                    baseline={"points": line_baseline},
                    mask={"points": refined_points},
                    points=refined_points,
                    kraken_ceiling=ceiling,
                    source_metadata=line_metadata,
                )
            )

    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)
