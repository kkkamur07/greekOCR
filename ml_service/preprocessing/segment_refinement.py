"""Line segment refinement orchestration for Kraken ceilings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ml_service.preprocessing.otsu_contours import (
    cluster_contours_by_vertical_gap,
    combine_contours,
    otsu_band_contours,
)
from ml_service.preprocessing.segment_geometry import (
    MIN_VERTEX_SPACING_PX,
    clean_polygon,
    default_baseline,
    line_height,
    simplify_with_quality_gate,
)

REFINEMENT_MARGIN_PX = 4.0
TARGET_MAX_POINTS = 80
MIN_IOU = 0.97
MIN_AREA_RATIO = 0.95
SPLIT_VERTICAL_GAP_PX = 12.0


@dataclass(frozen=True)
class SegmentRefinementResult:
    points: list[list[float]]
    metadata: dict[str, Any]
    baseline: list[list[float]] | None = None


def _fallback_result(
    points: list[list[float]],
    *,
    raw_point_count: int,
    status: str,
) -> list[SegmentRefinementResult]:
    return [
        SegmentRefinementResult(
            points=points,
            metadata={
                "raw_point_count": raw_point_count,
                "simplified_point_count": len(points),
                "simplification_status": status,
            },
        )
    ]


def _refine_cluster(
    cluster: list[list[list[float]]],
    *,
    image_size: tuple[int, int],
    fallback: list[list[float]],
    baseline: list[list[float]] | None,
    split_index: int,
    split_count: int,
    raw_point_count: int,
    margin_px: float,
    target_max_points: int,
    min_iou: float,
    min_area_ratio: float,
    split_large_lines: bool,
    split_vertical_gap_px: float,
) -> SegmentRefinementResult:
    width, height = image_size
    contour = combine_contours(cluster)
    points, metrics = simplify_with_quality_gate(
        contour,
        width=width,
        height=height,
        target_max_points=target_max_points,
        min_iou=min_iou,
        min_area_ratio=min_area_ratio,
        baseline=baseline if split_count == 1 else None,
    )
    if len(points) < 3:
        points = fallback
        metrics = {"simplification_status": "fallback_after_invalid_simplification"}

    return SegmentRefinementResult(
        points=points,
        baseline=baseline if split_count == 1 else default_baseline(points),
        metadata={
            **metrics,
            "raw_point_count": raw_point_count,
            "otsu_contour_point_count": sum(len(contour) for contour in cluster),
            "simplified_point_count": len(points),
            "otsu_margin_px": margin_px,
            "target_max_points": target_max_points,
            "split_large_lines": split_large_lines,
            "split_vertical_gap_px": split_vertical_gap_px,
            "split_index": split_index,
            "split_count": split_count,
        },
    )


def refine_segment(
    image: Image.Image,
    ceiling: list[list[float]],
    *,
    baseline: list[list[float]] | None = None,
    margin_px: float = REFINEMENT_MARGIN_PX,
    target_max_points: int = TARGET_MAX_POINTS,
    min_iou: float = MIN_IOU,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> SegmentRefinementResult:
    """Refine one Kraken ceiling without splitting it into multiple lines."""
    return refine_segment_candidates(
        image,
        ceiling,
        baseline=baseline,
        margin_px=margin_px,
        target_max_points=target_max_points,
        min_iou=min_iou,
        min_area_ratio=min_area_ratio,
        split_large_lines=False,
    )[0]


def refine_segment_candidates(
    image: Image.Image,
    ceiling: list[list[float]],
    *,
    baseline: list[list[float]] | None = None,
    margin_px: float = REFINEMENT_MARGIN_PX,
    target_max_points: int = TARGET_MAX_POINTS,
    min_iou: float = MIN_IOU,
    min_area_ratio: float = MIN_AREA_RATIO,
    split_large_lines: bool = True,
    split_vertical_gap_px: float = SPLIT_VERTICAL_GAP_PX,
) -> list[SegmentRefinementResult]:
    """Refine a Kraken ceiling, optionally splitting over-merged vertical text bands."""
    raw_point_count = len(ceiling)
    if len(ceiling) < 3:
        return _fallback_result(
            ceiling,
            raw_point_count=raw_point_count,
            status="invalid_ceiling",
        )

    min_spacing = max(MIN_VERTEX_SPACING_PX, 0.02 * line_height(ceiling))
    fallback = clean_polygon(ceiling, min_distance=min_spacing)

    try:
        import cv2
    except ImportError:
        return _fallback_result(
            fallback,
            raw_point_count=raw_point_count,
            status="opencv_unavailable",
        )

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    contours = otsu_band_contours(gray, ceiling, margin_px=margin_px)
    if not contours:
        return _fallback_result(
            fallback,
            raw_point_count=raw_point_count,
            status="no_otsu_contour",
        )

    clusters = (
        cluster_contours_by_vertical_gap(contours, gap_px=split_vertical_gap_px)
        if split_large_lines
        else [contours]
    )
    if not clusters:
        clusters = [contours]

    return [
        _refine_cluster(
            cluster,
            image_size=image.size,
            fallback=fallback,
            baseline=baseline,
            split_index=split_index,
            split_count=len(clusters),
            raw_point_count=raw_point_count,
            margin_px=margin_px,
            target_max_points=target_max_points,
            min_iou=min_iou,
            min_area_ratio=min_area_ratio,
            split_large_lines=split_large_lines,
            split_vertical_gap_px=split_vertical_gap_px,
        )
        for split_index, cluster in enumerate(clusters)
    ]
