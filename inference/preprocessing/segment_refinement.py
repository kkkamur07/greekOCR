"""Line segment refinement orchestration for BLLA ceiling polygons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from inference.preprocessing.otsu_contours import (
    cluster_contours_by_vertical_gap,
    combine_contours,
    otsu_band_contours,
)
from inference.preprocessing.segment_geometry import (
    MIN_VERTEX_SPACING_PX,
    bbox,
    bottom_edge_baseline,
    clean_polygon,
    clip_baseline_to_x_span,
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


def _cluster_baseline(
    contour: list[list[float]],
    baseline: list[list[float]] | None,
) -> tuple[list[list[float]], str]:
    """The baseline belonging to one cluster of a ceiling Otsu split apart.

    The decoder emitted one baseline for the whole ceiling, and the split bands
    are stacked vertically, so that baseline belongs to at most one of them.
    Clip it to the band's horizontal span and keep it when it also falls inside
    the band vertically: that band is the line the decoder actually measured.

    Every other band has no measured baseline. Deriving one from its own bottom
    edge is at least a text baseline; the bounding-box mid-height this used to
    return is a strike-through, and a transcriber who accepts the split line
    inherits it.
    """
    x0, y0, x1, y1 = bbox(contour)
    clipped = clip_baseline_to_x_span(baseline, x0, x1)
    if clipped and all(y0 <= point[1] <= y1 for point in clipped):
        return clipped, "decoder"
    return bottom_edge_baseline(contour), "bottom_edge"


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
    if split_count == 1:
        # Nothing was split off, so the decoded baseline still describes the
        # whole line and is passed through untouched.
        cluster_baseline = baseline
        baseline_source = "decoder" if baseline else "none"
    else:
        cluster_baseline, baseline_source = _cluster_baseline(contour, baseline)

    points, metrics = simplify_with_quality_gate(
        contour,
        width=width,
        height=height,
        target_max_points=target_max_points,
        min_iou=min_iou,
        min_area_ratio=min_area_ratio,
        # A baseline the decoder measured is independent evidence about where
        # the text sits, so the gate must keep the mask around it. One derived
        # from this cluster's own bottom edge is not - it moves with the mask -
        # and holding the mask to it would only ever be circular.
        baseline=cluster_baseline if baseline_source == "decoder" else None,
    )
    if len(points) < 4:
        points = fallback
        metrics = {"simplification_status": "fallback_after_invalid_simplification"}

    return SegmentRefinementResult(
        points=points,
        baseline=cluster_baseline,
        metadata={
            "baseline_source": baseline_source,
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


def grayscale_image(image: Image.Image) -> np.ndarray | None:
    """One grayscale conversion per page, or ``None`` when cv2 is unavailable.

    ``refine_segment_candidates`` runs once per decoded line, and converting the
    whole page to grayscale on every call was a full-page RGB->gray pass per
    line. The caller converts once and passes the result down; ``None`` (cv2
    missing) falls back to the per-call conversion inside the callee.
    """
    try:
        import cv2
    except ImportError:
        return None
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def refine_segment_candidates(
    image: Image.Image,
    ceiling: list[list[float]],
    *,
    baseline: list[list[float]] | None = None,
    gray: np.ndarray | None = None,
    margin_px: float = REFINEMENT_MARGIN_PX,
    target_max_points: int = TARGET_MAX_POINTS,
    min_iou: float = MIN_IOU,
    min_area_ratio: float = MIN_AREA_RATIO,
    split_large_lines: bool = True,
    split_vertical_gap_px: float = SPLIT_VERTICAL_GAP_PX,
) -> list[SegmentRefinementResult]:
    """Refine a ceiling polygon, optionally splitting over-merged text bands."""
    raw_point_count = len(ceiling)
    if len(ceiling) < 3:
        return _fallback_result(
            ceiling,
            raw_point_count=raw_point_count,
            status="invalid_ceiling",
        )

    min_spacing = max(MIN_VERTEX_SPACING_PX, 0.02 * line_height(ceiling))
    fallback = clean_polygon(ceiling, min_distance=min_spacing, min_vertices=4)

    try:
        import cv2
    except ImportError:
        return _fallback_result(
            fallback,
            raw_point_count=raw_point_count,
            status="opencv_unavailable",
        )

    if gray is None:
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
