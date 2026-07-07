"""Kraken segmentation inference adapter."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse
from inference.preprocessing.segment_geometry import simplify_kraken_boundary
from inference.preprocessing.segment_refinement import (
    MIN_AREA_RATIO,
    MIN_IOU,
    SPLIT_VERTICAL_GAP_PX,
    TARGET_MAX_POINTS,
    SegmentRefinementResult,
    refine_segment_candidates,
)


class KrakenUnavailableError(RuntimeError):
    """Raised when the optional Kraken runtime is not installed."""


def _point_pair(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _coerce_points(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []

    if len(value) == 4 and all(not isinstance(item, (list, tuple)) for item in value):
        try:
            x0, y0, x1, y1 = [float(item) for item in value]
        except (TypeError, ValueError):
            return []
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    points = [_point_pair(item) for item in value]
    return [point for point in points if point is not None]


def _polygon_from_baseline(baseline: list[list[float]]) -> list[list[float]]:
    xs = [point[0] for point in baseline]
    ys = [point[1] for point in baseline]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y0 == y1:
        y0 -= 1.0
        y1 += 1.0
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line_items(raw_result: Any) -> list[Any]:
    lines = getattr(raw_result, "lines", None)
    if isinstance(lines, list):
        return lines
    if isinstance(raw_result, dict):
        for key in ("lines", "boxes"):
            value = raw_result.get(key)
            if isinstance(value, list):
                return value
    if isinstance(raw_result, list):
        return raw_result
    return []


def _line_value(item: Any, *keys: str) -> Any:
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item[key]
    for key in keys:
        value = getattr(item, key, None)
        if value is not None:
            return value
    return None


@lru_cache(maxsize=4)
def _load_segmentation_model(model_path: str) -> Any:
    try:
        from kraken.lib import vgsl
    except ImportError as exc:
        raise KrakenUnavailableError(
            "Kraken is required for real segmentation; install the project with the `kraken` extra"
        ) from exc

    return vgsl.TorchVGSLModel.load_model(model_path)


def _run_blla_segment(image: Image.Image, model: Any) -> Any:
    try:
        from kraken import blla
    except ImportError as exc:
        raise KrakenUnavailableError(
            "Kraken BLLA is required for real segmentation; "
            "install the project with the `kraken` extra"
        ) from exc

    return blla.segment(image, model=model, raise_on_error=True)


def _bool_param(params: dict[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_float_param(params: dict[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_int_param(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def run_kraken_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    if not model_path.exists():
        raise FileNotFoundError(f"Kraken model not found: {model_path}")

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

    model = _load_segmentation_model(str(model_path))
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        raw_result = _run_blla_segment(image, model)
        refinement_image = image.copy()

    block = SegmentBlock(
        external_id="kraken-block-1",
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
    for order, item in enumerate(_line_items(raw_result)):
        baseline = _coerce_points(_line_value(item, "baseline"))
        raw_points = _coerce_points(_line_value(item, "boundary", "bounds", "polygon", "bbox"))
        points = raw_points
        if not points and baseline:
            points = _polygon_from_baseline(baseline)
        kraken_ceiling = points
        if not baseline:
            baseline = points
        if len(points) < 4 or len(baseline) < 2:
            continue

        source_metadata: dict[str, Any] = {"adapter": "kraken", "raw_order": order}
        if use_otsu_refinement:
            refinements = refine_segment_candidates(
                refinement_image,
                points,
                baseline=baseline,
                margin_px=otsu_sphere_radius,
                target_max_points=target_max_points,
                min_iou=min_iou,
                min_area_ratio=min_area_ratio,
                split_large_lines=split_large_lines,
                split_vertical_gap_px=split_vertical_gap_px,
            )
        else:
            simplified_points, simplify_metrics = simplify_kraken_boundary(points)
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
                    external_id=f"kraken-line-{order + 1}-{split_index + 1}",
                    order=len(lines),
                    block_external_id=block.external_id,
                    baseline={"points": line_baseline},
                    mask={"points": refined_points},
                    points=refined_points,
                    kraken_ceiling=kraken_ceiling,
                    source_metadata=line_metadata,
                )
            )

    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)
