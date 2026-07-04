"""Kraken segmentation inference adapter."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ml_service.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse


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


def run_kraken_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
) -> SegmentRunResponse:
    if not model_path.exists():
        raise FileNotFoundError(f"Kraken model not found: {model_path}")

    model = _load_segmentation_model(str(model_path))
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        raw_result = _run_blla_segment(image, model)

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
        points = _coerce_points(_line_value(item, "boundary", "bounds", "polygon", "bbox"))
        if not points and baseline:
            points = _polygon_from_baseline(baseline)
        if not baseline:
            baseline = points
        if len(points) < 4 or len(baseline) < 2:
            continue

        lines.append(
            SegmentLine(
                external_id=f"kraken-line-{order + 1}",
                order=order,
                block_external_id=block.external_id,
                baseline={"points": baseline},
                mask={"points": points},
                points=points,
                kraken_ceiling=points,
                source_metadata={"adapter": "kraken", "raw_order": order},
            )
        )

    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)
