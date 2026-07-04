"""Kraken segmentation adapter for the ML service."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ml.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse


def _getattr_or_item(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _coerce_points(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("points") or value.get("boundary") or value.get("baseline")
    points: list[list[float]] = []
    for point in value or []:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        else:
            try:
                x, y = point[:2]
            except (TypeError, ValueError):
                continue
        points.append([float(x), float(y)])
    return points


def _box_from_points(points: list[list[float]]) -> dict[str, list[list[float]]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "points": [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ]
    }


def _line_points(line: Any) -> list[list[float]]:
    return (
        _coerce_points(_getattr_or_item(line, "boundary"))
        or _coerce_points(_getattr_or_item(line, "mask"))
        or _coerce_points(_getattr_or_item(line, "points"))
    )


def _baseline_points(line: Any, fallback: list[list[float]]) -> list[list[float]]:
    return _coerce_points(_getattr_or_item(line, "baseline")) or fallback


def _iter_regions(segmentation: Any) -> list[Any]:
    regions = _getattr_or_item(segmentation, "regions")
    if not regions:
        return []
    if isinstance(regions, dict):
        flattened: list[Any] = []
        for values in regions.values():
            flattened.extend(values or [])
        return flattened
    return list(regions)


def run_kraken_segment(image_bytes: bytes, *, weights_path: Path) -> SegmentRunResponse:
    """Run Kraken BLLA segmentation with the configured local model weights."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Kraken weights not found: {weights_path}")

    try:
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks import SegmentationTaskModel
    except ImportError as exc:  # pragma: no cover - depends on optional kraken extra
        raise RuntimeError("Kraken is not installed; install the kraken optional extra") from exc

    with Image.open(BytesIO(image_bytes)) as image:
        segmentation = SegmentationTaskModel.load_model(weights_path).predict(
            image.convert("RGB"),
            SegmentationInferenceConfig(),
        )

    blocks: list[SegmentBlock] = []
    for order, region in enumerate(_iter_regions(segmentation)):
        points = _coerce_points(_getattr_or_item(region, "boundary"))
        if len(points) < 4:
            continue
        external_id = str(_getattr_or_item(region, "id") or f"kraken-block-{order + 1}")
        blocks.append(
            SegmentBlock(
                external_id=external_id,
                order=order,
                box={"points": points},
            )
        )

    lines: list[SegmentLine] = []
    for order, line in enumerate(_getattr_or_item(segmentation, "lines") or []):
        points = _line_points(line)
        if len(points) < 4:
            continue
        external_id = str(_getattr_or_item(line, "id") or f"kraken-line-{order + 1}")
        lines.append(
            SegmentLine(
                external_id=external_id,
                order=order,
                block_external_id=None,
                baseline={"points": _baseline_points(line, points)},
                mask={"points": points},
                points=points,
                kraken_ceiling=points,
                source_metadata={
                    "adapter": "kraken",
                    "weights_path": str(weights_path),
                },
            )
        )

    if not blocks and lines:
        all_points = [point for line in lines for point in line.points]
        blocks.append(
            SegmentBlock(
                external_id="kraken-block-1",
                order=0,
                box=_box_from_points(all_points),
            )
        )
        for line in lines:
            line.block_external_id = "kraken-block-1"

    return SegmentRunResponse(blocks=blocks, lines=lines)
