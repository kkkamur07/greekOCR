"""Build Kraken segmentation documents from refined annotation JSON."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable

from kraken.containers import Region, Segmentation


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def close_polygon(points: list[list[float]]) -> list[tuple[int, int]]:
    """Convert app polygon points to closed Kraken integer coordinates."""
    closed = [(int(round(x)), int(round(y))) for x, y in points]
    if len(closed) >= 3 and closed[0] != closed[-1]:
        closed.append(closed[0])
    return closed


def iter_annotation_files(annotations_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in annotations_dir.glob("*.json") if path.name != "_state.json")


def resolve_image(stem: str, data_root: Path, *, prefer_processed: bool) -> Path | None:
    image_dirs = [
        data_root / "manuscripts" / "pages_processed",
        data_root / "manuscripts" / "pages",
    ]
    if not prefer_processed:
        image_dirs.reverse()

    for image_dir in image_dirs:
        for ext in IMAGE_EXTENSIONS:
            path = image_dir / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def build_segmentation_documents(
    *,
    data_root: Path,
    annotations_dir: Path,
    region_name: str,
    use_kraken_ceiling: bool,
    prefer_processed: bool,
) -> list[dict[str, Segmentation]]:
    """Build in-memory Kraken training docs from annotation app JSON."""
    docs: list[dict[str, Segmentation]] = []
    skipped = 0

    for annotation_path in iter_annotation_files(annotations_dir):
        stem = annotation_path.stem
        image_path = resolve_image(stem, data_root, prefer_processed=prefer_processed)
        if image_path is None:
            LOGGER.warning("Skipping %s: no matching image found", stem)
            skipped += 1
            continue

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        regions: list[Region] = []
        for segment in payload.get("segments", []):
            raw_points = segment.get("kraken_ceiling") if use_kraken_ceiling else segment.get("points")
            if not raw_points or len(raw_points) < 3:
                continue
            boundary = close_polygon(raw_points)
            if len(boundary) < 4:
                continue
            regions.append(
                Region(
                    id=str(segment.get("id") or f"{stem}-{len(regions) + 1}"),
                    boundary=boundary,
                    imagename=str(image_path),
                    tags={"type": [{"type": region_name}]},
                )
            )

        if not regions:
            LOGGER.warning("Skipping %s: no usable polygon regions", stem)
            skipped += 1
            continue

        docs.append(
            {
                "doc": Segmentation(
                    type="baselines",
                    imagename=str(image_path),
                    text_direction="horizontal-lr",
                    script_detection=False,
                    lines=[],
                    regions={region_name: regions},
                )
            }
        )

    LOGGER.info("Prepared %d page annotations; skipped %d", len(docs), skipped)
    return docs


def split_documents(
    docs: list[dict[str, Segmentation]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Segmentation]], list[dict[str, Segmentation]]]:
    if len(docs) < 2:
        raise ValueError("Need at least two annotated pages to create train/validation splits.")

    docs = docs[:]
    random.Random(seed).shuffle(docs)
    val_count = max(1, int(round(len(docs) * validation_ratio)))
    val_count = min(val_count, len(docs) - 1)
    return docs[val_count:], docs[:val_count]
