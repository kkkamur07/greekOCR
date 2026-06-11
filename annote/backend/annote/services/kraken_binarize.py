"""Kraken nlbin whole-page binarization."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from annote.schemas.annotation import PageAnnotation
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.page_image import (
    load_source_page_rgb,
    processed_page_path,
    save_processed_page_png,
)


def _kraken_install_hint(cause: ImportError) -> str:
    return (
        f"Kraken is not available in this Python ({sys.executable}): {cause}. "
        "Stop the server, then from annote/backend run: "
        "source .venv/bin/activate && pip install -e '.[kraken]' && annote"
    )


def binarize_image(image: Image.Image) -> Image.Image:
    """Binarize a page image with Kraken nlbin."""
    try:
        from kraken import binarization
    except ImportError as e:
        raise RuntimeError(_kraken_install_hint(e)) from e

    gray = image.convert("L")
    return binarization.nlbin(gray).convert("RGB")


def binarize_page(data_root: Path, stem: str) -> PageAnnotation:
    """Binarize the source page and persist a processed PNG for downstream work."""
    _, source_pil = load_source_page_rgb(data_root, stem)
    binarized = binarize_image(source_pil)
    out_path = processed_page_path(data_root, stem)
    save_processed_page_png(binarized, out_path)

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    annotation = load_annotation(data_root, stem)
    updated = annotation.model_copy(update={"binarized_at": now})
    return save_annotation(data_root, stem, updated)


def clear_binarized_page(data_root: Path, stem: str) -> PageAnnotation:
    """Revert to the original manuscript image for display and processing."""
    out_path = processed_page_path(data_root, stem)
    if out_path.is_file():
        out_path.unlink()

    annotation = load_annotation(data_root, stem)
    updated = annotation.model_copy(update={"binarized_at": None})
    return save_annotation(data_root, stem, updated)
