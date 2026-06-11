"""Resolve the working page image (original or Kraken-binarized)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from annote.services.annotation_store import load_annotation
from annote.services.image_export import load_page_rgb
from annote.services.page_catalogue import resolve_page_image


def processed_page_path(data_root: Path, stem: str) -> Path:
    return data_root / "manuscripts" / "pages_processed" / f"{stem}.png"


def pages_dir(data_root: Path) -> Path:
    return data_root / "manuscripts" / "pages"


def is_binarized_active(data_root: Path, stem: str) -> bool:
    annotation = load_annotation(data_root, stem)
    if annotation.binarized_at is None:
        return False
    return processed_page_path(data_root, stem).is_file()


def resolve_source_page_image(data_root: Path, stem: str) -> Path | None:
    """Original manuscript page on disk."""
    return resolve_page_image(pages_dir(data_root), stem)


def resolve_working_page_image(data_root: Path, stem: str) -> Path | None:
    """Page image used for display, segmentation, OCR, and export."""
    if is_binarized_active(data_root, stem):
        return processed_page_path(data_root, stem)
    return resolve_source_page_image(data_root, stem)


def load_working_page_rgb(data_root: Path, stem: str) -> tuple:
    """Load the working page as an RGB numpy array."""
    image_path = resolve_working_page_image(data_root, stem)
    if image_path is None:
        raise FileNotFoundError(f"Page image not found: {stem}")
    return load_page_rgb(image_path)


def load_source_page_rgb(data_root: Path, stem: str) -> tuple:
    """Load the original manuscript page as RGB (for binarization input)."""
    image_path = resolve_source_page_image(data_root, stem)
    if image_path is None:
        raise FileNotFoundError(f"Page image not found: {stem}")
    return load_page_rgb(image_path)


def save_processed_page_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    rgb.save(path, format="PNG")
