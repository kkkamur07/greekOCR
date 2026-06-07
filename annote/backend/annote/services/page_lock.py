"""Page lock — freeze annotation edits until explicitly unlocked."""

from pathlib import Path

from fastapi import HTTPException

from annote.schemas.annotation import PageAnnotation
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.page_catalogue import resolve_page_image


def _ensure_page_exists(data_root: Path, stem: str) -> None:
    image_path = resolve_page_image(data_root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"Page not found: {stem}")


def lock_page(data_root: Path, stem: str) -> PageAnnotation:
    _ensure_page_exists(data_root, stem)
    annotation = load_annotation(data_root, stem)
    annotation.locked = True
    return save_annotation(data_root, stem, annotation)


def unlock_page(data_root: Path, stem: str) -> PageAnnotation:
    _ensure_page_exists(data_root, stem)
    annotation = load_annotation(data_root, stem)
    annotation.locked = False
    return save_annotation(data_root, stem, annotation)


def assert_page_unlocked(annotation: PageAnnotation) -> None:
    if annotation.locked:
        raise HTTPException(status_code=409, detail="Page is locked")
