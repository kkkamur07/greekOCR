"""Annotation store — load/save per-page segment JSON."""

import hashlib
import json
from pathlib import Path

from fastapi import HTTPException

from annote.schemas.annotation import PageAnnotation


def annotation_path(data_root: Path, stem: str) -> Path:
    return data_root / "annotations" / "pages" / f"{stem}.json"


def load_annotation(data_root: Path, stem: str) -> PageAnnotation:
    path = annotation_path(data_root, stem)
    if not path.is_file():
        return PageAnnotation()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PageAnnotation.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Corrupt annotation file: {stem}") from exc


def save_annotation(data_root: Path, stem: str, annotation: PageAnnotation) -> PageAnnotation:
    path = annotation_path(data_root, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    return annotation


def annotation_content_hash(annotation: PageAnnotation) -> str:
    payload = annotation.model_dump(exclude={"export_metadata", "locked"})
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
