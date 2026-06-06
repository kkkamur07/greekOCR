"""Export state — dirty/clean tracking."""

from datetime import datetime, timezone
from pathlib import Path

from annote.schemas.annotation import ExportMetadata, PageAnnotation
from annote.services.annotation_store import annotation_content_hash, load_annotation, save_annotation


def is_export_dirty(data_root: Path, stem: str, annotation: PageAnnotation | None = None) -> bool:
    ann = annotation if annotation is not None else load_annotation(data_root, stem)
    if ann.export_metadata is None:
        return len(ann.segments) > 0 or any(s.paired_text_line_index for s in ann.segments)
    current_hash = annotation_content_hash(ann)
    return current_hash != ann.export_metadata.content_hash


def mark_exported(data_root: Path, stem: str, annotation: PageAnnotation) -> PageAnnotation:
    content_hash = annotation_content_hash(annotation)
    annotation.export_metadata = ExportMetadata(
        exported_at=datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash,
    )
    return save_annotation(data_root, stem, annotation)
