"""Preview rectified segment crops without writing export files."""

import io
from pathlib import Path

from annote.services.annotation_store import load_annotation
from annote.services.export_service import resolve_export_steps
from annote.services.page_image import load_working_page_rgb, resolve_source_page_image
from annote.services.processing.pipeline import process


def preview_segment_jpeg(data_root: Path, stem: str, segment_id: str) -> bytes:
    """Return a JPEG preview of the rectified crop for one segment."""
    annotation = load_annotation(data_root, stem)
    segment = next((s for s in annotation.segments if s.id == segment_id), None)
    if segment is None:
        raise LookupError(f"Segment not found: {segment_id}")

    if resolve_source_page_image(data_root, stem) is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    page_image, _ = load_working_page_rgb(data_root, stem)
    crop = process(page_image, segment.model_dump(), resolve_export_steps())

    from PIL import Image

    pil = Image.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()
