"""Preview rectified segment crops without writing export files."""

import io
from pathlib import Path

from annote.services.annotation_store import load_annotation
from annote.services.export_service import resolve_export_steps
from annote.services.image_export import load_page_rgb
from annote.services.page_catalogue import resolve_page_image
from annote.services.processing.pipeline import process


def preview_segment_jpeg(data_root: Path, stem: str, segment_id: str) -> bytes:
    """Return a JPEG preview of the rectified crop for one segment."""
    annotation = load_annotation(data_root, stem)
    segment = next((s for s in annotation.segments if s.id == segment_id), None)
    if segment is None:
        raise LookupError(f"Segment not found: {segment_id}")

    image_path = resolve_page_image(data_root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    page_image, page_pil = load_page_rgb(image_path)
    crop = process(page_image, segment.model_dump(), resolve_export_steps())

    from PIL import Image

    pil = Image.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()
