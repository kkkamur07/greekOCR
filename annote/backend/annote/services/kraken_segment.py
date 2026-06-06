"""Kraken BLLA line segmentation — default bundled model."""

from __future__ import annotations

import secrets
from importlib import resources
from pathlib import Path

from PIL import Image

from annote.schemas.annotation import PageAnnotation, Segment
from annote.services.annotation_store import load_annotation, save_annotation
from annote.services.page_catalogue import resolve_page_image
from annote.services.processing.polygon import merge_close_polygon_points
from annote.services.text_lines import split_text_lines

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from kraken.lib.vgsl import TorchVGSLModel
    except ImportError as e:
        raise RuntimeError(
            "Kraken is required for auto-segmentation. Install with: pip install 'annote[kraken]'"
        ) from e
    model_path = resources.files("kraken").joinpath("blla.mlmodel")
    _model = TorchVGSLModel.load_model(str(model_path))
    return _model


def _new_segment_id() -> str:
    return f"seg-{secrets.token_hex(4)}"


def _boundary_to_points(boundary) -> list[list[float]]:
    raw = [[float(x), float(y)] for x, y in boundary]
    return merge_close_polygon_points(raw)


def kraken_lines_to_segments(lines, *, start_number: int = 1) -> list[Segment]:
    """Convert Kraken segmentation lines to annote Segment DTOs."""
    segments: list[Segment] = []
    for idx, line in enumerate(lines):
        boundary = getattr(line, "boundary", None)
        if not boundary or len(boundary) < 3:
            continue
        segments.append(
            Segment(
                id=_new_segment_id(),
                number=start_number + idx,
                kind="polygon",
                points=_boundary_to_points(boundary),
                paired_text_line_index=None,
                text_override=None,
            )
        )
    return segments


def segment_image(image: Image.Image, *, device: str = "cpu") -> list[Segment]:
    """Run Kraken's default BLLA segmenter on a page image."""
    try:
        from kraken.blla import segment
    except ImportError as e:
        raise RuntimeError(
            "Kraken is required for auto-segmentation. Install with: pip install 'annote[kraken]'"
        ) from e

    model = _load_model()
    segmented = segment(im=image, device=device, model=model)
    return kraken_lines_to_segments(segmented.lines)


def pair_segments_to_transcription(
    segments: list[Segment],
    text_lines: list,
) -> list[Segment]:
    """Pair segments to text lines by reading order (1-based text line index)."""
    if not text_lines:
        return segments
    paired: list[Segment] = []
    for idx, seg in enumerate(segments):
        line_index = text_lines[idx].index if idx < len(text_lines) else None
        paired.append(seg.model_copy(update={"paired_text_line_index": line_index}))
    return paired


def auto_segment_page(
    data_root: Path,
    stem: str,
    *,
    replace: bool = True,
    pair_transcription: bool = True,
    device: str = "cpu",
) -> PageAnnotation:
    """Run Kraken segmentation for a page and persist annotation JSON."""
    image_path = resolve_page_image(data_root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise FileNotFoundError(f"Page not found: {stem}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        new_segments = segment_image(image, device=device)

    if not new_segments:
        raise RuntimeError("Kraken found no line segments on this page")

    existing = load_annotation(data_root, stem)
    if replace:
        merged_segments = new_segments
    else:
        next_number = max((s.number for s in existing.segments), default=0) + 1
        renumbered = [
            seg.model_copy(update={"number": next_number + i}) for i, seg in enumerate(new_segments)
        ]
        merged_segments = [*existing.segments, *renumbered]

    if pair_transcription:
        tx_path = data_root / "transcriptions" / "pages" / f"{stem}.txt"
        if tx_path.is_file():
            raw = tx_path.read_text(encoding="utf-8")
            text_lines = split_text_lines(raw)
            merged_segments = pair_segments_to_transcription(merged_segments, text_lines)

    annotation = PageAnnotation(segments=merged_segments, export_metadata=existing.export_metadata)
    return save_annotation(data_root, stem, annotation)
