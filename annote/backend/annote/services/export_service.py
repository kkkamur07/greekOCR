"""Export service — write paired segment outputs."""

from collections.abc import Iterator
from pathlib import Path

from annote.schemas.annotation import Segment
from annote.schemas.export import ExportDoneEvent, ExportProgressEvent
from annote.schemas.warnings import ExportResponse, ExportWarnings
from annote.services.annotation_store import load_annotation
from annote.services.data_layout import export_dir
from annote.services.image_export import load_page_rgb, save_line_image
from annote.services.export_state import mark_exported
from annote.services.page_catalogue import resolve_page_image
from annote.services.processing.pipeline import apply_step
from annote.services.text_lines import split_text_lines

DEFAULT_STEPS = ["rectify"]


def resolve_export_steps(*, binarize: bool = False, steps: list[str] | None = None) -> list[str]:
    if steps is not None:
        return steps
    if binarize:
        return ["rectify", "binarize"]
    return DEFAULT_STEPS


def _paired_text(segment: Segment, text_lines: list) -> str | None:
    if segment.text_override:
        return segment.text_override
    if segment.paired_text_line_index is None:
        return None
    for line in text_lines:
        if line.index == segment.paired_text_line_index:
            return line.text
    return None


def export_page_events(
    data_root: Path,
    stem: str,
    *,
    steps: list[str] | None = None,
    binarize: bool = False,
) -> Iterator[ExportProgressEvent | ExportDoneEvent]:
    """Yield per-step progress while exporting paired segments."""
    steps = resolve_export_steps(binarize=binarize, steps=steps)
    annotation = load_annotation(data_root, stem)
    image_path = resolve_page_image(data_root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    page_image, page_pil = load_page_rgb(image_path)
    transcription_path = data_root / "transcriptions" / "pages" / f"{stem}.txt"
    raw_text = transcription_path.read_text(encoding="utf-8") if transcription_path.is_file() else ""
    text_lines = split_text_lines(raw_text)

    out_dir = export_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    used_line_indices: set[int] = set()
    exported_count = 0
    unpaired: list[int] = []
    paired: list[tuple[Segment, str]] = []

    for segment in annotation.segments:
        text = _paired_text(segment, text_lines)
        if text is None:
            unpaired.append(segment.number)
            continue
        paired.append((segment, text))

    total = len(paired)

    for index, (segment, text) in enumerate(paired, start=1):
        crop = page_image
        segment_data = segment.model_dump()
        for step in steps:
            yield ExportProgressEvent(
                current=index,
                total=total,
                segment_number=segment.number,
                step=step,  # type: ignore[arg-type]
            )
            crop = apply_step(crop, segment_data, step)

        yield ExportProgressEvent(
            current=index,
            total=total,
            segment_number=segment.number,
            step="save",
        )

        out_img = out_dir / f"{stem}_{segment.number}.jpg"
        save_line_image(crop, out_img, source=page_pil)

        out_txt = out_dir / f"{stem}_{segment.number}.txt"
        out_txt.write_text(text, encoding="utf-8")

        if segment.paired_text_line_index is not None:
            used_line_indices.add(segment.paired_text_line_index)
        exported_count += 1

    unused = [line.index for line in text_lines if line.index not in used_line_indices]

    mark_exported(data_root, stem, annotation)

    yield ExportDoneEvent(
        result=ExportResponse(
            exported_count=exported_count,
            warnings=ExportWarnings(unpaired_segments=unpaired, unused_text_lines=unused),
            steps=steps,
        )
    )


def export_page(
    data_root: Path,
    stem: str,
    *,
    steps: list[str] | None = None,
    binarize: bool = False,
) -> ExportResponse:
    """Export paired segments to line image and transcription files."""
    result: ExportResponse | None = None
    for event in export_page_events(data_root, stem, steps=steps, binarize=binarize):
        if isinstance(event, ExportDoneEvent):
            result = event.result
    if result is None:
        raise RuntimeError("Export finished without a result")
    return result
