"""Generate a PDF with transcription text placed at segment positions."""

import io
from pathlib import Path

from PIL import Image, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from annote.services.annotation_store import load_annotation
from annote.services.fonts import resolve_unicode_font
from annote.services.page_catalogue import resolve_page_image
from annote.services.segment_geometry import segment_bbox, sort_segments_reading_order
from annote.services.segment_text import segment_text
from annote.services.text_lines import split_text_lines

_FONT_NAME = "AnnoteUnicode"
_font_registered = False

_PAGE_FILL = (1.0, 1.0, 1.0)
_TEXT_FILL = (0.1, 0.1, 0.35)


def _ensure_font() -> str:
    global _font_registered
    if not _font_registered:
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(resolve_unicode_font())))
        _font_registered = True
    return _FONT_NAME


def _fit_font_size(
    text: str,
    font_path: Path,
    max_width: float,
    max_height: float,
    *,
    min_size: int = 8,
    max_size: int = 48,
) -> int:
    if not text:
        return min_size
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return size
    return min_size


def _wrap_text_to_width(text: str, font_path: Path, font_size: int, max_width: float) -> list[str]:
    font = ImageFont.truetype(str(font_path), size=font_size)
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _page_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as page_image:
        return page_image.size


def generate_transcription_pdf(data_root: Path, stem: str) -> bytes:
    """Build a single-page PDF: blank page with paired transcription at segment positions."""
    annotation = load_annotation(data_root, stem)
    image_path = resolve_page_image(data_root / "manuscripts" / "pages", stem)
    if image_path is None:
        raise FileNotFoundError(f"Page image not found: {stem}")

    width, height = _page_size(image_path)

    transcription_path = data_root / "transcriptions" / "pages" / f"{stem}.txt"
    raw_text = transcription_path.read_text(encoding="utf-8") if transcription_path.is_file() else ""
    text_lines = split_text_lines(raw_text)

    font_path = resolve_unicode_font()
    font_name = _ensure_font()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(width, height))
    pdf.setFillColorRGB(*_PAGE_FILL)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    text_placements: list[tuple[float, float, int, str]] = []

    for segment in sort_segments_reading_order(annotation.segments):
        text = segment_text(segment, text_lines)
        if text is None:
            continue

        x0, y0, x1, y1 = segment_bbox(segment.points)
        box_w = max(x1 - x0, 1)
        box_h = max(y1 - y0, 1)
        font_size = _fit_font_size(text, font_path, box_w * 0.95, box_h * 0.85)
        lines = _wrap_text_to_width(text, font_path, font_size, box_w * 0.95)

        line_height = font_size * 1.15
        total_h = line_height * len(lines)
        start_y = y0 + max((box_h - total_h) / 2, 0)

        for i, line in enumerate(lines):
            line_y = start_y + i * line_height
            text_placements.append((x0 + 2, line_y, font_size, line))

    for x, line_y, font_size, line in text_placements:
        pdf.setFont(font_name, font_size)
        pdf.setFillColorRGB(*_TEXT_FILL)
        pdf.drawString(x, height - line_y - font_size, line)

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
