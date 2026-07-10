"""Stateless Transcription PDF artifact generation."""

from __future__ import annotations

import io
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.core.fonts import resolve_unicode_font
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store import MediaStore, get_media_store
from backend.document.infrastructure.orm_models import Line, TranscriptionKind
from backend.users.infrastructure.orm_models import User

_FONT_NAME = "AnnotePlatformTranscriptionPdf"
_font_registered = False
_PAGE_FILL = (1.0, 1.0, 1.0)
_TEXT_FILL = (0.1, 0.1, 0.35)


class TranscriptionPdfService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
        media: MediaStore | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()
        self._media = media or get_media_store()

    async def generate_part_pdf(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")

        with Image.open(io.BytesIO(self._media.read(part.image_key))) as page_image:
            width, height = page_image.size

        return self._render_pdf(
            width=width,
            height=height,
            lines=await self._documents.list_part_lines(session, part.id),
        )

    async def generate_part_pdf_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> bytes:
        part = await self._document_service.get_published_part(
            session, project_id, document_id, part_id
        )
        with Image.open(io.BytesIO(self._media.read(part.image_key))) as page_image:
            width, height = page_image.size

        return self._render_pdf(
            width=width,
            height=height,
            lines=await self._documents.list_part_lines(session, part.id),
        )

    def _render_pdf(self, *, width: int, height: int, lines: list[Line]) -> bytes:
        font_path = resolve_unicode_font()
        font_name = _ensure_font(font_path)
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=(width, height))
        pdf.setFillColorRGB(*_PAGE_FILL)
        pdf.rect(0, 0, width, height, fill=1, stroke=0)

        for line in lines:
            text = self._ground_truth_text(line)
            if text is None:
                continue
            x0, y0, x1, y1 = _line_bbox(line.points)
            box_w = max(x1 - x0, 1)
            box_h = max(y1 - y0, 1)
            font_size = _fit_font_size(text, font_path, box_w * 0.95, box_h * 0.85)
            pdf.setFont(font_name, font_size)
            pdf.setFillColorRGB(*_TEXT_FILL)
            pdf.drawString(x0 + 2, height - y0 - font_size, text)

        pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    def _ground_truth_text(self, line: Line) -> str | None:
        for transcription in line.transcriptions:
            if (
                transcription.transcription.kind == TranscriptionKind.ground_truth
                and transcription.text.strip()
            ):
                return transcription.text
        return None


def _ensure_font(font_path: Path) -> str:
    global _font_registered
    if not _font_registered:
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(font_path)))
        _font_registered = True
    return _FONT_NAME


def _line_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _fit_font_size(
    text: str,
    font_path: Path,
    max_width: float,
    max_height: float,
    *,
    min_size: int = 8,
    max_size: int = 48,
) -> int:
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = font.getbbox(text)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return size
    return min_size
