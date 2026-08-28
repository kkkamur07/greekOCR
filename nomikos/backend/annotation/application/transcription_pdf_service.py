"""Stateless Transcription PDF artifact generation."""

from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from PIL import ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError, ValidationError
from backend.core.fonts import resolve_unicode_font
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import DocumentPart, Line, TranscriptionKind
from backend.users.infrastructure.orm_models import User

_FONT_NAME = "AnnotePlatformTranscriptionPdf"
_font_registered = False
# Rendering runs on worker threads, so two requests can reach the one-time font
# registration at once; without the lock the second could draw with a font reportlab has
# not finished registering.
_font_lock = threading.Lock()
_PAGE_FILL = (1.0, 1.0, 1.0)
_TEXT_FILL = (0.1, 0.1, 0.35)


@dataclass(frozen=True)
class _PdfLine:
    """A line reduced to the values the renderer needs.

    ORM rows never cross the worker-thread boundary: touching an expired attribute there
    would drive the async session from a thread that has no greenlet context.
    """

    text: str
    points: list[list[float]]


class TranscriptionPdfService:
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService()

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

        width, height = await self._page_size(session, part)
        lines = self._pdf_lines(await self._documents.list_part_lines(session, part.id))
        # reportlab lays out and compresses the page synchronously; on a single-worker
        # event loop that would stall every other in-flight request.
        return await asyncio.to_thread(self._render_pdf, width=width, height=height, lines=lines)

    async def generate_part_pdf_public(
        self,
        session: AsyncSession,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        token: str | None = None,
    ) -> bytes:
        part = await self._document_service.get_published_part(
            session, project_id, document_id, part_id, token=token
        )
        width, height = await self._page_size(session, part)
        lines = self._pdf_lines(await self._documents.list_part_lines(session, part.id))
        return await asyncio.to_thread(self._render_pdf, width=width, height=height, lines=lines)

    async def _page_size(self, session: AsyncSession, part: DocumentPart) -> tuple[int, int]:
        """Page dimensions come from the row, not from re-decoding the stored image.

        Parts uploaded before the dimensions were persisted are backfilled lazily by the
        document service, so at most one request per legacy part pays for a decode.
        """
        if part.width is None or part.height is None:
            await self._document_service.backfill_part_dimensions(session, [part])
        if part.width is None or part.height is None:
            raise NotFoundError("Part image not found")
        return part.width, part.height

    def _pdf_lines(self, lines: list[Line]) -> list[_PdfLine]:
        prepared: list[_PdfLine] = []
        for line in lines:
            text = self._ground_truth_text(line)
            if text is None:
                continue
            prepared.append(_PdfLine(text=text, points=line.points))
        return prepared

    def _render_pdf(self, *, width: int, height: int, lines: list[_PdfLine]) -> bytes:
        try:
            font_path = resolve_unicode_font()
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from exc
        font_name = _ensure_font(font_path)
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=(width, height))
        pdf.setFillColorRGB(*_PAGE_FILL)
        pdf.rect(0, 0, width, height, fill=1, stroke=0)

        for line in lines:
            text = line.text
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
    with _font_lock:
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
