"""One transcription PDF for a whole document, pages in reading order.

:class:`DocumentTranscriptionPdfService` extends the per-page renderer rather than
wrapping it, so a page in the chapter PDF is built from exactly the values a page in the
single-page PDF is built from: the page size off the same row (with the same lazy
backfill for parts uploaded before dimensions were persisted), the same ground-truth
selection, the same font fitting.

What is *not* reused is ``generate_part_pdf`` itself, and the reason is the dependency
closure. That method returns a finished one-page PDF, and turning eighteen of those into
one document means reading PDFs back in - which needs a PDF parser. The production
requirements carry reportlab, which writes PDFs and cannot read them; ``pypdf`` is in
the test group only. Driving one reportlab canvas across every page needs no parser at
all, so the loop lives here and the drawing stays with the renderer.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.transcription_pdf_service import (
    _PAGE_FILL,
    _TEXT_FILL,
    TranscriptionPdfService,
    _ensure_font,
    _fit_font_size,
    _line_bbox,
    _PdfLine,
)
from backend.core.exceptions import ValidationError
from backend.core.fonts import resolve_unicode_font
from backend.document.application.document_export_selection import (
    DocumentExportSelector,
    document_download_name,
)
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class DocumentPdfExport:
    filename: str
    content: bytes


@dataclass(frozen=True)
class _PdfPage:
    """One page reduced to what the renderer needs, off the async session.

    Same reason :class:`_PdfLine` exists: nothing that came out of the ORM crosses the
    worker-thread boundary, because touching an expired attribute there would drive the
    async session from a thread with no greenlet context.
    """

    width: int
    height: int
    lines: list[_PdfLine]


class DocumentTranscriptionPdfService(TranscriptionPdfService):
    def __init__(
        self,
        *,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
        selector: DocumentExportSelector | None = None,
    ) -> None:
        super().__init__(documents=documents, document_service=document_service)
        self._selector = selector or DocumentExportSelector(documents=self._documents)

    async def generate_document_pdf(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        *,
        reviewed_only: bool = False,
    ) -> DocumentPdfExport:
        """Every included page of ``document_id``, drawn into one PDF in page order."""
        selection = await self._selector.select(
            session, user, project_id, document_id, reviewed_only=reviewed_only
        )
        pages: list[_PdfPage] = []
        for part in selection.parts:
            width, height = await self._page_size(session, part)
            lines = self._pdf_lines(await self._documents.list_part_lines(session, part.id))
            # A page with nothing transcribed on it still gets a page. Dropping it would
            # silently renumber everything after it, and the reader comparing this
            # against the scans needs page 7 to be page 7.
            pages.append(_PdfPage(width=width, height=height, lines=lines))
        content = await asyncio.to_thread(self._render_document_pdf, pages)
        return DocumentPdfExport(
            filename=document_download_name(selection.document.name, "transcription.pdf"),
            content=content,
        )

    def _render_document_pdf(self, pages: list[_PdfPage]) -> bytes:
        """Draw every page onto one canvas.

        reportlab lays out and compresses synchronously, which is why the caller hands
        this to a worker thread: on a single-worker event loop it would otherwise stall
        every other in-flight request for the length of a chapter.
        """
        try:
            font_path = resolve_unicode_font()
        except RuntimeError as exc:
            raise ValidationError(str(exc)) from exc
        font_name = _ensure_font(font_path)
        buffer = io.BytesIO()
        first = pages[0]
        pdf = canvas.Canvas(buffer, pagesize=(first.width, first.height))
        for page in pages:
            # Set per page, not once: a document can hold a folio spread and a single
            # leaf, and each page's line coordinates are in that page's own pixel grid.
            pdf.setPageSize((page.width, page.height))
            _draw_page(pdf, page, font_path=font_path, font_name=font_name)
            pdf.showPage()
        pdf.save()
        return buffer.getvalue()


def _draw_page(pdf: canvas.Canvas, page: _PdfPage, *, font_path: Path, font_name: str) -> None:
    """The per-page drawing the single-page renderer does, against a shared canvas.

    The one part of that renderer this module restates rather than calls, because
    ``_render_pdf`` owns the canvas it draws on and a document needs all its pages on
    one. A test holds the two to the same output - same text, same page box - so a
    reader cannot tell which endpoint drew the page in front of them. If the two ever
    need to diverge, this is the seam to move down into the renderer instead.
    """
    pdf.setFillColorRGB(*_PAGE_FILL)
    pdf.rect(0, 0, page.width, page.height, fill=1, stroke=0)
    for line in page.lines:
        x0, y0, x1, y1 = _line_bbox(line.points)
        box_w = max(x1 - x0, 1)
        box_h = max(y1 - y0, 1)
        font_size = _fit_font_size(line.text, font_path, box_w * 0.95, box_h * 0.85)
        pdf.setFont(font_name, font_size)
        pdf.setFillColorRGB(*_TEXT_FILL)
        pdf.drawString(x0 + 2, page.height - y0 - font_size, line.text)
