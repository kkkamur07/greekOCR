"""Artifact generation must not block the single-worker event loop.

PDF rendering and line-image export are CPU-bound on manuscript-sized pages: run on the
loop thread they stall every other in-flight request, including SSE and health checks.
The tests assert the work happens on a worker thread and that the PDF page size is read
from the persisted row instead of re-decoding the stored image on every request.
"""

from __future__ import annotations

import threading
import uuid
from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfReader

from backend.annotation.application.export_service import AnnotationExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.document.infrastructure.media_store import PresignUnsupported
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    LineGeometryKind,
    TranscriptionKind,
)

_PAGE_WIDTH = 160
_PAGE_HEIGHT = 90
_POINTS = [[10.0, 10.0], [120.0, 10.0], [120.0, 30.0], [10.0, 30.0]]


def _png_bytes(width: int = _PAGE_WIDTH, height: int = _PAGE_HEIGHT) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class _Transcription:
    def __init__(self, kind: TranscriptionKind) -> None:
        self.kind = kind


class _LineTranscription:
    def __init__(self, text: str, kind: TranscriptionKind) -> None:
        self.text = text
        self.transcription = _Transcription(kind)


class _StubLine:
    """Stands in for a loaded ``Line`` row without needing a live session."""

    def __init__(self, order: int, text: str | None) -> None:
        self.id = uuid.uuid4()
        self.order = order
        self.points = _POINTS
        self.kind = LineGeometryKind.polygon
        self.transcriptions = (
            [] if text is None else [_LineTranscription(text, TranscriptionKind.ground_truth)]
        )


class _Repository:
    def __init__(self, part: DocumentPart, lines: list[_StubLine]) -> None:
        self._part = part
        self._lines = lines

    async def get_part(self, _session, part_id):
        return self._part if part_id == self._part.id else None

    async def list_part_lines(self, _session, _part_id):
        return list(self._lines)

    async def list_page_transcription_lines(self, _session, _part_id):
        return []


class _DocumentServiceStub:
    def __init__(self, document: Document) -> None:
        self._document = document
        self.backfill_calls = 0

    async def get_document(self, *_args, **_kwargs):
        return self._document

    async def backfill_part_dimensions(self, _session, parts) -> None:
        self.backfill_calls += 1
        for part in parts:
            part.width, part.height = _PAGE_WIDTH, _PAGE_HEIGHT


class _Store:
    def __init__(self) -> None:
        self.blobs = {"page.webp": _png_bytes()}
        self.reads: list[str] = []

    def read(self, image_key: str) -> bytes:
        self.reads.append(image_key)
        return self.blobs[image_key]

    def write(self, image_key: str, data: bytes) -> None:
        self.blobs[image_key] = data

    def signed_object_url(self, image_key, *, expires_at):
        return f"/media/signed/{image_key}"

    def create_upload_url(self, image_key, *, expires_at):
        raise PresignUnsupported("cannot presign")


def _fixtures(*, width: int | None, height: int | None, texts: list[str | None]):
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(
        id=uuid.uuid4(),
        document_id=document.id,
        order=0,
        image_key="page.webp",
        width=width,
        height=height,
    )
    lines = [_StubLine(order, text) for order, text in enumerate(texts)]
    return document, part, _Repository(part, lines), _DocumentServiceStub(document)


# --- Transcription PDF ---


@pytest.mark.asyncio
async def test_transcription_pdf_reads_page_size_from_the_persisted_row() -> None:
    document, part, repository, document_service = _fixtures(
        width=_PAGE_WIDTH, height=_PAGE_HEIGHT, texts=["Αθήνα"]
    )
    service = TranscriptionPdfService(documents=repository, document_service=document_service)

    pdf_bytes = await service.generate_part_pdf(
        object(), object(), document.project_id, document.id, part.id
    )

    # No decode at all: dimensions came from the row, so no backfill was needed either.
    assert document_service.backfill_calls == 0
    page = PdfReader(BytesIO(pdf_bytes)).pages[0]
    assert (page.mediabox.width, page.mediabox.height) == (_PAGE_WIDTH, _PAGE_HEIGHT)


@pytest.mark.asyncio
async def test_transcription_pdf_backfills_dimensions_for_a_legacy_part() -> None:
    document, part, repository, document_service = _fixtures(
        width=None, height=None, texts=["Αθήνα"]
    )
    service = TranscriptionPdfService(documents=repository, document_service=document_service)

    pdf_bytes = await service.generate_part_pdf(
        object(), object(), document.project_id, document.id, part.id
    )

    assert document_service.backfill_calls == 1
    assert (part.width, part.height) == (_PAGE_WIDTH, _PAGE_HEIGHT)
    page = PdfReader(BytesIO(pdf_bytes)).pages[0]
    assert (page.mediabox.width, page.mediabox.height) == (_PAGE_WIDTH, _PAGE_HEIGHT)


@pytest.mark.asyncio
async def test_transcription_pdf_render_runs_off_the_event_loop(monkeypatch) -> None:
    document, part, repository, document_service = _fixtures(
        width=_PAGE_WIDTH, height=_PAGE_HEIGHT, texts=["Αθήνα"]
    )
    service = TranscriptionPdfService(documents=repository, document_service=document_service)
    loop_thread = threading.get_ident()
    render_threads: list[int] = []
    original = TranscriptionPdfService._render_pdf

    def spy(self, *args, **kwargs):
        render_threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TranscriptionPdfService, "_render_pdf", spy)

    await service.generate_part_pdf(object(), object(), document.project_id, document.id, part.id)

    assert render_threads and loop_thread not in render_threads


# --- Approved line image export ---


@pytest.mark.asyncio
async def test_export_line_images_run_off_the_event_loop(monkeypatch) -> None:
    document, part, repository, document_service = _fixtures(
        width=_PAGE_WIDTH, height=_PAGE_HEIGHT, texts=["alpha", None]
    )
    service = AnnotationExportService(
        documents=repository, document_service=document_service, media=_Store()
    )
    loop_thread = threading.get_ident()
    encode_threads: list[int] = []
    original = AnnotationExportService._processed_image_base64

    def spy(self, *args, **kwargs):
        encode_threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AnnotationExportService, "_processed_image_base64", spy)

    result = await service.export_part(
        object(), object(), document.project_id, document.id, part.id
    )

    assert result.exported_count == 1
    assert result.artifacts[0].image_base64
    assert result.warnings.unpaired_segments == [2]
    assert encode_threads and loop_thread not in encode_threads
