"""Public read-only routes for published documents."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.page_xml_export_service import PageXmlExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.core.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, paginate_rows
from backend.document.api.responses import document_with_parts_response
from backend.document.api.schemas import (
    DEFAULT_PUBLIC_LAYOUT_LINES,
    MAX_PUBLIC_LAYOUT_LINES,
    DocumentWithPartsResponse,
    LineTranscriptionResponse,
    PublicBlockResponse,
    PublicLayoutResponse,
    PublicLineResponse,
    PublicTranscriptionLayerResponse,
)
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.part_service import DocumentPartService
from infrastructure.db import get_db

router = APIRouter(prefix="/public", tags=["public"])
_service = DocumentCatalog()
_parts = DocumentPartService()
_transcription_pdf_service = TranscriptionPdfService()
_page_xml_export_service = PageXmlExportService()

PDF_RESPONSE = {
    200: {
        "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Transcription PDF bytes",
    }
}
XML_RESPONSE = {
    200: {
        "content": {"application/xml": {"schema": {"type": "string", "format": "binary"}}},
        "description": "PAGE XML bytes",
    }
}


def _public_line_response(line) -> PublicLineResponse:
    return PublicLineResponse(
        id=line.id,
        part_id=line.part_id,
        order=line.order,
        points=line.points,
        line_transcriptions=[
            LineTranscriptionResponse(
                id=transcription.id,
                transcription_id=transcription.transcription_id,
                transcription_kind=transcription.transcription.kind,
                text=transcription.text,
                confidence=transcription.confidence,
            )
            for transcription in line.transcriptions
        ],
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}",
    response_model=DocumentWithPartsResponse,
)
async def get_published_document(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentWithPartsResponse:
    document = await _service.get_document_public(db, project_id, document_id)
    # Parts uploaded before dimensions were persisted are filled here; without width and
    # height the published page canvas has no coordinate space to render into.
    await _parts.backfill_part_dimensions(db, document.parts)
    return document_with_parts_response(document, public=True)


@router.get(
    "/projects/{project_id}/documents/{document_id}/layout",
    response_model=PublicLayoutResponse,
)
async def get_published_layout(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_PUBLIC_LAYOUT_LINES, ge=1, le=MAX_PUBLIC_LAYOUT_LINES),
    cursor: str | None = Query(default=None, max_length=MAX_CURSOR_LENGTH),
) -> PublicLayoutResponse:
    page_cursor = decode_cursor(cursor) if cursor else None
    blocks, lines = await _service.list_document_layout_public(
        db,
        project_id,
        document_id,
        limit=limit + 1,
        cursor=page_cursor,
    )
    page, next_cursor = paginate_rows(
        lines,
        limit=limit,
        created_at_getter=lambda line: line.created_at,
        id_getter=lambda line: line.id,
    )
    return PublicLayoutResponse(
        blocks=[
            PublicBlockResponse(
                id=block.id,
                part_id=block.part_id,
                order=block.order,
                box=block.box,
            )
            for block in blocks
        ],
        lines=[_public_line_response(line) for line in page],
        next_cursor=next_cursor,
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/transcriptions",
    response_model=list[PublicTranscriptionLayerResponse],
)
async def list_published_transcriptions(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[PublicTranscriptionLayerResponse]:
    transcriptions = await _service.list_transcriptions_public(db, project_id, document_id)
    return [PublicTranscriptionLayerResponse.model_validate(t) for t in transcriptions]


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
)
async def get_published_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    pdf_bytes = await _transcription_pdf_service.generate_part_pdf_public(
        db,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="transcription.pdf"'},
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/page-xml",
    response_class=Response,
    responses=XML_RESPONSE,
)
async def get_published_page_xml(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    xml_bytes = await _page_xml_export_service.export_part_public(
        db,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=xml_bytes,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="page.xml"'},
    )
