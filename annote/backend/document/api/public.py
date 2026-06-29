"""Public read-only routes for published documents."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.responses import part_response
from backend.document.api.schemas import (
    DocumentResponse,
    DocumentWithPartsResponse,
    LineTranscriptionResponse,
    PublicBlockResponse,
    PublicLayoutResponse,
    PublicLineResponse,
    PublicTranscriptionLayerResponse,
)
from backend.document.application.document_service import DocumentService
from infrastructure.db import get_db

router = APIRouter(prefix="/public", tags=["public"])
_service = DocumentService()


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
    parts = sorted(document.parts, key=lambda p: p.order)
    return DocumentWithPartsResponse(
        **DocumentResponse.model_validate(document).model_dump(),
        parts=[part_response(p, public=True) for p in parts],
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/layout",
    response_model=PublicLayoutResponse,
)
async def get_published_layout(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> PublicLayoutResponse:
    blocks, lines = await _service.list_document_layout_public(db, project_id, document_id)
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
        lines=[_public_line_response(line) for line in lines],
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
