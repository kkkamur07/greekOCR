"""Public read-only routes for published documents."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.documents import _part_response
from backend.document.api.schemas import (
    DocumentResponse,
    DocumentWithPartsResponse,
    PublicLayoutResponse,
    PublicTranscriptionLayerResponse,
)
from backend.document.application.document_service import DocumentService
from infrastructure.db import get_db

router = APIRouter(prefix="/public", tags=["public"])
_service = DocumentService()


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
        parts=[_part_response(p) for p in parts],
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
    await _service.get_document_public(db, project_id, document_id)
    return PublicLayoutResponse(blocks=[], lines=[])


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
