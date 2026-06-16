"""Public media for published document parts."""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.media_responses import IMAGE_MEDIA_RESPONSES, part_image_file_response
from backend.document.application.document_service import DocumentService
from infrastructure.db import get_db

router = APIRouter(prefix="/public/media", tags=["public"])
_service = DocumentService()


@router.get(
    "/parts/{part_id}",
    response_class=FileResponse,
    responses=IMAGE_MEDIA_RESPONSES,
)
async def get_public_part_image(
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    part = await _service.get_part_for_public_media(db, part_id)
    return part_image_file_response(_service, part)
