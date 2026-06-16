"""Serve DocumentPart media files."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.media_responses import IMAGE_MEDIA_RESPONSES, part_image_file_response
from backend.document.application.document_service import DocumentService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/media", tags=["media"])
_service = DocumentService()


@router.get(
    "/parts/{part_id}",
    response_class=FileResponse,
    responses=IMAGE_MEDIA_RESPONSES,
)
async def get_part_image(
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileResponse:
    part = await _service.get_part_for_media(db, current_user, part_id)
    return part_image_file_response(_service, part)
