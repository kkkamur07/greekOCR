"""Serve DocumentPart media files."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.media_responses import (
    IMAGE_MEDIA_RESPONSES,
    MAX_THUMBNAIL_WIDTH,
    PRIVATE_MEDIA_CACHE_CONTROL,
    part_image_response,
)
from backend.document.application.document_service import DocumentService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/media", tags=["media"])
_service = DocumentService()


@router.get(
    "/parts/{part_id}",
    response_class=Response,
    responses=IMAGE_MEDIA_RESPONSES,
)
async def get_part_image(
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    w: Annotated[int | None, Query(ge=1, le=MAX_THUMBNAIL_WIDTH)] = None,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    part = await _service.get_part_for_media(db, current_user, part_id)
    return await part_image_response(
        _service,
        part,
        width=w,
        if_none_match=if_none_match,
        cache_control=PRIVATE_MEDIA_CACHE_CONTROL,
    )
