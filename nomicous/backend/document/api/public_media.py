"""Public media for published document parts."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.media_responses import (
    IMAGE_MEDIA_RESPONSES,
    MAX_THUMBNAIL_WIDTH,
    PUBLIC_MEDIA_CACHE_CONTROL,
    part_image_response,
)
from backend.document.application.document_service import DocumentService
from infrastructure.db import get_db

router = APIRouter(prefix="/public/media", tags=["public"])
_service = DocumentService()


@router.get(
    "/parts/{part_id}",
    response_class=Response,
    responses=IMAGE_MEDIA_RESPONSES,
)
async def get_public_part_image(
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    w: Annotated[int | None, Query(ge=1, le=MAX_THUMBNAIL_WIDTH)] = None,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    part = await _service.get_part_for_public_media(db, part_id)
    return await part_image_response(
        _service,
        part,
        width=w,
        if_none_match=if_none_match,
        cache_control=PUBLIC_MEDIA_CACHE_CONTROL,
    )
