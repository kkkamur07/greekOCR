"""Serve DocumentPart media files."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_service import DocumentService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/media", tags=["media"])
_service = DocumentService()


@router.get("/parts/{part_id}")
async def get_part_image(
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    part = await _service.get_part_for_media(db, current_user, part_id)
    data = _service.read_part_bytes(part)
    suffix = part.image_key.rsplit(".", 1)[-1].lower()
    media_type = "application/octet-stream"
    if suffix in ("png", "jpg", "jpeg", "gif", "webp"):
        media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return Response(content=data, media_type=media_type)
