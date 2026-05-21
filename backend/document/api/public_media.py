"""Public media for published document parts."""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.document_service import DocumentService
from infrastructure.db import get_db

router = APIRouter(prefix="/public/media", tags=["public"])
_service = DocumentService()


@router.get("/parts/{part_id}")
async def get_public_part_image(
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    part = await _service.get_part_for_public_media(db, part_id)
    data = _service.read_part_bytes(part)
    suffix = part.image_key.rsplit(".", 1)[-1].lower()
    media_type = "application/octet-stream"
    if suffix in ("png", "jpg", "jpeg", "gif", "webp"):
        media_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return Response(content=data, media_type=media_type)
