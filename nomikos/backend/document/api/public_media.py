"""Public media for published document parts."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.media_responses import (
    IMAGE_MEDIA_RESPONSES,
    PUBLIC_MEDIA_CACHE_CONTROL,
    PublicThumbnailWidth,
    part_image_response,
)
from backend.document.api.public_rate_limit import throttle_public_thumbnail
from backend.document.api.schemas import MAX_SHARE_TOKEN_LENGTH
from backend.document.application.part_service import DocumentPartService
from infrastructure.db import get_db

router = APIRouter(prefix="/public/media", tags=["public"])
_service = DocumentPartService()


@router.get(
    "/parts/{part_id}",
    response_class=Response,
    responses={**IMAGE_MEDIA_RESPONSES, 429: {"description": "Thumbnail rate limit exceeded"}},
)
async def get_public_part_image(
    request: Request,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    w: Annotated[PublicThumbnailWidth | None, Query()] = None,
    if_none_match: Annotated[str | None, Header()] = None,
    # This route addresses a part by id alone, with no document in the path to carry
    # a token past - so ``t`` is the only thing standing between an anonymous caller
    # and any page image on the platform. ``get_part_for_public_media`` resolves the
    # part's parent document and applies the exact same published-and-token rule the
    # document routes do, rather than trusting that a part id was ever public at all.
    t: Annotated[str | None, Query(max_length=MAX_SHARE_TOKEN_LENGTH)] = None,
) -> Response:
    part = await _service.get_part_for_public_media(db, part_id, token=t)
    return await part_image_response(
        _service,
        part,
        # Plain ``int`` past the boundary: the enum only exists to constrain the query.
        width=None if w is None else int(w),
        if_none_match=if_none_match,
        cache_control=PUBLIC_MEDIA_CACHE_CONTROL,
        before_render=(lambda: throttle_public_thumbnail(request)) if w is not None else None,
    )
