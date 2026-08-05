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
from backend.document.application.document_service import DocumentService
from backend.users.api.rate_limit import attributable_client_ip, consume_rate_limit
from infrastructure.db import get_db

router = APIRouter(prefix="/public/media", tags=["public"])
_service = DocumentService()

#: Generous enough for a reader opening a long document (each page is one request, and
#: revalidations are not charged), tight enough that scripted enumeration of published
#: parts cannot keep the encoder saturated.
THUMBNAIL_RATE_LIMIT_REQUESTS = 240
THUMBNAIL_RATE_LIMIT_WINDOW_SECONDS = 60


async def _throttle_public_thumbnail(request: Request) -> None:
    """Cap anonymous thumbnail renders per client.

    Only the resized variant is charged: a full-size read streams bytes that already
    exist in storage, while a thumbnail can cost a decode plus a LANCZOS resize whenever
    the render cache misses. ``attributable_client_ip`` yields ``None`` behind a proxy
    tier the deployment has not allowlisted; no bucket is the right answer there, because
    the alternative is one global bucket that throttles every visitor at once.
    """
    client_ip = attributable_client_ip(request)
    if client_ip is None:
        return
    await consume_rate_limit(
        [f"public-thumbnail:{client_ip}"],
        limit=THUMBNAIL_RATE_LIMIT_REQUESTS,
        window_seconds=THUMBNAIL_RATE_LIMIT_WINDOW_SECONDS,
        detail="Too many thumbnail requests; try again later",
    )


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
) -> Response:
    part = await _service.get_part_for_public_media(db, part_id)
    return await part_image_response(
        _service,
        part,
        # Plain ``int`` past the boundary: the enum only exists to constrain the query.
        width=None if w is None else int(w),
        if_none_match=if_none_match,
        cache_control=PUBLIC_MEDIA_CACHE_CONTROL,
        before_render=(lambda: _throttle_public_thumbnail(request)) if w is not None else None,
    )
