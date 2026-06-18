"""Shared response builders for document API routers."""

from backend.document.api.schemas import DocumentPartResponse
from backend.document.infrastructure.orm_models import DocumentPart


def part_response(part: DocumentPart, *, public: bool = False) -> DocumentPartResponse:
    media_prefix = "/public/media" if public else "/media"
    return DocumentPartResponse(
        id=part.id,
        document_id=part.document_id,
        order=part.order,
        image_url=f"{media_prefix}/parts/{part.id}",
        width=part.width,
        height=part.height,
        reviewed=part.reviewed,
        created_at=part.created_at,
    )
