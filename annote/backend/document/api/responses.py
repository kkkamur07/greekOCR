"""Shared response builders for document API routers."""

from backend.document.api.schemas import DocumentPartResponse, DocumentResponse, DocumentWithPartsResponse
from backend.document.infrastructure.orm_models import Document, DocumentPart


def document_response(document: Document, *, part_count: int | None = None) -> DocumentResponse:
    count = part_count if part_count is not None else len(document.parts)
    return DocumentResponse(
        id=document.id,
        project_id=document.project_id,
        name=document.name,
        workflow=document.workflow,
        part_count=count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def document_with_parts_response(
    document: Document, *, public: bool = False
) -> DocumentWithPartsResponse:
    return DocumentWithPartsResponse(
        **document_response(document).model_dump(),
        parts=[
            part_response(part, public=public)
            for part in sorted(document.parts, key=lambda p: p.order)
        ],
    )


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
