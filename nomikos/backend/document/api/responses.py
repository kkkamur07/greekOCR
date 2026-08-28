"""Shared response builders for document API routers."""

from backend.document.api.schemas import (
    DocumentPartResponse,
    DocumentResponse,
    DocumentWithPartsResponse,
)
from backend.document.infrastructure.orm_models import Document, DocumentPart


def document_response(
    document: Document, *, part_count: int | None = None, public: bool = False
) -> DocumentResponse:
    count = part_count if part_count is not None else len(document.parts)
    return DocumentResponse(
        id=document.id,
        project_id=document.project_id,
        name=document.name,
        workflow=document.workflow,
        part_count=count,
        # The one choke point for the share token: every caller that wants a document
        # rendered for the public surface passes ``public=True`` here, and this is the
        # only place that decides whether the secret rides along. Getting this backwards
        # would hand out a live, unrevoked link to anyone who could already read the
        # published copy - i.e. everyone.
        public_share_token=None if public else document.public_share_token,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def document_with_parts_response(
    document: Document, *, public: bool = False
) -> DocumentWithPartsResponse:
    parts = sorted(document.parts, key=lambda p: p.order)
    if public:
        # The document being published does not mean every page on it is; an
        # unpublished part must not appear in the body of the public document read.
        parts = [part for part in parts if part.published]
    return DocumentWithPartsResponse(
        **document_response(document, part_count=len(parts), public=public).model_dump(),
        parts=[part_response(part, public=public) for part in parts],
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
        published=part.published,
        created_at=part.created_at,
    )
