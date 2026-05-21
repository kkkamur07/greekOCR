"""Document and DocumentPart routes under projects."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.schemas import (
    DocumentCreateRequest,
    DocumentPartResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentWithPartsResponse,
    ReorderPartsRequest,
)
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.orm_models import DocumentPart
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_service = DocumentService()


def _part_response(part: DocumentPart) -> DocumentPartResponse:
    return DocumentPartResponse(
        id=part.id,
        document_id=part.document_id,
        order=part.order,
        image_url=f"/media/parts/{part.id}",
        width=part.width,
        height=part.height,
        created_at=part.created_at,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = Query(default=False),
) -> list[DocumentResponse]:
    documents = await _service.list_documents(
        db, current_user, project_id, include_archived=include_archived
    )
    return [DocumentResponse.model_validate(d) for d in documents]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    project_id: UUID,
    body: DocumentCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    document = await _service.create_document(
        db, current_user, project_id, name=body.name
    )
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentWithPartsResponse)
async def get_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentWithPartsResponse:
    document = await _service.get_document(db, current_user, project_id, document_id)
    parts = await _service.list_parts(db, current_user, project_id, document_id)
    return DocumentWithPartsResponse(
        **DocumentResponse.model_validate(document).model_dump(),
        parts=[_part_response(p) for p in parts],
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    project_id: UUID,
    document_id: UUID,
    body: DocumentUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    updates = body.model_dump(exclude_unset=True)
    document = await _service.update_document(
        db, current_user, project_id, document_id, **updates
    )
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_document(db, current_user, project_id, document_id)


@router.post(
    "/{document_id}/parts",
    response_model=DocumentPartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_part(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentPartResponse:
    from backend.core.exceptions import ValidationError

    data = await file.read()
    if not data:
        raise ValidationError("Uploaded file is empty")
    part = await _service.upload_part(
        db,
        current_user,
        project_id,
        document_id,
        data=data,
        filename=file.filename,
    )
    return _part_response(part)


@router.patch("/{document_id}/parts/reorder", response_model=list[DocumentPartResponse])
async def reorder_parts(
    project_id: UUID,
    document_id: UUID,
    body: ReorderPartsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentPartResponse]:
    parts = await _service.reorder_parts(
        db, current_user, project_id, document_id, body.part_ids
    )
    return [_part_response(p) for p in parts]


@router.delete("/{document_id}/parts/{part_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_part(db, current_user, project_id, document_id, part_id)
