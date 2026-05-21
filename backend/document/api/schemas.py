"""Pydantic DTOs for documents and parts."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from backend.document.infrastructure.orm_models import DocumentWorkflow


class DocumentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)


class DocumentUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=512)
    workflow: DocumentWorkflow | None = None

    @field_validator("name", "workflow", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class DocumentResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    workflow: DocumentWorkflow
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentPartResponse(BaseModel):
    id: UUID
    document_id: UUID
    order: int
    image_url: str
    width: int | None
    height: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentWithPartsResponse(DocumentResponse):
    parts: list[DocumentPartResponse]


class ReorderPartsRequest(BaseModel):
    part_ids: list[UUID] = Field(min_length=1)


class PublicLayoutResponse(BaseModel):
    blocks: list[dict] = Field(default_factory=list)
    lines: list[dict] = Field(default_factory=list)


class PublicTranscriptionLayerResponse(BaseModel):
    id: UUID
    name: str
    kind: str

    model_config = {"from_attributes": True}
