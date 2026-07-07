"""Project API request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    slug: str = Field(min_length=1, max_length=512)
    guidelines: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    slug: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    guidelines: str | None = None

    @field_validator("name", "slug", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class ShareUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    guidelines: str | None
    owner_id: UUID | None
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
