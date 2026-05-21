"""Project API request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    slug: str = Field(min_length=1, max_length=512)
    guidelines: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=512)
    slug: str | None = Field(default=None, min_length=1, max_length=512)
    guidelines: str | None = None


class ShareUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    guidelines: str | None
    owner_id: UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
