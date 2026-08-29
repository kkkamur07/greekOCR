"""Project API request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
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
    """Who to share with: a username or an email address, exactly one of them.

    Email is the address people actually know each other by; the username is
    kept for callers that predate email sharing. Matching on email is
    case-insensitive, like login.
    """

    username: str | None = Field(default=None, min_length=1, max_length=150)
    email: EmailStr | None = None

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str | None) -> str | None:
        return value.strip().lower() if value is not None else None

    @model_validator(mode="after")
    def exactly_one_identifier(self) -> "ShareUserRequest":
        if (self.username is None) == (self.email is None):
            raise ValueError("provide exactly one of username or email")
        return self


class ProjectCollaboratorResponse(BaseModel):
    id: UUID
    username: str
    email: str

    model_config = {"from_attributes": True}


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


class ProjectPageResponse(BaseModel):
    items: list[ProjectResponse]
    next_cursor: str | None = None
