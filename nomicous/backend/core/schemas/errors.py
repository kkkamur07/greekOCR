"""Shared API error response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Any | None = Field(default=None, description="Optional structured error context")


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail
