"""Shared, intentionally minimal public API error response schemas."""

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Allowlisted user-safe error message")


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail
