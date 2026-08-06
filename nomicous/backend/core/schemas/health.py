"""Health endpoint DTOs."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database: Literal["ok", "error"]
    oldest_pending_job_seconds: float | None = Field(
        default=None,
        description=(
            "Age of the oldest job still in `pending`, in seconds. `null` means "
            "the queue is empty, or - on a 503 - that it could not be read. "
            "Reported, never gated on: see the route docstring."
        ),
    )
