"""Health endpoint DTOs."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: Literal["ok", "error"]
