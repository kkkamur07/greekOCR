"""Health endpoint DTOs."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str
