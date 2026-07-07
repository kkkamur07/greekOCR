"""Root / welcome endpoint DTOs."""

from pydantic import BaseModel


class WelcomeResponse(BaseModel):
    service: str
    message: str
    version: str
    docs_url: str
    health_url: str
