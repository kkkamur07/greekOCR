"""Root / welcome endpoint DTOs."""

from pydantic import BaseModel


class WelcomeResponse(BaseModel):
    service: str
    tagline: str
    message: str
    docs_url: str
    health_url: str
    version: str
