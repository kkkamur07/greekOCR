"""Root / welcome endpoint DTOs."""

from pydantic import BaseModel


class WelcomeResponse(BaseModel):
    service: str
    message: str
    version: str
    #: ``None`` in production, where the interactive docs are not served.
    docs_url: str | None = None
    health_url: str
