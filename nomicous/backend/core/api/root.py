"""API root - public welcome and service discovery."""

from fastapi import APIRouter

from backend.core.schemas.root import WelcomeResponse
from backend.core.settings import get_infrastructure_settings
from backend.core.version import get_version

router = APIRouter(tags=["root"])


@router.get("/", response_model=WelcomeResponse)
async def welcome() -> WelcomeResponse:
    # Production serves no interactive docs, so the root route must not advertise
    # a path that only exists in development.
    docs_url = None if get_infrastructure_settings().is_production else "/docs"
    return WelcomeResponse(
        service="Kalamos API",
        message=("Welcome to Kalamos API"),
        version=get_version(),
        docs_url=docs_url,
        health_url="/health",
    )
