"""API root — public welcome and service discovery."""

from fastapi import APIRouter

from backend.core.schemas.root import WelcomeResponse
from backend.core.version import get_version

router = APIRouter(tags=["root"])


@router.get("/", response_model=WelcomeResponse)
async def welcome() -> WelcomeResponse:
    return WelcomeResponse(
        service="Kalamos API",
        message=("Welcome to Kalamos API"),
        version=get_version(),
        docs_url="/docs",
        health_url="/health",
    )
