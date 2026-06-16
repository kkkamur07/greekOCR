"""API root — public welcome and service discovery."""

from fastapi import APIRouter

from backend.core.schemas.root import WelcomeResponse

router = APIRouter(tags=["root"])


@router.get("/", response_model=WelcomeResponse)
async def welcome() -> WelcomeResponse:
    return WelcomeResponse(
        service="Kalamos API",
        message=(
            "Welcome to Kalamos API"
        ),
        version="0.1.0",
        docs_url="/docs",
        health_url="/health",
    )
