"""API root — public welcome and service discovery."""

from fastapi import APIRouter

from backend.core.schemas.root import WelcomeResponse

router = APIRouter(tags=["root"])


@router.get("/", response_model=WelcomeResponse)
async def welcome() -> WelcomeResponse:
    return WelcomeResponse(
        service="Kalamos API",
        tagline="Where manuscripts meet the machine",
        message=(
            "Welcome to Kalamos — the greekOCR platform for scholars, curators, and builders. "
            "Upload facsimiles, segment pages into blocks and lines, run Kraken and friends for "
            "transcription, then refine ground truth by hand. Layout you draw is law; models "
            "propose, humans decide."
        ),
        docs_url="/docs",
        health_url="/health",
        version="0.1.0",
    )
