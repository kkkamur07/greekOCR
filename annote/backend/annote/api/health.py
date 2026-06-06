"""Health check and root routes."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from annote.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return "Welcome to annote"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
