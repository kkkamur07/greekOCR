"""Health check routes for the inference service."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference.infrastructure.settings import get_inference_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    registry: str = "ok"


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Registry unavailable"}},
)
def health() -> HealthResponse | JSONResponse:
    """Readiness: process is up and the inference registry file is readable."""
    registry_path: Path = get_inference_settings().inference_registry_path
    if not registry_path.is_file():
        body = HealthResponse(status="degraded", registry="missing")
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse(status="ok", registry="ok")
