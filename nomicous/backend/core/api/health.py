"""Health check routes."""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.schemas.health import HealthResponse
from infrastructure.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse | JSONResponse:
    """Liveness/readiness: SELECT 1 only. Dev-user seeding lives in app lifespan."""
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        body = HealthResponse(status="degraded", database="error")
        return JSONResponse(status_code=503, content=body.model_dump())

    return HealthResponse(status="ok", database="ok")
