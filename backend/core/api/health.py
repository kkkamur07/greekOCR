"""Health check routes."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.schemas.health import HealthResponse
from infrastructure.db import get_db

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    await db.execute(text("SELECT 1"))
    return HealthResponse(status="ok", database="ok")
