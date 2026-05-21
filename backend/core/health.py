"""Health check — verifies API and database connectivity."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.infrastructure.db import get_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    await db.execute(text("SELECT 1"))
    return HealthResponse(status="ok", database="ok")
