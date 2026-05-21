"""Minimal guarded projects route (issue 001 smoke — full CRUD in issue 002)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
async def list_projects(
    _current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict]:
    """Stub: authenticated users get an empty list until issue 002."""
    return []
