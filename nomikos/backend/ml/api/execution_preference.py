"""The account-level **execution target** setting, and what it currently means.

One setting, "use my computer when it is available", and a read of **capacity**
so a client can say whether the preference is being honoured right now. There is
no per-job control here and there must not be one: see ADR 0002.

This issue is backend only - issue 059 builds the interface over these two
routes. What it needs is exactly what they return: the stored preference, the
host that preference resolves to, and which hosts can take work.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ml.application.capacity_service import InferenceCapacityService
from backend.ml.domain.execution import ExecutionTarget
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/account", tags=["ml"])
_capacity = InferenceCapacityService()


class ExecutionPreferenceRequest(BaseModel):
    prefer_local_inference: bool


class ExecutionPreferenceResponse(BaseModel):
    prefer_local_inference: bool
    preferred_execution_target: ExecutionTarget
    available_targets: list[ExecutionTarget]


async def _current(session: AsyncSession, user: User) -> ExecutionPreferenceResponse:
    request = await _capacity.execution_request(session, user)
    return ExecutionPreferenceResponse(
        prefer_local_inference=bool(user.prefer_local_inference),
        preferred_execution_target=request.preferred,
        # Sorted so the response is stable; a set has no order and a client that
        # renders one would flicker between equivalent bodies.
        available_targets=sorted(request.available),
    )


@router.get("/execution-target", response_model=ExecutionPreferenceResponse)
async def read_execution_preference(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ExecutionPreferenceResponse:
    return await _current(db, current_user)


@router.put("/execution-target", response_model=ExecutionPreferenceResponse)
async def set_execution_preference(
    body: ExecutionPreferenceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ExecutionPreferenceResponse:
    """Change the preference. Jobs already submitted keep the host they were given.

    The setting selects the *next* job's preferred host and nothing else - an
    **execution target** is fixed at submission, so changing this can never move
    work that is already queued.
    """
    current_user.prefer_local_inference = body.prefer_local_inference
    await db.commit()
    await db.refresh(current_user)
    return await _current(db, current_user)
