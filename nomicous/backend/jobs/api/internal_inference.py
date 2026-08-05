"""Internal inference routes: job completion callbacks from an inference agent.

Completion and failure are **not** new endpoints. Since ADR 0003 there is one
inference agent, local or hosted, and it reports the way the inference service
always did: one ``JobCallbackRequest`` to this route, validated by the same
contract, applied by the same ``JobCallbackService``, announced on the same SSE
channel. The claim endpoint is the only thing that layer adds.

What did have to change is *who may present a callback*. The webhook secret is a
platform-side credential; a researcher's laptop does not have one and must not be
given one, since a shared secret handed to every agent would let any of them
complete any job on the platform. So this route also accepts the two agent
credentials, and narrows them to the page that agent is actually holding:
``jobs.claimed_by`` names the claiming device, and nothing else is accepted. A
device that merely *could* have claimed a page cannot report on it.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from inference.contracts.jobs import JobCallbackRequest
from inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import InvalidCredentialsError
from backend.jobs.api.dependencies import require_inference_webhook_secret
from backend.jobs.application.job_callback_service import JobCallbackService
from backend.jobs.application.job_claim_service import job_is_held_by
from backend.jobs.infrastructure.orm_models import Job
from backend.ml.application.agent_credentials import (
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
    resolve_inference_agent,
)
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import get_db

router = APIRouter(prefix="/internal/inference", tags=["internal-inference"])


def _callback_service(db: AsyncSession = Depends(get_db)) -> JobCallbackService:
    return JobCallbackService(db)


async def _authorize_agent_callback(
    db: AsyncSession,
    body: JobCallbackRequest,
    *,
    device_token: str | None,
    service_token: str | None,
    worker_name: str | None,
) -> None:
    try:
        agent = await resolve_inference_agent(
            db,
            device_token=device_token,
            service_token=service_token,
            worker_name=worker_name,
        )
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent credential",
            headers={"WWW-Authenticate": DEVICE_TOKEN_HEADER},
        ) from None

    job = (await db.execute(select(Job).where(Job.id == body.product_job_id))).scalar_one_or_none()
    # An unknown job is 403 here rather than 404: an agent that does not hold a
    # page must not be able to probe which job ids exist.
    if job is None or not job_is_held_by(job, agent):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This agent is not holding that job",
        )


@router.post(
    "/job-complete",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def complete_inference_job(
    body: JobCallbackRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    service: Annotated[JobCallbackService, Depends(_callback_service)],
    x_inference_webhook_secret: Annotated[
        str | None, Header(alias=INFERENCE_WEBHOOK_SECRET_HEADER)
    ] = None,
    x_nomicous_device_token: Annotated[str | None, Header(alias=DEVICE_TOKEN_HEADER)] = None,
    x_nomicous_service_token: Annotated[str | None, Header(alias=SERVICE_TOKEN_HEADER)] = None,
    x_nomicous_worker_name: Annotated[str | None, Header(alias=WORKER_NAME_HEADER)] = None,
) -> Response:
    if x_nomicous_device_token is not None or x_nomicous_service_token is not None:
        await _authorize_agent_callback(
            db,
            body,
            device_token=x_nomicous_device_token,
            service_token=x_nomicous_service_token,
            worker_name=x_nomicous_worker_name,
        )
    else:
        # Unchanged: no agent credential presented means this is the platform-side
        # webhook path, with the same 503 / 401 / 403 outcomes it always had.
        require_inference_webhook_secret(x_inference_webhook_secret)
    await service.apply_callback(body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
