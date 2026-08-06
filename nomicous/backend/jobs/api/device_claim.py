"""The claim endpoint: one page of work, to one authenticated inference agent.

**This route must never take ``Depends(get_db)``**, and that is the single most
load-bearing line in the module. ``get_db`` pins a pooled connection for the
whole request; a 25 second long poll per idle agent would therefore exhaust the
async pool at ``DB_POOL_SIZE + DB_MAX_OVERFLOW`` - fifteen on the defaults. ADR
0003 moves *all* inference onto this path, so that ceiling binds sooner than it
would have for laptops alone. ``stream_job_events`` already avoids ``get_db`` for
exactly this reason.

So the shape here is: acquire a session, do a unit of work, release it, wait
holding nothing, repeat. Authentication opens and closes its own short-lived
session before the loop starts, and each claim attempt opens and closes a sync
session inside a worker thread. Between attempts the request holds no connection
at all - it is asleep.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from backend.core.exceptions import InvalidCredentialsError
from backend.core.settings.device import get_device_settings
from backend.jobs.api.claim_schemas import (
    ClaimedPageRequest,
    ClaimedPageResponse,
    JobClaimRequest,
    JobClaimResponse,
)
from backend.jobs.application.job_claim_service import ClaimedPage, claim_one_page
from backend.jobs.infrastructure.stale_sweep import sweep_stale_jobs_on_read
from backend.ml.api.agent_version import (
    AGENT_VERSION_REFUSED_STATUS,
    AgentVersionRefusalResponse,
    SupportedAgentVersion,
)
from backend.ml.api.device_dependencies import require_device_pairing_enabled
from backend.ml.application.agent_credentials import (
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
    InferenceAgent,
    resolve_inference_agent,
)
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from backend.users.api.rate_limit import client_ip_for_request
from infrastructure.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["device-jobs"], dependencies=[Depends(require_device_pairing_enabled)])


async def _authenticate(
    *,
    device_token: str | None,
    service_token: str | None,
    worker_name: str | None,
    request_ip: str | None,
) -> InferenceAgent:
    """Resolve the caller on a session that is closed before the wait begins."""
    try:
        async with AsyncSessionLocal() as session:
            return await resolve_inference_agent(
                session,
                device_token=device_token,
                service_token=service_token,
                worker_name=worker_name,
                request_ip=request_ip,
            )
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid agent credential",
            headers={"WWW-Authenticate": DEVICE_TOKEN_HEADER},
        ) from None


def _page_response(page: ClaimedPage) -> ClaimedPageResponse:
    return ClaimedPageResponse(
        product_job_id=page.product_job_id,
        inference_job_id=page.inference_job_id,
        job_type=page.job_type,
        execution_target=page.execution_target,
        lease_expires_at=page.lease_expires_at,
        request=ClaimedPageRequest(
            task=page.request.task,
            registry_model_id=page.request.registry_model_id,
            registry_tag=page.request.registry_tag,
            product_job_id=page.request.product_job_id,
            params=page.request.params,
        ),
        page_image_url=page.page_image_url,
        page_image_expires_at=page.page_image_expires_at,
    )


@router.post(
    "/device/v1/jobs/claim",
    response_model=JobClaimResponse,
    responses={
        AGENT_VERSION_REFUSED_STATUS: {
            "model": AgentVersionRefusalResponse,
            "description": (
                "The agent is below the version floor, or did not say what version it "
                "is. It must upgrade; retrying the same build cannot succeed."
            ),
        }
    },
)
async def claim_job(
    request: Request,
    agent_version: SupportedAgentVersion,
    body: JobClaimRequest | None = None,
    x_nomicous_device_token: Annotated[str | None, Header(alias=DEVICE_TOKEN_HEADER)] = None,
    x_nomicous_service_token: Annotated[str | None, Header(alias=SERVICE_TOKEN_HEADER)] = None,
    x_nomicous_worker_name: Annotated[str | None, Header(alias=WORKER_NAME_HEADER)] = None,
) -> JobClaimResponse:
    """Claim at most one page of work.

    The credential decides the **execution target**, and the caller cannot ask for
    a different one: a device token claims ``local`` work on its own account, a
    service credential claims ``cloud`` work for the platform.

    ``agent_version`` is resolved before this body runs, so an agent below the
    floor is refused without a session being opened and without its
    ``last_seen_at`` being touched - it stops reporting **capacity**, and
    submission announces "no host available" rather than creating pages it may
    not claim.
    """
    settings = get_device_settings()
    agent = await _authenticate(
        device_token=x_nomicous_device_token,
        service_token=x_nomicous_service_token,
        worker_name=x_nomicous_worker_name,
        request_ip=client_ip_for_request(request),
    )

    # An agent that stopped without reporting still holds its page. Sweeping here
    # is what makes that page claimable again on a host with no background loop -
    # the production API is serverless and has none. Issue 054 owns the lease
    # itself; this is only the existing opportunistic sweep, called from one more
    # read path.
    await sweep_stale_jobs_on_read()

    wait_seconds = min(
        (body.wait_seconds if body is not None else 0),
        settings.device_claim_max_wait_seconds,
    )
    deadline = time.monotonic() + wait_seconds
    interval = settings.device_claim_poll_interval_seconds

    while True:
        # Checked before the claim, not only after it. A long-poller that hung up
        # mid-wait would otherwise take a page on its way out the door, and
        # nothing would notice until the lease expired minutes later - the agent
        # is gone, so it will never report, and the page is unclaimable until the
        # sweep releases it. There is no cost to asking: an agent that is still
        # there answers False immediately.
        if await request.is_disconnected():
            break
        page = await asyncio.to_thread(
            claim_one_page,
            agent,
            lease_seconds=settings.device_lease_seconds,
            # Two lifetimes, not one. The link only has to cover a single
            # download immediately after claiming; the lease covers the whole
            # run. See ADR 0002.
            page_image_ttl_seconds=settings.device_page_image_url_ttl_seconds,
            base_url=str(request.base_url),
        )
        if page is not None:
            return JobClaimResponse(
                page=_page_response(page),
                poll_after_seconds=0.0,
                lease_seconds=settings.device_lease_seconds,
                server_time=datetime.now(UTC),
                agent=agent_version,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0 or await request.is_disconnected():
            break
        # Nothing is held across this. No session, no connection, no row lock.
        await asyncio.sleep(min(interval, remaining))

    return JobClaimResponse(
        page=None,
        poll_after_seconds=settings.device_claim_idle_poll_seconds,
        lease_seconds=settings.device_lease_seconds,
        server_time=datetime.now(UTC),
        agent=agent_version,
    )
