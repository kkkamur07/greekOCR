"""Wire shapes for the claim endpoint.

The claimed page carries a whole ``JobSubmitRequest`` rather than a
platform-shaped DTO of its own. That is the contract the inference runtime
already takes, so local and cloud stay literally the same program with different
credentials - which is the property ADR 0003 exists to buy.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from inference.contracts.jobs import JobSubmitRequest
from pydantic import BaseModel, Field

from backend.jobs.infrastructure.orm_models import JobType
from backend.ml.domain.execution import ExecutionTarget


class JobClaimRequest(BaseModel):
    """How long the agent is willing to wait for work.

    A laptop long-polls, because an idle researcher's page should start within a
    second of being submitted. A hosted worker sends ``0`` and short-polls: it is
    never idle for long, it does not need the latency, and every second it waits
    is a request-handler slot it is holding on a serverless host.

    There is deliberately no "how many pages" field. One claim is one page.
    """

    wait_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds to wait for a page before returning empty. Clamped server-side to "
            "DEVICE_CLAIM_MAX_WAIT_SECONDS. 0 is a short poll."
        ),
    )


class ClaimedPageResponse(BaseModel):
    """One page of work, and the short-lived link to its image.

    The link is not authenticated by anything the agent presents: the signature
    in it *is* the authorization, and it reaches exactly one object (ADR 0002).
    An authenticated ``GET /device/v1/jobs/{id}/image`` was rejected because the
    production API is serverless - streaming manuscript scans through it costs
    money for nothing - and because it would put a route on the device credential
    that must independently re-derive job ownership.
    """

    product_job_id: UUID
    inference_job_id: UUID
    job_type: JobType
    execution_target: ExecutionTarget
    lease_expires_at: datetime
    request: JobSubmitRequest
    page_image_url: str = Field(
        description=(
            "Signed link to this page's image, and to nothing else. Carries its own "
            "authorization, so it is fetched with no device credential attached."
        )
    )
    page_image_expires_at: datetime = Field(
        description=(
            "When the link above stops working - about a minute out, and deliberately "
            "not the lease. The agent fetches once, right after claiming."
        )
    )


class JobClaimResponse(BaseModel):
    """Always 200, with or without a page.

    An empty queue is the normal state of a healthy platform, not an error. A 404
    or a 204 here would make an agent's logs unreadable and would tempt a client
    into treating "nothing to do" as a failure to back off from.
    """

    page: ClaimedPageResponse | None = None
    poll_after_seconds: float = Field(
        description="What the agent should wait before asking again. 0 when a page was handed over."
    )
    lease_seconds: int = Field(
        description="How long a claimed page stays with the agent. Read from the platform, not compiled in."
    )
    server_time: datetime = Field(
        description="So an agent with a wrong clock can still reason about its lease."
    )
