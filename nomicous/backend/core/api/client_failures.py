"""Frontend failure beacon — logging-first observability (no Prometheus)."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])


class ClientFailureRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    ref: str | None = Field(default=None, max_length=64)
    path: str | None = Field(default=None, max_length=512)
    status: int | None = Field(default=None, ge=100, le=599)
    source: str | None = Field(default=None, max_length=64)


class ClientFailureResponse(BaseModel):
    accepted: bool = True
    ref: str


@router.post("/client-failures", response_model=ClientFailureResponse, status_code=202)
async def report_client_failure(
    body: ClientFailureRequest,
    request: Request,
) -> ClientFailureResponse:
    correlation_id = (body.ref or "").strip() or uuid.uuid4().hex
    logger.warning(
        "client_failure correlation_id=%s method=%s path=%s ui_path=%s status=%s source=%s message=%s",
        correlation_id,
        request.method,
        request.url.path,
        body.path or "-",
        body.status if body.status is not None else "-",
        body.source or "ui",
        body.message[:200],
    )
    return ClientFailureResponse(accepted=True, ref=correlation_id)
