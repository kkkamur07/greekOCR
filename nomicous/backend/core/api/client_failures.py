"""Frontend failure beacon — logging-first observability (no Prometheus)."""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from backend.users.api.rate_limit import client_ip_for_request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RATE_LIMIT = 30
_RATE_WINDOW_SECONDS = 60.0
_rate_lock = threading.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def clear_client_failure_rate_limit_state() -> None:
    with _rate_lock:
        _rate_buckets.clear()


def _sanitize_log_field(value: str, *, max_len: int) -> str:
    cleaned = _CONTROL_CHARS.sub(" ", value).replace("\r", " ").replace("\n", " ")
    return cleaned.strip()[:max_len]


def _throttle_client_failure(request: Request) -> None:
    """Rate-limit before body validation (route dependency)."""
    host = client_ip_for_request(request)
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_SECONDS
    with _rate_lock:
        # Drop idle IPs so the map cannot grow without bound in long-lived workers.
        stale = [ip for ip, stamps in _rate_buckets.items() if not stamps or stamps[-1] < cutoff]
        for ip in stale:
            del _rate_buckets[ip]
        bucket = _rate_buckets[host]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Too many client failure reports; try again later",
                headers={"Retry-After": str(int(_RATE_WINDOW_SECONDS))},
            )
        bucket.append(now)


class ClientFailureRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    ref: str | None = Field(default=None, max_length=64)
    path: str | None = Field(default=None, max_length=512)
    status: int | None = Field(default=None, ge=100, le=599)
    source: str | None = Field(default=None, max_length=64)

    @field_validator("message", "ref", "path", "source", mode="before")
    @classmethod
    def _reject_control_characters(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if _CONTROL_CHARS.search(value) or "\r" in value or "\n" in value:
            raise ValueError("control characters are not allowed")
        return value


class ClientFailureResponse(BaseModel):
    accepted: bool = True
    ref: str


@router.post(
    "/client-failures",
    response_model=ClientFailureResponse,
    status_code=202,
    dependencies=[Depends(_throttle_client_failure)],
)
async def report_client_failure(
    body: ClientFailureRequest,
    request: Request,
) -> ClientFailureResponse:
    correlation_id = _sanitize_log_field((body.ref or "").strip(), max_len=64) or uuid.uuid4().hex
    logger.warning(
        "client_failure correlation_id=%s method=%s path=%s ui_path=%s status=%s source=%s message=%s",
        correlation_id,
        request.method,
        request.url.path,
        _sanitize_log_field(body.path or "-", max_len=512),
        body.status if body.status is not None else "-",
        _sanitize_log_field(body.source or "ui", max_len=64),
        _sanitize_log_field(body.message, max_len=200),
    )
    return ClientFailureResponse(accepted=True, ref=correlation_id)
