"""Postgres-backed rate limiter for sensitive auth endpoints.

Each auth attempt is recorded as a row in ``auth_rate_limit_attempts``.
Because the store lives in the shared database, the limit is enforced
uniformly across all uvicorn worker processes — an in-process dict would
silently divide the effective limit by the number of workers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request
from sqlalchemy import delete, func, select

from backend.core.settings.app import get_app_settings
from backend.core.settings.auth import get_auth_settings
from backend.users.infrastructure.orm_models import AuthRateLimitAttempt
from infrastructure.db import AsyncSessionLocal


def clear_auth_rate_limit_state() -> None:
    """No-op — state lives in Postgres and is cleared by database truncation in tests."""


def _is_trusted_proxy_peer(host: str | None, forwarded_allow_ips: str | None) -> bool:
    if not host or not forwarded_allow_ips:
        return False
    allowed = {entry.strip() for entry in forwarded_allow_ips.split(",") if entry.strip()}
    return "*" in allowed or host in allowed


def _real_ip(request: Request) -> str:
    app_settings = get_app_settings()
    client_host = request.client.host if request.client else None
    forwarded_for = request.headers.get("X-Forwarded-For") if app_settings.behind_proxy else None
    if forwarded_for and _is_trusted_proxy_peer(client_host, app_settings.forwarded_allow_ips):
        first_hop = forwarded_for.split(",", maxsplit=1)[0].strip()
        if first_hop:
            return first_hop[:128]
    if client_host:
        return client_host[:128]
    raise HTTPException(status_code=400, detail="Unable to identify request client")


async def throttle_auth_attempts(request: Request) -> None:
    settings = get_auth_settings()
    window_seconds = settings.auth_rate_limit_window_seconds
    limit = settings.auth_rate_limit_requests
    now = datetime.now(UTC)
    window_start = now - timedelta(seconds=window_seconds)
    host = _real_ip(request)
    key = f"{host}:{request.url.path}"

    # Use a dedicated session so the rate-limit record always commits
    # independently of the surrounding auth transaction.
    async with AsyncSessionLocal() as db:
        await db.execute(
            delete(AuthRateLimitAttempt).where(
                AuthRateLimitAttempt.key == key,
                AuthRateLimitAttempt.attempted_at < window_start,
            )
        )

        count: int = await db.scalar(
            select(func.count()).select_from(AuthRateLimitAttempt).where(
                AuthRateLimitAttempt.key == key,
                AuthRateLimitAttempt.attempted_at >= window_start,
            )
        ) or 0

        if count >= limit:
            raise HTTPException(
                status_code=429,
                detail="Too many authentication attempts; try again later",
                headers={"Retry-After": str(window_seconds)},
            )

        db.add(AuthRateLimitAttempt(key=key, attempted_at=now))
        await db.commit()
