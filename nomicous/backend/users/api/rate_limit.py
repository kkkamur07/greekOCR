"""Postgres-backed rate limiter for sensitive and unauthenticated endpoints.

Each attempt is recorded as a row in ``auth_rate_limit_attempts``. Because the
store lives in the shared database, the limit is enforced uniformly across all
uvicorn worker processes and across serverless invocations - an in-process dict
would silently divide the effective limit by the number of workers, or reset on
every cold start.

Choosing the key is the harder half of the problem. ``request.client.host`` is
the direct TCP peer, which on a managed platform is the provider's proxy rather
than the browser. Keying a "per-client" limit on that value produces one global
bucket: it throttles the whole product instead of the attacker. So the peer is
only used when the deployment declares it meaningful (see
``AppSettings.trust_peer_ip``), and sensitive auth routes carry a second,
independent bucket keyed on the account being targeted - which is correct no
matter what the network path looks like.

That second bucket only holds if the account is read out of every request that
could reach a password check, so nothing about how the client *describes* its
body is allowed to decide whether the body gets read. See
``_request_account_identity``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address, ip_network

from fastapi import HTTPException, Request
from sqlalchemy import delete, func, select, text

from backend.core.settings.app import get_app_settings
from backend.core.settings.auth import get_auth_settings
from backend.users.infrastructure.orm_models import AuthRateLimitAttempt
from infrastructure.db import AsyncSessionLocal

logger = logging.getLogger(__name__)

#: Auth payloads under this dependency are a handful of short fields; nothing
#: legitimate comes near this. A larger body is *refused*, not skipped: pydantic
#: ignores unknown keys, so `{"email": ..., "password": ..., "pad": "x" * 9000}`
#: still authenticates. If "too big to read" meant "too big to attribute", the
#: account bucket would be one padding field away from being switched off.
MAX_IDENTITY_BODY_BYTES = 8 * 1024

#: Ceiling for requests that name no account and come from no attributable
#: address. Deliberately far above the per-account limit - see
#: ``throttle_auth_attempts`` for why a shared bucket is safe *here* and would be
#: an outage on the main path.
UNATTRIBUTABLE_AUTH_RATE_LIMIT = 300


def clear_auth_rate_limit_state() -> None:
    """No-op - state lives in Postgres and is cleared by database truncation in tests."""


def client_ip_for_request(request: Request) -> str:
    """Public helper for IP-based throttles outside auth routes."""
    return _real_ip(request)


def _is_trusted_proxy_peer(host: str | None, forwarded_allow_ips: str | None) -> bool:
    if not host or not forwarded_allow_ips:
        return False
    try:
        peer = ip_address(host)
    except ValueError:
        return False

    for entry in forwarded_allow_ips.split(","):
        try:
            if peer in ip_network(entry.strip(), strict=False):
                return True
        except ValueError:
            continue
    return False


def _forwarded_client_ip(forwarded_for: str) -> str | None:
    """Return the canonical leftmost X-Forwarded-For address, if valid."""
    first_hop = forwarded_for.split(",", maxsplit=1)[0].strip()
    try:
        return str(ip_address(first_hop))
    except ValueError:
        return None


def _trusted_forwarded_ip(request: Request) -> str | None:
    app_settings = get_app_settings()
    if not app_settings.behind_proxy:
        return None
    forwarded_for = request.headers.get("X-Forwarded-For")
    if not forwarded_for:
        return None
    client_host = request.client.host if request.client else None
    if not _is_trusted_proxy_peer(client_host, app_settings.forwarded_allow_ips):
        return None
    return _forwarded_client_ip(forwarded_for)


def _real_ip(request: Request) -> str:
    client_ip = _trusted_forwarded_ip(request)
    if client_ip:
        return client_ip
    client_host = request.client.host if request.client else None
    if client_host:
        return client_host[:128]
    raise HTTPException(status_code=400, detail="Unable to identify request client")


def attributable_client_ip(request: Request) -> str | None:
    """Return an address that identifies one client, or ``None`` if none does.

    ``None`` is the honest answer when the process sits behind a proxy tier it
    cannot allowlist: the peer address is then shared by every visitor, and a
    bucket keyed on it says nothing about who sent the request.
    """
    client_ip = _trusted_forwarded_ip(request)
    if client_ip:
        return client_ip
    if not get_app_settings().trust_peer_ip:
        return None
    client_host = request.client.host if request.client else None
    return client_host[:128] if client_host else None


def _account_identity(payload: object) -> str | None:
    """Hash the account an auth request targets, for identity-keyed throttling."""
    if not isinstance(payload, dict):
        return None
    email = payload.get("email")
    if not isinstance(email, str):
        return None
    normalized = email.strip().casefold()
    if not normalized:
        return None
    # Hashed so the limiter table never becomes a list of registered addresses.
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def _request_account_identity(request: Request) -> str | None:
    """Hash the account a request targets, whatever it claims to be sending.

    Deliberately *not* gated on ``Content-Type``. Media types are case-insensitive
    (RFC 9110 s8.3.1) and FastAPI decodes a body as JSON for any
    ``application/*+json`` subtype and for a request that declares no content type
    at all - so a gate that string-compares the header against
    ``"application/json"`` is one an attacker steps over by capitalising a letter,
    while the route behind it parses the body and checks the password regardless.
    Erasing the account key is the whole attack, because on a deployment with
    ``TRUST_PEER_IP=false`` there is no IP key to fall back on. The bytes decide,
    not the header.

    Raises 413 for a body too large to attribute; see ``MAX_IDENTITY_BODY_BYTES``.
    """
    try:
        raw = await request.body()
    except Exception:  # pragma: no cover - client disconnect during read
        return None
    if not raw:
        return None
    if len(raw) > MAX_IDENTITY_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    # Starlette caches the body on the request, so the route still parses it.
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        # Not JSON, so it cannot reach a password check either: every route under
        # this dependency declares a pydantic body model, and FastAPI only fills
        # those from JSON - none of them accept form-encoded or multipart input.
        # There is nothing further to probe. RecursionError joins ValueError
        # because the stdlib scanner raises it, not a JSONDecodeError, on deeply
        # nested input, and a 500 here would be its own denial of service.
        return None
    return _account_identity(payload)


async def consume_rate_limit(
    keys: Sequence[str],
    *,
    limit: int,
    window_seconds: int,
    detail: str,
) -> None:
    """Charge one attempt against every key, or raise 429 if any is exhausted.

    All keys are checked before any is charged, so a request rejected by one
    bucket does not consume budget in the others.
    """
    unique_keys = sorted({key[:255] for key in keys})
    if not unique_keys:
        return
    now = datetime.now(UTC)
    window_start = now - timedelta(seconds=window_seconds)

    # Use a dedicated session so the rate-limit record always commits
    # independently of the surrounding request transaction. Locks are taken in
    # sorted order so two requests sharing two keys cannot deadlock.
    async with AsyncSessionLocal() as db:
        for key in unique_keys:
            await db.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": key},
            )
        for key in unique_keys:
            await db.execute(
                delete(AuthRateLimitAttempt).where(
                    AuthRateLimitAttempt.key == key,
                    AuthRateLimitAttempt.attempted_at < window_start,
                )
            )
            count: int = (
                await db.scalar(
                    select(func.count())
                    .select_from(AuthRateLimitAttempt)
                    .where(
                        AuthRateLimitAttempt.key == key,
                        AuthRateLimitAttempt.attempted_at >= window_start,
                    )
                )
                or 0
            )
            if count >= limit:
                await db.rollback()
                raise HTTPException(
                    status_code=429,
                    detail=detail,
                    headers={"Retry-After": str(window_seconds)},
                )
        for key in unique_keys:
            db.add(AuthRateLimitAttempt(key=key, attempted_at=now))
        await db.commit()


async def auth_rate_limit_keys(request: Request) -> list[str]:
    """Buckets an auth attempt is charged against.

    Two independent dimensions, either of which may be absent:

    * ``ip:`` - only when the address identifies one client. Skipped rather than
      collapsed into a global bucket, because a global login limit is a
      self-inflicted outage, not a defence.
    * ``account:`` - the targeted account, hashed. Survives any proxy topology
      and is what actually caps online password guessing against a user.

    An empty list means neither applies; ``throttle_auth_attempts`` decides what
    to charge such a request against rather than letting it through free.
    """
    path = request.url.path
    keys: list[str] = []
    client_ip = attributable_client_ip(request)
    if client_ip:
        keys.append(f"ip:{client_ip}:{path}")
    identity = await _request_account_identity(request)
    if identity:
        keys.append(f"account:{identity}:{path}")
    return keys


async def throttle_auth_attempts(request: Request) -> None:
    settings = get_auth_settings()
    keys = await auth_rate_limit_keys(request)
    if not keys:
        # Fail closed, but coarsely. Now that identity extraction ignores the
        # declared content type, the only requests reaching this branch are ones
        # that name no account at all - and a body without an `email` is rejected
        # by the route before any password is checked. That is precisely what
        # makes one shared bucket safe here and an outage on the main path: no
        # real sign-in ever enters this bucket, so exhausting it locks nobody
        # out of signing in.
        # So this is not a per-client limit and is not pretending to be one; it
        # bounds what an unattributable caller can make the database do for free,
        # which is why it sits far above the per-account budget and is scoped per
        # path so one route cannot starve another.
        #
        # Returning without charging anything, as this branch used to, meant a
        # single capitalised header bought unmetered password guessing.
        logger.warning("auth_rate_limit_unattributable path=%s", request.url.path)
        await consume_rate_limit(
            [f"unattributable:{request.url.path}"],
            limit=UNATTRIBUTABLE_AUTH_RATE_LIMIT,
            window_seconds=settings.auth_rate_limit_window_seconds,
            detail="Too many unattributable requests; try again later",
        )
        return
    await consume_rate_limit(
        keys,
        limit=settings.auth_rate_limit_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        detail="Too many authentication attempts; try again later",
    )
