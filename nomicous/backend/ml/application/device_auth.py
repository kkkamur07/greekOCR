"""Authenticate an outbound helper by its device token.

Deliberately a *different* credential channel from ``get_current_user``:

* a dedicated ``X-Nomicous-Device-Token`` header, matching the existing
  ``X-Inference-Webhook-Secret`` idiom, so a device token is structurally
  incapable of reaching any route guarded by ``Authorization: Bearer``;
* a different return type (:class:`AuthenticatedDevice`, not ``User``), so no
  existing route can silently start accepting a device token by annotation.

Authorization scope is the ``helper_devices.user_id`` foreign key and nothing
else. A device token can never resolve to a user other than the one who paired
it, so cross-user access is a schema property rather than a code review promise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import InvalidCredentialsError
from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.application.opaque_tokens import parse_device_token, secret_matches
from backend.ml.infrastructure.device_orm_models import HelperDevice
from backend.ml.infrastructure.device_repository import HelperDeviceRepository
from backend.users.infrastructure.orm_models import User

DEVICE_TOKEN_HEADER = "X-Nomicous-Device-Token"  # noqa: S105 - a header name, not a token

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticatedDevice:
    """A paired helper plus the single user it is allowed to act for."""

    device: HelperDevice
    user: User

    @property
    def user_id(self):
        return self.user.id


def _token_is_live(device: HelperDevice, secret: str, *, key: str, now: datetime) -> bool:
    """Accept the current token, or a renewal predecessor inside its overlap."""
    current_is_live = (
        device.token_expires_at is not None
        and device.token_expires_at > now
        and secret_matches(device.token_hash, secret, key)
    )
    previous_is_live = (
        bool(device.previous_token_hash)
        and device.previous_token_expires_at is not None
        and device.previous_token_expires_at > now
        and secret_matches(device.previous_token_hash, secret, key)
    )
    return current_is_live or previous_is_live


async def authenticate_device(
    session: AsyncSession,
    raw_token: str | None,
    *,
    repository: HelperDeviceRepository | None = None,
    settings: DeviceSettings | None = None,
    now: datetime | None = None,
) -> AuthenticatedDevice:
    """Resolve a raw device token to ``(device, user)``.

    Raises :class:`InvalidCredentialsError` (401) for every failure mode -
    malformed, unknown, uncollected, expired, or revoked - so the response never
    tells a caller which one it was.
    """
    settings = settings or get_device_settings()
    repository = repository or HelperDeviceRepository()
    now = now or datetime.now(UTC)

    # Every rejection below is logged with the same fields and the same 401. The
    # ``reason`` is for us, in a server log; the caller learns only "invalid".
    # ``device_id`` is a public identifier, never the presented secret.
    parsed = parse_device_token(raw_token)
    if parsed is None:
        _log_rejection("malformed", None)
        raise InvalidCredentialsError("Invalid device token")
    device_id, secret = parsed

    device = await repository.get_device(session, device_id)
    if device is None:
        _log_rejection("unknown_device", device_id)
        raise InvalidCredentialsError("Invalid device token")
    # Re-read every request: no JWT, no cache, no TTL to wait out, so a
    # revocation from the browser lands on the very next device call.
    if device.revoked_at is not None:
        _log_rejection("revoked", device_id)
        raise InvalidCredentialsError("Device has been removed")
    if not _token_is_live(device, secret, key=settings.hmac_key(), now=now):
        _log_rejection("expired_or_mismatched", device_id)
        raise InvalidCredentialsError("Invalid device token")
    if device.user is None:
        _log_rejection("orphaned_device", device_id)
        raise InvalidCredentialsError("Invalid device token")
    return AuthenticatedDevice(device=device, user=device.user)


def _log_rejection(reason: str, device_id: UUID | None) -> None:
    logger.warning("device_auth_rejected reason=%s device_id=%s", reason, device_id)
