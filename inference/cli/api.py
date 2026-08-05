"""The CLI's side of the device-pairing protocol, over plain HTTP.

The protocol itself already exists and is tested on the platform
(`backend/ml/api/device_pairing.py`): RFC 8628's device authorization grant with
the typable `user_code` removed. Nothing here designs anything - it is the client
for a wire contract that is already settled, and the shapes below are the
platform's DTOs read back.

`urllib.request` rather than an HTTP library, because `[project].dependencies` is
the closure that reaches a researcher's laptop and this is three requests with no
streaming, no connection reuse, and no authentication scheme to negotiate. The
**claim** loop in #57 has the same properties.
"""

from __future__ import annotations

import json
import platform as platform_module
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from inference.cli.version import installed_version

DEFAULT_PLATFORM_URL = "https://api.nomicous.com"
PLATFORM_URL_ENV = "NOMICOUS_API_URL"

DEVICE_TOKEN_HEADER = "X-Nomicous-Device-Token"
AGENT_VERSION_HEADER = "X-Nomicous-Agent-Version"
"""Which build of the agent is calling. The **version floor** is enforced on the
**claim** path only (`backend/ml/api/agent_version.py`); the constant lives here
so the run loop in #57 states the same version this CLI reports."""

# The platform caps these on the way in (`PairingStartRequest`). Truncating here
# rather than sending an over-long value keeps a long hostname from turning into
# a 422 a researcher cannot act on.
DEVICE_NAME_LIMIT = 120
PLATFORM_LIMIT = 32
VERSION_LIMIT = 32

REQUEST_TIMEOUT_SECONDS = 30.0

# Pairing states, as the platform names them (`backend/ml/domain/devices.py`).
# Every one of them arrives inside a 200 body - the platform's error envelope
# replaces `HTTPException.detail` with a fixed public string, so a
# machine-readable protocol state cannot survive a non-2xx response.
STATUS_AUTHORIZATION_PENDING = "authorization_pending"
STATUS_SLOW_DOWN = "slow_down"
STATUS_ACCESS_DENIED = "access_denied"
STATUS_EXPIRED = "expired"
STATUS_APPROVED = "approved"


class PlatformError(RuntimeError):
    """The platform could not be reached, or answered something unusable."""


@dataclass(frozen=True)
class StartedPairing:
    """What `POST /device/v1/pairings` hands an unpaired machine."""

    pairing_id: str
    device_code: str
    verification_url: str
    confirmation_code: str
    """Not a secret and not a `user_code`: no endpoint accepts it. It is a keyed
    derivation of the pairing id, shown here so the researcher can compare it
    with the consent screen (ADR 0001, decision 13)."""
    expires_in: int
    interval_seconds: int


@dataclass(frozen=True)
class PairingPoll:
    """One result from the token collection loop."""

    status: str
    interval_seconds: int
    device_id: str | None = None
    device_token: str | None = None
    token_expires_at: datetime | None = None
    account_email: str | None = None


@dataclass(frozen=True)
class DeviceIdentity:
    """What `GET /device/v1/self` tells a paired machine about itself."""

    device_id: str
    name: str
    account_email: str
    token_expires_at: datetime | None


def default_platform_url(environ: dict[str, str] | None = None) -> str:
    import os

    source = environ if environ is not None else os.environ
    return (source.get(PLATFORM_URL_ENV) or DEFAULT_PLATFORM_URL).rstrip("/")


def this_machine_name() -> str:
    """What the consent screen will call this computer.

    Self-reported and rendered as inert text by the `/pair` page, which says so:
    a convincing name is not evidence, and the **confirmation code** is what a
    researcher actually checks.
    """
    name = socket.gethostname().strip() or "unnamed machine"
    return name[:DEVICE_NAME_LIMIT]


def this_machine_platform() -> str:
    machine = platform_module.machine() or "unknown"
    return f"{platform_module.system().lower()}-{machine}"[:PLATFORM_LIMIT]


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class PlatformClient:
    """Three requests: start a pairing, poll for the token, confirm the token."""

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        """Send one request; return `(status, decoded body)`.

        Non-2xx statuses are returned rather than raised, because on this surface
        they are answers: a 401 from `/device/v1/self` is how a revoked device
        finds out. Only a failure to reach the platform at all is an exception.
        """
        url = f"{self.base_url}{path}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method=method)
        request.add_header("Accept", "application/json")
        # Recorded on `helper_pairings.user_agent` for support correlation.
        request.add_header("User-Agent", f"nomicous-inference/{installed_version()}")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except urllib.error.URLError as exc:
            raise PlatformError(f"Cannot reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PlatformError(f"{self.base_url} did not answer within {self.timeout:g}s") from exc
        except OSError as exc:  # pragma: no cover - socket-level failures
            raise PlatformError(f"Cannot reach {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def start_pairing(
        self,
        *,
        device_name: str,
        device_platform: str,
        agent_version: str,
        capabilities: dict | None = None,
    ) -> StartedPairing:
        status, body = self._request(
            "POST",
            "/device/v1/pairings",
            body={
                "device_name": device_name,
                "platform": device_platform,
                "helper_version": agent_version[:VERSION_LIMIT],
                "capabilities": capabilities or {},
            },
        )
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving device pairing. "
                "It is disabled by default in production until the consent page ships."
            )
        if status == 429:
            raise PlatformError(
                f"{self.base_url} is refusing new pairing requests right now. Try again shortly."
            )
        if status != 201 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "starting the pairing"))
        try:
            return StartedPairing(
                pairing_id=body["pairing_id"],
                device_code=body["device_code"],
                verification_url=body["verification_url"],
                confirmation_code=body["confirmation_code"],
                expires_in=int(body["expires_in"]),
                interval_seconds=int(body["interval_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlatformError(f"{self.base_url} returned an unusable pairing response") from exc

    def collect_token(self, *, pairing_id: str, device_code: str) -> PairingPoll:
        status, body = self._request(
            "POST",
            "/device/v1/pairings/token",
            body={"pairing_id": pairing_id, "device_code": device_code},
        )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "waiting for approval"))
        return PairingPoll(
            status=str(body.get("status") or ""),
            interval_seconds=int(body.get("interval_seconds") or 0),
            device_id=body.get("device_id"),
            device_token=body.get("device_token"),
            token_expires_at=_parse_datetime(body.get("token_expires_at")),
            account_email=body.get("account_email"),
        )

    def read_self(self, *, device_token: str) -> DeviceIdentity | None:
        """Confirm a stored credential. `None` means the platform refused it.

        The platform answers every rejection - unknown, expired, revoked - with
        the same 401 and the same public message, so this cannot say *why*. The
        caller pairs it with the stored expiry, which is the one thing it does
        know locally, to name the likely cause.
        """
        status, body = self._request(
            "GET", "/device/v1/self", headers={DEVICE_TOKEN_HEADER: device_token}
        )
        if status == 401:
            return None
        if status == 404:
            raise PlatformError(
                f"{self.base_url} is not serving the device layer, so this "
                "machine's credential cannot be checked."
            )
        if status != 200 or not isinstance(body, dict):
            raise PlatformError(_unexpected(status, body, "checking this machine's credential"))
        return DeviceIdentity(
            device_id=str(body.get("device_id") or ""),
            name=str(body.get("name") or ""),
            account_email=str(body.get("account_email") or ""),
            token_expires_at=_parse_datetime(body.get("token_expires_at")),
        )


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _unexpected(status: int, body: Any, doing: str) -> str:
    message = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
    detail = f": {message}" if message else ""
    return f"The platform answered {status} while {doing}{detail}"
