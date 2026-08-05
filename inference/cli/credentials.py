"""Where the **device token** lives on a researcher's machine, and who may read it.

ADR 0001 states the risk this file *is*: a long-lived credential now sits on a
laptop where none existed, and whoever can read it can claim that researcher's
`local` work until the device is revoked. The one control this layer owns is the
mode bits - `0600` in a `0700` directory - so another account on a shared machine
cannot claim jobs as them. Everything else (per-user scope, revocation on the
next call) is enforced by the platform.

The file is written through a temporary file in the same directory and renamed,
so an interrupted write leaves the previous credential intact rather than a
truncated one. `os.open` with `0o600` is not enough on its own: the process
umask is subtracted from that mode, so an `fchmod` follows it.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# The same root the **Hub cache** uses (`inference/hub/cache.py`). One directory
# in the researcher's home for everything the installed package writes, rather
# than one per subsystem.
DEFAULT_NOMICOUS_HOME = Path.home() / ".nomicous"
NOMICOUS_HOME_ENV = "NOMICOUS_HOME"

CREDENTIAL_FILENAME = "device.json"

CREDENTIAL_FILE_MODE = 0o600
CREDENTIAL_DIR_MODE = 0o700


def nomicous_home() -> Path:
    """The directory the CLI keeps its state in, overridable for tests.

    Read from the environment on every call rather than at import time, for the
    same reason `default_cache_root()` is: the process that sets it is often not
    the one that imported this module.
    """
    override = os.environ.get(NOMICOUS_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_NOMICOUS_HOME


def credential_path() -> Path:
    return nomicous_home() / CREDENTIAL_FILENAME


@dataclass(frozen=True)
class DeviceCredential:
    """One paired machine's **device token** and what it is bound to.

    `platform_url` is part of the record because the credential is only
    meaningful against the platform that minted it; pairing against a second one
    is a different device row, not a replacement for this one.
    """

    platform_url: str
    device_id: str
    device_token: str
    account_email: str
    device_name: str
    token_expires_at: datetime | None
    paired_at: datetime

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Whether the 180-day TTL has run out on this machine's clock.

        Only ever used to *explain* a rejection, never to pre-empt one: the
        platform is the authority on whether a credential is live, and a laptop
        with a wrong clock must not talk itself out of a working token.
        """
        if self.token_expires_at is None:
            return False
        return self.token_expires_at <= (now or datetime.now(UTC))

    def to_json(self) -> dict:
        return {
            "platform_url": self.platform_url,
            "device_id": self.device_id,
            "device_token": self.device_token,
            "account_email": self.account_email,
            "device_name": self.device_name,
            "token_expires_at": _isoformat(self.token_expires_at),
            "paired_at": _isoformat(self.paired_at),
        }


class CredentialError(RuntimeError):
    """The credential file exists but cannot be used as one."""


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def load_credential(path: Path | None = None) -> DeviceCredential | None:
    """The stored credential, or `None` when this machine is not paired.

    A malformed file raises rather than reading as "not paired": silently
    starting a second pairing because a byte flipped would leave an orphan
    device row on the account and no explanation for it.
    """
    target = path or credential_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialError(f"Cannot read {target}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CredentialError(f"{target} does not contain a device credential")

    missing = [
        field
        for field in ("platform_url", "device_id", "device_token")
        if not isinstance(payload.get(field), str) or not payload[field]
    ]
    if missing:
        raise CredentialError(f"{target} is missing {', '.join(missing)}")

    return DeviceCredential(
        platform_url=payload["platform_url"],
        device_id=payload["device_id"],
        device_token=payload["device_token"],
        account_email=payload.get("account_email") or "",
        device_name=payload.get("device_name") or "",
        token_expires_at=_parse_datetime(payload.get("token_expires_at")),
        paired_at=_parse_datetime(payload.get("paired_at")) or datetime.now(UTC),
    )


def save_credential(credential: DeviceCredential, path: Path | None = None) -> Path:
    """Write the credential owner-only, and return where it landed."""
    target = path or credential_path()
    directory = target.parent
    directory.mkdir(mode=CREDENTIAL_DIR_MODE, parents=True, exist_ok=True)
    # `mkdir(mode=...)` does nothing when the directory already exists, and the
    # **Hub cache** may well have created it first. Narrow it either way: this
    # directory holds a bearer credential for the researcher's account.
    os.chmod(directory, CREDENTIAL_DIR_MODE)

    temporary = directory / f".{target.name}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, CREDENTIAL_FILE_MODE)
    try:
        # The umask is subtracted from the mode above, so it is re-applied here.
        # A umask of 0 would otherwise be the only configuration in which the
        # requested mode is the mode that lands.
        os.fchmod(descriptor, CREDENTIAL_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(credential.to_json(), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, target)
    return target


def file_mode(path: Path) -> int:
    """Permission bits of `path`, for reporting and for tests to assert on."""
    return stat.S_IMODE(path.stat().st_mode)
