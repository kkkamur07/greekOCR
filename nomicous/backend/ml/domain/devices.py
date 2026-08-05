"""Device pairing domain vocabulary.

The protocol is RFC 8628 (device authorization grant) with the typable
``user_code`` removed: our device can open a browser, so there is no reason to
carry a short human-entered secret and no reason to defend one.
"""

from __future__ import annotations

from enum import StrEnum


class DeviceStatus(StrEnum):
    """Liveness of a paired helper, derived from ``last_seen_at``."""

    pairing = "pairing"
    """Approved in the browser but the helper has not collected its token yet."""

    online = "online"
    idle = "idle"
    offline = "offline"
    revoked = "revoked"


class PairingStatus(StrEnum):
    """Poll result for the helper's token collection loop.

    Every one of these is returned inside a ``200`` body. The platform error
    envelope replaces ``HTTPException.detail`` with a fixed public string
    (``backend/core/app.py``), so a machine-readable protocol state can never
    survive a non-2xx response.
    """

    authorization_pending = "authorization_pending"
    slow_down = "slow_down"
    access_denied = "access_denied"
    expired = "expired"
    approved = "approved"
