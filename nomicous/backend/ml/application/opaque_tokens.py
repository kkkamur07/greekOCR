"""Opaque credential primitives for device pairing.

This is deliberately the *same* scheme as
``backend/users/application/browser_sessions.py``: a 256-bit
``secrets.token_urlsafe(32)`` secret, stored as a keyed HMAC-SHA256 hexdigest in
a ``String(64)`` column and verified with :func:`hmac.compare_digest`. bcrypt is
the codebase's password hash (12 rounds, ``users/application/password.py``) and
is deliberately *not* used here - it is a slow KDF for low-entropy human input,
not for a 256-bit random string.

The credential wire format embeds its own primary key
(``nmd1.<device_id>.<secret>``), matching the session cookie's ``<id>.<secret>``
shape. That is what lets authentication be a primary-key fetch plus one
constant-time compare instead of a scan over hashed columns.

The ``nmd1.`` prefix exists so secret scanners (gitleaks, GitHub push
protection) have a stable pattern to match if a token ever leaks into a repo.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from uuid import UUID

DEVICE_TOKEN_PREFIX = "nmd1"  # noqa: S105 - a format version tag, not a secret
"""Version tag; bump when the token construction changes."""

SECRET_BYTES = 32
"""256 bits of entropy per secret - see the ADR for why this size is not negotiable."""

CONFIRMATION_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""32 unambiguous characters - no I/O/0/1. 256 % 32 == 0, so a byte maps uniformly."""

CONFIRMATION_LENGTH = 8


def new_secret() -> str:
    """Return a fresh 256-bit URL-safe secret."""
    return secrets.token_urlsafe(SECRET_BYTES)


def hash_secret(secret: str, key: str) -> str:
    """Keyed digest of *secret*; the stored form of every device credential."""
    return hmac.new(key.encode(), secret.encode(), hashlib.sha256).hexdigest()


def secret_matches(stored_hash: str | None, secret: str, key: str) -> bool:
    """Constant-time comparison of a presented secret against a stored digest.

    An empty or missing digest never matches: ``token_hash = ''`` marks a device
    that has been approved in the browser but has not collected a token yet.
    """
    if not stored_hash or not secret:
        return False
    return hmac.compare_digest(stored_hash, hash_secret(secret, key))


def format_device_token(device_id: UUID, secret: str) -> str:
    """Build the wire credential. The only place a raw token is ever assembled."""
    return f"{DEVICE_TOKEN_PREFIX}.{device_id}.{secret}"


def device_token_prefix(device_id: UUID) -> str:
    """Log-safe handle for support correlation. Contains no secret material."""
    return f"{DEVICE_TOKEN_PREFIX}.{device_id.hex[:8]}"


def confirmation_code(pairing_id: UUID, key: str) -> str:
    """Human-comparable confirmation code for one pairing request.

    This is **not** a secret and grants nothing. It is not RFC 8628's
    ``user_code`` returning by another name: no endpoint accepts it, so it adds
    no brute-forceable surface. Its only job is to be shown in two places at
    once - by the helper that started the pairing, and on the consent screen -
    so a researcher can see that the computer asking for access is the computer
    in front of them.

    Derived rather than stored: it needs no column, and because the derivation
    is keyed, one pairing's code cannot be computed from another's ``pairing_id``.
    """
    digest = hmac.new(key.encode(), f"confirm:{pairing_id}".encode(), hashlib.sha256).digest()
    chars = "".join(
        CONFIRMATION_ALPHABET[byte % len(CONFIRMATION_ALPHABET)]
        for byte in digest[:CONFIRMATION_LENGTH]
    )
    return f"{chars[:4]}-{chars[4:]}"


def parse_device_token(token: str | None) -> tuple[UUID, str] | None:
    """Split a wire credential into ``(device_id, secret)``; ``None`` when malformed.

    Neither the UUID text nor a URL-safe secret can contain ``.``, so a
    three-way split is unambiguous.
    """
    if not token:
        return None
    parts = token.split(".", 2)
    if len(parts) != 3 or parts[0] != DEVICE_TOKEN_PREFIX or not parts[2]:
        return None
    try:
        return UUID(parts[1]), parts[2]
    except ValueError:
        return None
