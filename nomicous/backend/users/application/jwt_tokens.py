"""JWT issue and validation.

Access tokens carry the id of the browser session that minted them (``sid``).
That claim is what makes logout effective: the session row is the revocation
record, so a token presented after its session is revoked or expired is refused
on the next request instead of staying valid until ``exp``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from jwt import InvalidTokenError

from backend.core.settings.auth import AuthSettings


@dataclass(frozen=True)
class AccessTokenClaims:
    user_id: UUID
    #: ``None`` only for tokens minted outside a browser session. Authenticated
    #: API routes reject those - see ``backend.users.api.dependencies``.
    session_id: UUID | None


def create_access_token(
    user_id: UUID,
    settings: AuthSettings,
    *,
    session_id: UUID | None = None,
) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload: dict[str, object] = {"sub": str(user_id), "exp": expire, "typ": "access"}
    if session_id is not None:
        payload["sid"] = str(session_id)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: AuthSettings) -> AccessTokenClaims:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "typ"]},
        )
        if payload.get("typ") != "access":
            raise InvalidTokenError("invalid token type")
        sub = payload.get("sub")
        if not sub:
            raise InvalidTokenError("missing sub")
        raw_sid = payload.get("sid")
        session_id = UUID(raw_sid) if raw_sid else None
        return AccessTokenClaims(user_id=UUID(sub), session_id=session_id)
    except (InvalidTokenError, ValueError, TypeError) as exc:
        raise InvalidTokenError("invalid token") from exc
