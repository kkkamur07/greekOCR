"""JWT issue and validation."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from jwt import InvalidTokenError

from backend.core.settings.auth import AuthSettings


def create_access_token(user_id: UUID, settings: AuthSettings) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "exp": expire, "typ": "access"}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: AuthSettings) -> UUID:
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
        return UUID(sub)
    except (InvalidTokenError, ValueError, TypeError) as exc:
        raise InvalidTokenError("invalid token") from exc
