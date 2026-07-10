"""Opaque, rotating browser-session credentials."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.exceptions import AccessDeniedError
from backend.core.settings.auth import AuthSettings
from backend.users.application.jwt_tokens import create_access_token
from backend.users.infrastructure.orm_models import AuthSession, User


@dataclass(frozen=True)
class BrowserSessionTokens:
    access_token: str
    session_cookie: str
    csrf_token: str


def _hash(secret: str, settings: AuthSettings) -> str:
    return hmac.new(settings.jwt_secret.encode(), secret.encode(), hashlib.sha256).hexdigest()


def _new_secret() -> str:
    return secrets.token_urlsafe(32)


def _parse(cookie: str | None) -> tuple[UUID, str] | None:
    if not cookie or "." not in cookie:
        return None
    raw_id, secret = cookie.split(".", 1)
    try:
        return UUID(raw_id), secret
    except ValueError:
        return None


class BrowserSessionService:
    def __init__(self, settings: AuthSettings) -> None:
        self._settings = settings

    def _issue(self, user: User, session: AuthSession) -> BrowserSessionTokens:
        session_secret, csrf_token = _new_secret(), _new_secret()
        session.token_hash = _hash(session_secret, self._settings)
        session.csrf_token_hash = _hash(csrf_token, self._settings)
        return BrowserSessionTokens(
            access_token=create_access_token(user.id, self._settings),
            session_cookie=f"{session.id}.{session_secret}",
            csrf_token=csrf_token,
        )

    async def create(self, db: AsyncSession, user: User) -> BrowserSessionTokens:
        session = AuthSession(
            user_id=user.id,
            token_hash="",
            csrf_token_hash="",
            expires_at=datetime.now(UTC) + timedelta(days=self._settings.session_expire_days),
        )
        tokens = self._issue(user, session)
        db.add(session)
        await db.commit()
        return tokens

    async def rotate(
        self,
        db: AsyncSession,
        *,
        session_cookie: str | None,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> BrowserSessionTokens | None:
        session = await self._valid_session(db, session_cookie)
        if session is None:
            return None
        self._require_csrf(session, csrf_cookie, csrf_header)
        tokens = self._issue(session.user, session)
        await db.commit()
        return tokens

    async def revoke(
        self,
        db: AsyncSession,
        *,
        session_cookie: str | None,
        csrf_cookie: str | None,
        csrf_header: str | None,
    ) -> bool:
        session = await self._valid_session(db, session_cookie)
        if session is None:
            return False
        self._require_csrf(session, csrf_cookie, csrf_header)
        session.revoked_at = datetime.now(UTC)
        await db.commit()
        return True

    async def _valid_session(self, db: AsyncSession, cookie: str | None) -> AuthSession | None:
        parsed = _parse(cookie)
        if parsed is None:
            return None
        session_id, secret = parsed
        result = await db.execute(
            select(AuthSession)
            .options(selectinload(AuthSession.user))
            .where(AuthSession.id == session_id)
            .with_for_update()
        )
        session = result.scalar_one_or_none()
        if session is None or session.revoked_at or session.expires_at <= datetime.now(UTC):
            return None
        if not hmac.compare_digest(session.token_hash, _hash(secret, self._settings)):
            session.revoked_at = datetime.now(UTC)
            await db.commit()
            return None
        return session

    def _require_csrf(
        self, session: AuthSession, csrf_cookie: str | None, csrf_header: str | None
    ) -> None:
        if (
            not csrf_cookie
            or not csrf_header
            or not hmac.compare_digest(csrf_cookie, csrf_header)
            or not hmac.compare_digest(session.csrf_token_hash, _hash(csrf_header, self._settings))
        ):
            raise AccessDeniedError("CSRF validation failed")
