"""Opaque, rotating browser-session credentials."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
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
            # Binding the access token to this session id is what lets logout
            # invalidate it: revoking the row invalidates every token it minted.
            access_token=create_access_token(user.id, self._settings, session_id=session.id),
            session_cookie=f"{session.id}.{session_secret}",
            csrf_token=csrf_token,
        )

    async def resolve_active_session_user(
        self, db: AsyncSession, *, session_id: UUID, user_id: UUID
    ) -> User | None:
        """Return the user when ``session_id`` is a live session belonging to them.

        A single joined read replaces the plain user lookup an authenticated
        request already performed, so enforcing revocation costs no extra query.
        """
        result = await db.execute(
            select(User)
            .join(AuthSession, AuthSession.user_id == User.id)
            .where(
                AuthSession.id == session_id,
                User.id == user_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > datetime.now(UTC),
            )
        )
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, user: User) -> BrowserSessionTokens:
        # Assign id before issuing the cookie - SQLAlchemy column defaults run on flush.
        session = AuthSession(
            id=uuid.uuid4(),
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
        csrf_header: str | None,
    ) -> BrowserSessionTokens | None:
        session = await self._valid_session(db, session_cookie)
        if session is None:
            return None
        self._require_csrf(session, csrf_header)
        tokens = self._issue(session.user, session)
        await db.commit()
        return tokens

    async def revoke(
        self,
        db: AsyncSession,
        *,
        session_cookie: str | None,
        csrf_header: str | None,
    ) -> bool:
        session = await self._valid_session(db, session_cookie)
        if session is None:
            return False
        self._require_csrf(session, csrf_header)
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

    def _require_csrf(self, session: AuthSession, csrf_header: str | None) -> None:
        """Prove the caller holds *this session's* CSRF secret.

        The proof is the header alone, checked against the per-session hash this
        server stored when it issued the token. That is the synchronizer-token
        pattern: a cross-site page cannot read the secret (the response that
        carries it is CORS-protected, and so is the cookie that carries it), and
        it cannot set ``X-CSRF-Token`` on a form post at all, so it cannot
        produce a header that hashes to ``csrf_token_hash``.

        The CSRF *cookie* is deliberately not read here any more, and the
        reasoning is worth keeping because deleting a check is the kind of
        change that looks like a regression:

        * The pair of checks it used to satisfy - cookie present, cookie equal
          to header - is the unsigned double-submit pattern. Double submit is a
          *substitute* for a server-side secret, for servers that keep no
          per-session state. This one keeps that state, so the two stack rather
          than compose: an attacker who cannot produce the header is already
          stopped, and an attacker who can produce it can trivially also let the
          browser attach the matching cookie. There is no attacker that the
          equality check stops and the hash check does not.
        * Requiring the cookie *costs* something real. The cookie is what the
          client reads to build the header, and it is shared across the whole
          registrable domain; the value a browser lets script read on
          ``app.nomicous.com`` and the value it attaches to a request to
          ``api.nomicous.com`` are not guaranteed to be the same one under a
          cookie policy that partitions or blocks the sibling-subdomain read.
          When they diverge, equality fails on a request that is completely
          legitimate - which is the shape of the Safari 403 this change is a
          candidate fix for.

        The cookie is still set, and a client that can read it still sends it;
        nothing about the wire format changes. Only the server's willingness to
        insist on it does.
        """
        if not csrf_header or not hmac.compare_digest(
            session.csrf_token_hash, _hash(csrf_header, self._settings)
        ):
            raise AccessDeniedError("CSRF validation failed")
