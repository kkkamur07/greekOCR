"""How a browser session's CSRF token is delivered, and what proves possession of it.

The token reaches the client through two channels: the readable ``greekocr-csrf``
cookie, and - since the Safari ``/auth/refresh`` 403 investigation - the auth
response body. The cookie is set by ``api.nomikos.app`` for ``.nomikos.app`` so
that script on ``app.nomikos.app`` can read it back into ``X-CSRF-Token``, and
that sibling-subdomain read is the step a stricter cookie policy interferes with.

These tests pin three things: both channels carry the token, the cookie channel is
untouched by the addition, and what the server accepts as proof is the header
hashed against the session's stored secret - with or without a cookie beside it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import Response

from backend.core.exceptions import AccessDeniedError
from backend.core.settings import AuthSettings, reset_settings_caches
from backend.users.api.schemas import TokenResponse
from backend.users.application.browser_sessions import (
    BrowserSessionService,
    BrowserSessionTokens,
)
from backend.users.infrastructure.orm_models import AuthSession

STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


@pytest.fixture(autouse=True)
def platform_settings(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    reset_settings_caches()
    yield
    reset_settings_caches()


def _auth_routes():
    """The auth router module, imported only once a signing key is in the environment.

    It builds an ``AuthService`` at import time, which reads settings, so a
    module-scope import here would fail collection before any fixture ran.
    """
    from backend.users.api import auth

    return auth


def _settings() -> AuthSettings:
    return AuthSettings(JWT_SECRET=STRONG_SECRET, _env_file=None)


def _new_session(settings: AuthSettings) -> tuple[AuthSession, BrowserSessionTokens]:
    """A live session and the credentials issued for it, without a database."""
    session = AuthSession(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        token_hash="",
        csrf_token_hash="",
        expires_at=datetime.now(UTC) + timedelta(days=settings.session_expire_days),
    )
    user = type("User", (), {"id": session.user_id})()
    return session, BrowserSessionService(settings)._issue(user, session)


def _set_cookies(response: Response) -> dict[str, str]:
    """The ``Set-Cookie`` headers on a response, keyed by cookie name."""
    return {
        header.split("=", 1)[0]: header.split("=", 1)[1].split(";", 1)[0]
        for header in response.headers.getlist("set-cookie")
    }


# --- Two delivery channels ---


def test_token_response_always_carries_the_csrf_token() -> None:
    """The field is optional in the schema; the server never leaves it out.

    Optional is how a *client* has to treat it - the frontend deploys separately
    and can meet an API that predates the field. This is the assertion that keeps
    that from also being true of the API itself.
    """
    _session, tokens = _new_session(_settings())

    body = _auth_routes()._issue_session(Response(), tokens)

    assert body.csrf_token == tokens.csrf_token
    assert body.csrf_token
    assert body.access_token == tokens.access_token


def test_issuing_a_session_still_sets_both_cookies() -> None:
    """The body copy is a second channel, not a replacement for the first.

    A browser that reads the cookie today must keep working exactly as it did,
    including one that established its session before this code shipped.
    """
    _session, tokens = _new_session(_settings())
    response = Response()

    _auth_routes()._issue_session(response, tokens)

    cookies = _set_cookies(response)
    assert cookies["__Host-greekocr-session"] == tokens.session_cookie
    assert cookies["greekocr-csrf"] == tokens.csrf_token
    raw = "\n".join(response.headers.getlist("set-cookie"))
    assert "HttpOnly" in raw  # the session cookie, still not readable by script
    assert "Secure" in raw


def test_the_body_copy_is_the_value_the_cookie_carries() -> None:
    """Same value, two channels - so either one alone builds a valid header."""
    _session, tokens = _new_session(_settings())
    cookie_only, both = Response(), Response()

    _auth_routes()._set_session_cookies(cookie_only, tokens)
    body = _auth_routes()._issue_session(both, tokens)

    assert _set_cookies(cookie_only) == _set_cookies(both)
    assert body.csrf_token == _set_cookies(cookie_only)["greekocr-csrf"]


def test_the_schema_keeps_the_field_optional_for_clients() -> None:
    """A client built against this contract can be talking to an older API."""
    assert TokenResponse(access_token="jwt").csrf_token is None
    assert "csrf_token" not in TokenResponse.model_json_schema()["required"]


# --- What the server accepts as proof ---


def test_the_header_alone_authorises() -> None:
    """The Safari case: script could not read the cookie, so it echoed the body copy.

    Nothing about the request has to carry the CSRF cookie for this to be proof.
    The header is checked against a per-session secret this server hashed and
    stored when it issued the token, which a cross-site page cannot obtain.
    """
    settings = _settings()
    session, tokens = _new_session(settings)

    BrowserSessionService(settings)._require_csrf(session, tokens.csrf_token)


def test_the_header_read_back_from_the_cookie_still_authorises() -> None:
    """The path every browser that works today takes, unchanged."""
    settings = _settings()
    session, tokens = _new_session(settings)
    response = Response()
    _auth_routes()._issue_session(response, tokens)
    read_from_cookie = _set_cookies(response)["greekocr-csrf"]

    BrowserSessionService(settings)._require_csrf(session, read_from_cookie)


@pytest.mark.parametrize("header", [None, "", "not-the-session-token"])
def test_a_missing_or_wrong_header_is_refused(header: str | None) -> None:
    settings = _settings()
    session, _tokens = _new_session(settings)

    with pytest.raises(AccessDeniedError):
        BrowserSessionService(settings)._require_csrf(session, header)


def test_another_sessions_token_is_refused() -> None:
    """The token is bound to one session, so a valid-looking one is not enough."""
    settings = _settings()
    session, _tokens = _new_session(settings)
    _other_session, other_tokens = _new_session(settings)

    with pytest.raises(AccessDeniedError):
        BrowserSessionService(settings)._require_csrf(session, other_tokens.csrf_token)


def test_a_token_minted_under_a_different_signing_key_is_refused() -> None:
    """The stored hash is keyed on ``JWT_SECRET``, not a bare digest."""
    settings = _settings()
    session, tokens = _new_session(settings)
    other_key = AuthSettings(JWT_SECRET=STRONG_SECRET[::-1], _env_file=None)

    with pytest.raises(AccessDeniedError):
        BrowserSessionService(other_key)._require_csrf(session, tokens.csrf_token)


def test_rotation_retires_the_previous_token() -> None:
    """Which is why a stale in-memory copy in a second tab has to be recoverable."""
    settings = _settings()
    session, first = _new_session(settings)
    user = type("User", (), {"id": session.user_id})()

    second = BrowserSessionService(settings)._issue(user, session)

    assert second.csrf_token != first.csrf_token
    BrowserSessionService(settings)._require_csrf(session, second.csrf_token)
    with pytest.raises(AccessDeniedError):
        BrowserSessionService(settings)._require_csrf(session, first.csrf_token)
