"""Auth integration tests — real Postgres (kalamos)."""

import psycopg2
import pytest
from pydantic import ValidationError

from backend.core.settings.auth import AuthSettings, get_auth_settings
from backend.core.settings import get_app_settings, get_infrastructure_settings
from backend.users.api.rate_limit import clear_auth_rate_limit_state


# --- Register, login, and /me ---
# Tests happy-path auth and JWT bearer flow. Does not test rate limits or password policy edge cases.


@pytest.mark.integration
def test_register_login_me_success(client, unique_user):
    register = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert register.status_code == 201
    token = register.json()["access_token"]
    assert register.json()["token_type"] == "bearer"

    login = client.post(
        "/auth/login",
        json={"email": unique_user["email"], "password": unique_user["password"]},
    )
    assert login.status_code == 200
    assert login.json()["access_token"]

    me = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == unique_user["email"]
    assert body["username"] == unique_user["username"]
    assert "id" in body
    assert "created_at" in body


# --- Password storage ---
# Tests bcrypt hashing at rest. Does not test login or token issuance.


@pytest.mark.integration
def test_register_stores_bcrypt_hash(client, unique_user):
    response = client.post(
        "/auth/register",
        json={
            "email": unique_user["email"],
            "username": unique_user["username"],
            "password": unique_user["password"],
        },
    )
    assert response.status_code == 201

    settings = get_infrastructure_settings()
    with psycopg2.connect(settings.sync_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT hashed_password FROM users WHERE email = %s", (unique_user["email"],))
            hashed = cur.fetchone()[0]
    assert hashed != unique_user["password"]
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert "$12$" in hashed  # bcrypt cost factor 12 (see password.BCRYPT_ROUNDS)


# --- Login and token failures ---
# Tests bad credentials and missing/invalid JWT. Does not test account lockout or email verification.


@pytest.mark.integration
def test_login_wrong_password_fails(client, registered_user):
    response = client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "wrong-password-xyz"},
    )
    assert response.status_code == 401
    assert response.json()["error"] == {
        "code": "UNAUTHORIZED",
        "message": "Invalid email or password",
    }


@pytest.mark.integration
def test_me_without_token_returns_401(client):
    response = client.get("/me")
    assert response.status_code == 401


@pytest.mark.integration
def test_me_invalid_token_returns_401(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-valid-jwt"})
    assert response.status_code == 401


# --- Protected routes ---
# Tests auth is required for member APIs. Does not test per-resource authorization rules.


@pytest.mark.integration
def test_projects_without_auth_returns_401(client):
    response = client.get("/projects")
    assert response.status_code == 401


@pytest.mark.integration
def test_projects_with_auth_returns_empty_list(client, auth_headers):
    response = client.get("/projects", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


# --- Auth settings ---
# Tests required env config. Does not hit HTTP or the database.


def test_auth_settings_require_jwt_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValidationError):
        AuthSettings(_env_file=None)


# --- Login rate limiting ---
# Tests per-client throttling on /auth/login. Does not test register rate limits or successful logins.


def test_login_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setenv("AUTH_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")

    get_auth_settings.cache_clear()
    clear_auth_rate_limit_state()

    first = client.post(
        "/auth/login",
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert first.status_code == 401

    limited = client.post(
        "/auth/login",
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert limited.status_code == 429


def test_login_rate_limit_ignores_forwarded_for_without_trusted_proxy(client, monkeypatch):
    monkeypatch.setenv("AUTH_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")
    get_auth_settings.cache_clear()
    get_app_settings.cache_clear()
    clear_auth_rate_limit_state()

    first = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert first.status_code == 401

    limited = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert limited.status_code == 429

    other_client = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.11, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert other_client.status_code == 429


def test_login_rate_limit_uses_forwarded_for_from_trusted_proxy(client, monkeypatch):
    monkeypatch.setenv("AUTH_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("BEHIND_PROXY", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "testclient")
    
    get_auth_settings.cache_clear()
    get_app_settings.cache_clear()
    clear_auth_rate_limit_state()

    first = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert first.status_code == 401

    limited = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert limited.status_code == 429

    other_client = client.post(
        "/auth/login",
        headers={"X-Forwarded-For": "203.0.113.11, 10.0.0.1"},
        json={"email": "missing@test.kalamos", "password": "wrong-password-xyz"},
    )
    assert other_client.status_code == 401
