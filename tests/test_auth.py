"""Auth integration tests — real Postgres (kalamos)."""

import psycopg2
import pytest

from backend.core.settings import get_infrastructure_settings


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
    assert hashed.startswith("$2")


@pytest.mark.integration
def test_login_wrong_password_fails(client, registered_user):
    response = client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "wrong-password-xyz"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


@pytest.mark.integration
def test_me_without_token_returns_401(client):
    response = client.get("/me")
    assert response.status_code == 401


@pytest.mark.integration
def test_me_invalid_token_returns_401(client):
    response = client.get("/me", headers={"Authorization": "Bearer not-a-valid-jwt"})
    assert response.status_code == 401


@pytest.mark.integration
def test_projects_without_auth_returns_401(client):
    response = client.get("/projects")
    assert response.status_code == 401


@pytest.mark.integration
def test_projects_with_auth_returns_empty_list(client, auth_headers):
    response = client.get("/projects", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []
