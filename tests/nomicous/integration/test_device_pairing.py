"""End-to-end device pairing over real HTTP against Postgres.

These tests run against the application the deployment builds, through the
session-scoped ``client`` fixture. They used to assemble a local app instead,
because the device routers construct their service at import time and the poll
cadence had to be collapsed first; that env var now lives in the integration
conftest, which is imported earlier still, so the local app is unnecessary.

Removing it fixed four tests. A second ``TestClient`` runs its own event loop,
and the asyncpg pool is bound to whichever loop created it, so every query from
the second client failed with "attached to a different loop" (issue #63).

It also removes a hazard worth naming: an app assembled by the test module is
never evidence that anything is reachable in the deployed application. That is
how this phase once shipped with its routers unmounted and a green suite.
``test_device_routes_are_served_by_the_real_app`` below and
``test_device_routes_are_mounted_on_the_real_app`` in the unit suite hold that
line; now the rest of the module does too.

Requires migration ``003_helper_devices``.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from infrastructure.db import sync_engine

pytestmark = pytest.mark.integration


@pytest.fixture
def device_client(client: TestClient) -> TestClient:
    """The real application, on the one event loop its connection pool belongs to."""
    return client


def _register(client: TestClient) -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    response = client.post(
        "/auth/register",
        json={
            "email": f"device-{suffix}@test.kalamos",
            "username": f"device_{suffix}",
            "password": "test-pass-123",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _start_pairing(client: TestClient, name: str = "Researcher laptop") -> tuple[dict, str]:
    response = client.post(
        "/device/v1/pairings",
        json={
            "device_name": name,
            "platform": "darwin-arm64",
            "helper_version": "0.2.0",
            "capabilities": {"runtime": "torch"},
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert "?" not in body["verification_url"]
    # The helper must be able to show this next to the consent screen's copy.
    assert body["confirmation_code"]
    verification_token = body["verification_url"].split("#", 1)[1]
    return body, verification_token


def _approve(client: TestClient, headers: dict, pairing_id: str, verification_token: str) -> dict:
    response = client.post(
        f"/devices/pairings/{pairing_id}/approve",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _collect(client: TestClient, pairing_id: str, device_code: str) -> dict:
    response = client.post(
        "/device/v1/pairings/token",
        json={"pairing_id": pairing_id, "device_code": device_code},
    )
    # Protocol states always ride in a 200 body: the error envelope replaces
    # HTTPException.detail with a fixed public string.
    assert response.status_code == 200, response.text
    return response.json()


def _pair_device(client: TestClient, headers: dict, name: str = "Researcher laptop") -> dict:
    started, verification_token = _start_pairing(client, name)
    _approve(client, headers, started["pairing_id"], verification_token)
    approved = _collect(client, started["pairing_id"], started["device_code"])
    assert approved["status"] == "approved", approved
    approved["_pairing"] = started
    approved["_verification_token"] = verification_token
    return approved


def test_mint_redeem_then_authenticate(device_client: TestClient) -> None:
    headers = _register(device_client)
    started, verification_token = _start_pairing(device_client)

    pending = _collect(device_client, started["pairing_id"], started["device_code"])
    assert pending["status"] == "authorization_pending"
    assert pending["device_token"] is None

    lookup = device_client.post(
        "/devices/pairings/lookup",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert lookup.status_code == 200, lookup.text
    assert lookup.json()["device_name"] == "Researcher laptop"
    assert lookup.json()["pairing_id"] == started["pairing_id"]
    # The consent screen shows the code the helper showed, and no IP-derived
    # signal at all - behind an unallowlisted proxy those were the same value
    # for every user on the platform.
    assert lookup.json()["confirmation_code"] == started["confirmation_code"]
    assert "same_network" not in lookup.json()
    assert "request_ip" not in lookup.json()

    device = _approve(device_client, headers, started["pairing_id"], verification_token)
    assert device["status"] == "pairing"
    assert device["token_prefix"] == ""

    approved = _collect(device_client, started["pairing_id"], started["device_code"])
    assert approved["status"] == "approved"
    token = approved["device_token"]
    assert token.startswith("nmd1.")

    me = device_client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: token})
    assert me.status_code == 200, me.text
    assert me.json()["device_id"] == approved["device_id"]

    listed = device_client.get("/devices", headers=headers)
    assert listed.status_code == 200
    entries = listed.json()
    assert len(entries) == 1
    assert entries[0]["status"] == "online"
    assert entries[0]["token_prefix"].startswith("nmd1.")


def test_device_token_cannot_reach_another_users_devices(device_client: TestClient) -> None:
    owner_headers = _register(device_client)
    outsider_headers = _register(device_client)

    approved = _pair_device(device_client, owner_headers, "Owner laptop")
    token = approved["device_token"]
    device_id = approved["device_id"]

    # The outsider sees nothing and cannot revoke it.
    assert device_client.get("/devices", headers=outsider_headers).json() == []
    denied = device_client.delete(f"/devices/{device_id}", headers=outsider_headers)
    assert denied.status_code == 403, denied.text

    # A device token is not a Bearer credential and cannot impersonate a user.
    as_bearer = device_client.get("/devices", headers={"Authorization": f"Bearer {token}"})
    assert as_bearer.status_code == 401

    # A user's access token is not a device credential either.
    owner_access_token = owner_headers["Authorization"].split(" ", 1)[1]
    as_device = device_client.get(
        "/device/v1/self", headers={DEVICE_TOKEN_HEADER: owner_access_token}
    )
    assert as_device.status_code == 401

    # The owner still has it.
    assert [
        entry["id"] for entry in device_client.get("/devices", headers=owner_headers).json()
    ] == [device_id]


def test_revoked_device_token_is_rejected_immediately(device_client: TestClient) -> None:
    headers = _register(device_client)
    approved = _pair_device(device_client, headers)
    token = approved["device_token"]

    assert (
        device_client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: token}).status_code
        == 200
    )

    revoked = device_client.delete(f"/devices/{approved['device_id']}", headers=headers)
    assert revoked.status_code == 204

    after = device_client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: token})
    assert after.status_code == 401
    assert device_client.get("/devices", headers=headers).json() == []


def test_pair_code_cannot_be_redeemed_twice(device_client: TestClient) -> None:
    headers = _register(device_client)
    approved = _pair_device(device_client, headers)
    started = approved["_pairing"]

    second = _collect(device_client, started["pairing_id"], started["device_code"])
    assert second["status"] == "access_denied"
    assert second["device_token"] is None


def test_expired_pairing_cannot_be_approved(device_client: TestClient) -> None:
    headers = _register(device_client)
    started, verification_token = _start_pairing(device_client)

    with sync_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE helper_pairings SET expires_at = now() - interval '1 minute' "
                "WHERE id = :pairing_id"
            ),
            {"pairing_id": started["pairing_id"]},
        )

    expired = _collect(device_client, started["pairing_id"], started["device_code"])
    assert expired["status"] == "expired"
    assert expired["device_token"] is None

    # And the browser can no longer see it at all.
    lookup = device_client.post(
        "/devices/pairings/lookup",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert lookup.status_code == 404

    approve = device_client.post(
        f"/devices/pairings/{started['pairing_id']}/approve",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert approve.status_code == 409


def test_wrong_device_code_never_yields_a_token(device_client: TestClient) -> None:
    headers = _register(device_client)
    started, verification_token = _start_pairing(device_client)
    _approve(device_client, headers, started["pairing_id"], verification_token)

    wrong = _collect(device_client, started["pairing_id"], "not-the-device-code")
    assert wrong["status"] == "access_denied"
    assert wrong["device_token"] is None

    # The device row exists but has no usable credential.
    with sync_engine.begin() as connection:
        token_hash = connection.execute(text("SELECT token_hash FROM helper_devices")).scalar_one()
    assert token_hash == ""


def test_approve_requires_the_verification_token(device_client: TestClient) -> None:
    headers = _register(device_client)
    started, _ = _start_pairing(device_client)
    response = device_client.post(
        f"/devices/pairings/{started['pairing_id']}/approve",
        headers=headers,
        json={"verification_token": "not-the-verification-token"},
    )
    assert response.status_code == 404


def test_lookup_of_an_unknown_verification_token_is_404(device_client: TestClient) -> None:
    headers = _register(device_client)
    response = device_client.post(
        "/devices/pairings/lookup",
        headers=headers,
        json={"verification_token": "definitely-not-a-real-token"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_pairing_routes_require_authentication(device_client: TestClient) -> None:
    started, verification_token = _start_pairing(device_client)
    assert device_client.get("/devices").status_code == 401
    assert (
        device_client.post(
            "/devices/pairings/lookup", json={"verification_token": verification_token}
        ).status_code
        == 401
    )
    assert (
        device_client.post(
            f"/devices/pairings/{started['pairing_id']}/approve",
            json={"verification_token": verification_token},
        ).status_code
        == 401
    )
    assert device_client.get("/device/v1/self").status_code == 401
    assert device_client.post("/device/v1/token/renew").status_code == 401


def test_raw_secrets_are_absent_from_the_database(device_client: TestClient) -> None:
    headers = _register(device_client)
    approved = _pair_device(device_client, headers)
    started = approved["_pairing"]
    verification_token = approved["_verification_token"]
    secret = approved["device_token"].split(".", 2)[2]

    with sync_engine.begin() as connection:
        device_rows = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM helper_devices WHERE id = :device_id"),
                {"device_id": approved["device_id"]},
            ).mappings()
        ]
        pairing_rows = [
            dict(row)
            for row in connection.execute(
                text("SELECT * FROM helper_pairings WHERE id = :pairing_id"),
                {"pairing_id": started["pairing_id"]},
            ).mappings()
        ]

    assert device_rows and pairing_rows
    for row in device_rows + pairing_rows:
        for column, value in row.items():
            rendered = str(value)
            assert secret not in rendered, f"raw device secret found in {column}"
            assert started["device_code"] not in rendered, f"raw device_code found in {column}"
            assert verification_token not in rendered, f"raw verification token found in {column}"

    # And no read endpoint hands any of it back.
    listed = device_client.get("/devices", headers=headers).text
    for leaked in (secret, started["device_code"], verification_token):
        assert leaked not in listed


def test_token_renewal_replaces_the_credential(device_client: TestClient) -> None:
    headers = _register(device_client)
    token = _pair_device(device_client, headers)["device_token"]

    renewed = device_client.post("/device/v1/token/renew", headers={DEVICE_TOKEN_HEADER: token})
    assert renewed.status_code == 200, renewed.text
    new_token = renewed.json()["device_token"]
    assert new_token != token

    assert (
        device_client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: new_token}).status_code
        == 200
    )
    # The predecessor stays valid during the overlap so a lost response is harmless.
    assert (
        device_client.get("/device/v1/self", headers={DEVICE_TOKEN_HEADER: token}).status_code
        == 200
    )


def test_the_ip_scoped_recovery_list_is_gone(device_client: TestClient) -> None:
    """It had no user predicate - a pairing has no owner before consent.

    Behind a proxy the platform does not allowlist, its only filter matched every
    row, so an authenticated user saw every other user's live pairing requests
    and their ``pairing_id``.
    """
    headers = _register(device_client)
    _start_pairing(device_client, "Recoverable laptop")

    response = device_client.get("/devices/pairings", headers=headers)

    # 405 rather than 404, because on the real app the path falls through to
    # DELETE /devices/{device_id} with device_id="pairings". Which status it is
    # is an accident of routing; what this test holds is that no GET anywhere
    # hands an authenticated user a list of pairings it does not own.
    assert response.status_code in {404, 405}
    assert "pairing_id" not in response.text


def test_finished_pairings_are_deleted(device_client: TestClient) -> None:
    """Unauthenticated writes with no cleanup grow forever. Also proves the grant."""
    headers = _register(device_client)
    approved = _pair_device(device_client, headers, "Sweepable laptop")
    consumed_id = approved["_pairing"]["pairing_id"]

    with sync_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE helper_pairings SET created_at = now() - interval '90 days', "
                "expires_at = now() - interval '90 days' WHERE id = :pairing_id"
            ),
            {"pairing_id": consumed_id},
        )

    # The sweep runs from the one endpoint that inserts into this table.
    _start_pairing(device_client, "Fresh laptop")

    with sync_engine.begin() as connection:
        remaining = connection.execute(
            text("SELECT count(*) FROM helper_pairings WHERE id = :pairing_id"),
            {"pairing_id": consumed_id},
        ).scalar_one()
        devices = connection.execute(
            text("SELECT count(*) FROM helper_devices WHERE id = :device_id"),
            {"device_id": approved["device_id"]},
        ).scalar_one()

    assert remaining == 0
    # Sweeping a pairing must never cascade into the device it created.
    assert devices == 1


def test_device_routes_are_served_by_the_real_app(client: TestClient) -> None:
    """The application the deployment builds, not one assembled by this module."""
    paths = set(client.app.openapi()["paths"])
    assert {
        "/device/v1/pairings",
        "/device/v1/pairings/token",
        "/devices/pairings/lookup",
        "/devices/pairings/{pairing_id}/approve",
        "/devices",
        "/device/v1/self",
    } <= paths

    # The session client, not a fresh TestClient over a fresh create_app(): a second
    # client brings a second event loop, and the asyncpg pool belongs to the first.
    response = client.post(
        "/device/v1/pairings",
        json={
            "device_name": "Mounted laptop",
            "platform": "darwin-arm64",
            "helper_version": "0.2.0",
            "capabilities": {},
        },
    )
    assert response.status_code == 201, response.text
