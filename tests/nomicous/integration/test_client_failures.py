"""Client failure beacon endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.core.api.client_failures import clear_client_failure_rate_limit_state


def setup_function() -> None:
    clear_client_failure_rate_limit_state()


def test_client_failure_beacon_accepts_payload(client: TestClient):
    response = client.post(
        "/client-failures",
        json={
            "message": "Could not save Segment",
            "ref": "abc123",
            "path": "/projects/p1",
            "status": 500,
            "source": "toast",
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["ref"] == "abc123"


def test_client_failure_beacon_mints_ref_when_missing(client: TestClient):
    response = client.post(
        "/client-failures",
        json={"message": "Something went wrong"},
    )
    assert response.status_code == 202
    assert response.json()["ref"]


def test_client_failure_beacon_rejects_control_characters(client: TestClient):
    response = client.post(
        "/client-failures",
        json={"message": "bad\nmessage\x00"},
    )
    assert response.status_code == 422


def test_client_failure_beacon_rate_limits_by_ip(client: TestClient):
    for _ in range(30):
        ok = client.post("/client-failures", json={"message": "noise"})
        assert ok.status_code == 202, ok.text
    limited = client.post("/client-failures", json={"message": "too many"})
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After")


def test_client_failure_beacon_rate_limits_malformed_bodies(client: TestClient):
    """Throttle is a route dependency so 422s still consume the IP budget."""
    for _ in range(30):
        bad = client.post("/client-failures", json={"message": ""})
        assert bad.status_code == 422, bad.text
    limited = client.post("/client-failures", json={"message": "after spam"})
    assert limited.status_code == 429
