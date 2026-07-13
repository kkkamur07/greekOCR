"""Client failure beacon endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


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
