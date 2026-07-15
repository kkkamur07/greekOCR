"""ML API health endpoint."""

from fastapi.testclient import TestClient
from inference.api.app import create_app


# --- Health and root ---
# Tests liveness endpoints on a fresh app instance. Does not check database or model weights.


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "registry": "ok"}


def test_root_returns_service_message():
    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Nomicous ML inference API"}
