"""ML API health endpoint."""

from fastapi.testclient import TestClient
from ml_service.api.app import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_service_message():
    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Nomicous ML inference API"}
