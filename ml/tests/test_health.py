"""ML API health endpoint."""

from fastapi.testclient import TestClient
from ml.api.app import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
