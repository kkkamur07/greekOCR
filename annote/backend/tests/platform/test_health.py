"""Platform health — public HTTP interface via FastAPI TestClient."""


def test_health_returns_ok_when_database_is_reachable(client):
    """GET /health reports ok when Postgres (kalamos) accepts a connection."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "database": "ok"}
