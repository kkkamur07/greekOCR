"""Platform health — public HTTP interface via FastAPI TestClient."""


# --- Health with database ---
# Tests /health when Postgres is reachable. Does not test inference service health.


def test_health_returns_ok_when_database_is_reachable(client):
    """GET /health reports ok when Postgres (kalamos) accepts a connection."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "database": "ok"}
