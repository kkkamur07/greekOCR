"""Health and root endpoints — public HTTP interface."""


def test_root_returns_welcome_message(client):
    """GET / welcomes visitors to annote."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "Welcome to annote"


def test_health_returns_ok(client):
    """GET /health reports ok for the standalone annote API."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
