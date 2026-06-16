"""Root welcome endpoint."""


def test_root_welcome(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Kalamos API"
    assert "Kalamos" in body["message"]
    assert body["docs_url"] == "/docs"
    assert body["health_url"] == "/health"
