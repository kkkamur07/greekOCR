"""Root welcome endpoint."""


# --- Root metadata ---
# Tests welcome payload and version fields. Does not test authenticated routes.


def test_root_welcome(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Kalamos API"
    assert "Kalamos" in body["message"]
    assert body["version"] == "0.3.2"
    assert body["docs_url"] == "/docs"
    assert body["health_url"] == "/health"


# --- Security headers ---
# Tests baseline HTTP security headers on responses. Does not test CSP for the SPA.


def test_security_headers_are_present(client):
    response = client.get("/")

    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
