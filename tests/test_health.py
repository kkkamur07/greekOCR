"""Platform health — public HTTP interface."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.app import create_app


@pytest.mark.asyncio
async def test_health_returns_ok_when_database_is_reachable():
    """GET /health reports ok when Postgres accepts a connection."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
