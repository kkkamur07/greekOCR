"""OpenAPI surface — public HTTP interface."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.app import create_app


@pytest.mark.asyncio
async def test_openapi_docs_are_available():
    """GET /docs returns the interactive API explorer."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/docs")

    assert response.status_code == 200
