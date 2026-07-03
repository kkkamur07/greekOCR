"""Unit tests for the annote ML HTTP client."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport

from backend.ml.infrastructure.ml_client import MlServiceClient
from ml.api.app import create_app


@pytest.fixture
def ml_transport(monkeypatch: pytest.MonkeyPatch) -> ASGITransport:
    monkeypatch.setenv("ML_FORCE_MOCK_RUNNER", "true")
    return ASGITransport(app=create_app())


@pytest.mark.asyncio
async def test_run_transcribe_uses_ml_service_contract(ml_transport: ASGITransport) -> None:
    client = MlServiceClient(base_url="http://ml.test", transport=ml_transport)

    result = await client.run_transcribe(
        registry_model_id="greek-calamariv1",
        registry_tag="stable",
        image_bytes=b"line-image",
        params={"line_index": 1},
    )

    assert result.text == "mock transcription 2"
    assert result.confidence == 0.82
    assert len(result.character_confidences) == len(result.text)


@pytest.mark.asyncio
async def test_run_serializes_image_bytes_as_base64(ml_transport: ASGITransport) -> None:
    import base64

    from httpx import ASGITransport as Transport
    from ml.api.app import create_app as create_ml_app

    captured: dict[str, object] = {}

    class CaptureTransport(Transport):
        async def handle_async_request(self, request):
            if request.method == "POST" and request.url.path == "/ml/v1/run":
                import json

                captured["json"] = json.loads(request.content.decode())
            return await super().handle_async_request(request)

    transport = CaptureTransport(app=create_ml_app())
    os.environ["ML_FORCE_MOCK_RUNNER"] = "true"
    client = MlServiceClient(base_url="http://ml.test", transport=transport)
    await client.run_transcribe(
        registry_model_id="greek-calamariv1",
        registry_tag="stable",
        image_bytes=b"abc",
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["task"] == "transcribe"
    assert payload["registry_model_id"] == "greek-calamariv1"
    assert payload["image_bytes"] == base64.b64encode(b"abc").decode()
