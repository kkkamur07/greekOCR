"""Inference helper HTTP surface (no ML weights required)."""

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inference.helper.app import create_helper_app
from inference.helper.settings import HelperSettings, get_helper_settings
from pydantic import ValidationError

from tests.fixtures.paths import TRANSCRIBE_LINE

REPO_REGISTRY = Path(__file__).resolve().parents[3] / "inference" / "registry.yaml"


@pytest.fixture
def helper_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Isolate helper tests from ~/.nomicous cache and shell env."""
    monkeypatch.delenv("HELPER_REGISTRY_URL", raising=False)
    monkeypatch.delenv("HELPER_SECURE_MODE", raising=False)
    monkeypatch.delenv("HELPER_AUTH_SECRET", raising=False)
    monkeypatch.delenv("HELPER_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    get_helper_settings.cache_clear()
    return TestClient(create_helper_app(prefetch_weights=False))


def test_helper_health_returns_ok(helper_client: TestClient):
    response = helper_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_helper_catalog_lists_host_eligibility(helper_client: TestClient):
    response = helper_client.get("/inference/v1/catalog")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) >= 2
    model_ids = {item["registry_model_id"] for item in models}
    assert "greek-calamari-v1" not in model_ids
    syriac = next(item for item in models if item["registry_model_id"] == "syriac-calamari-v1")
    assert syriac["host_eligibility"] == "local"
    assert syriac["task"] == "transcribe"


def test_helper_run_requires_no_service_secret_for_unknown_model(helper_client: TestClient):
    response = helper_client.post(
        "/inference/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "missing-model",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(TRANSCRIBE_LINE.read_bytes()).decode(),
        },
    )
    assert response.status_code == 404


def test_helper_allows_only_configured_browser_origin(helper_client: TestClient):
    allowed_origin = "https://app.nomicous.com"
    preflight = helper_client.options(
        "/inference/v1/run",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == allowed_origin
    assert "access-control-allow-credentials" not in preflight.headers

    blocked = helper_client.options(
        "/inference/v1/run",
        headers={
            "Origin": "https://untrusted.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in blocked.headers


@pytest.mark.parametrize(
    "origins",
    [
        "*",
        "https://app.nomicous.com/path",
        "https://user:password@app.nomicous.com",
        "ftp://app.nomicous.com",
    ],
)
def test_helper_rejects_unsafe_cors_origins(origins: str):
    with pytest.raises(ValidationError, match="HELPER_CORS_ORIGINS"):
        HelperSettings(HELPER_CORS_ORIGINS=origins)


def test_helper_rejects_non_loopback_binding_without_secure_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HELPER_HOST", "0.0.0.0")
    monkeypatch.delenv("HELPER_SECURE_MODE", raising=False)
    monkeypatch.delenv("HELPER_AUTH_SECRET", raising=False)
    get_helper_settings.cache_clear()

    with pytest.raises(ValidationError, match="HELPER_HOST must be loopback"):
        get_helper_settings()

    assert (
        HelperSettings(
            HELPER_HOST="0.0.0.0",
            HELPER_SECURE_MODE=True,
            HELPER_AUTH_SECRET="secure-helper-test-secret-0123456789",
        ).helper_host
        == "0.0.0.0"
    )


def test_helper_secure_mode_requires_authentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HELPER_SECURE_MODE", "true")
    monkeypatch.setenv("HELPER_AUTH_SECRET", "secure-helper-test-secret-0123456789")
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    monkeypatch.setenv("HF_CACHE_ROOT", str(tmp_path / "hf-cache"))
    get_helper_settings.cache_clear()
    client = TestClient(create_helper_app(prefetch_weights=False))

    assert client.get("/health").status_code == 401
    assert (
        client.get(
            "/health",
            headers={"X-Inference-Helper-Secret": "secure-helper-test-secret-0123456789"},
        ).status_code
        == 200
    )
