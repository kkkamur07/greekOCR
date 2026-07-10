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
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    get_helper_settings.cache_clear()
    return TestClient(create_helper_app())


def test_helper_health_returns_ok(helper_client: TestClient):
    response = helper_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_helper_catalog_lists_host_eligibility(helper_client: TestClient):
    response = helper_client.get("/inference/v1/catalog")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) >= 3
    calamari = next(item for item in models if item["registry_model_id"] == "greek-calamari-v1")
    assert calamari["host_eligibility"] == "local"
    assert calamari["task"] == "transcribe"


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
    client = TestClient(create_helper_app())

    assert client.get("/health").status_code == 401
    assert (
        client.get(
            "/health",
            headers={"X-Inference-Helper-Secret": "secure-helper-test-secret-0123456789"},
        ).status_code
        == 200
    )
