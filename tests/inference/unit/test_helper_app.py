"""Inference helper HTTP surface (no ML weights required)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings

REPO_REGISTRY = Path(__file__).resolve().parents[3] / "inference" / "registry.yaml"


@pytest.fixture
def helper_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """Isolate helper tests from ~/.nomicous cache and shell env."""
    monkeypatch.delenv("HELPER_REGISTRY_URL", raising=False)
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
            "image_bytes": "",
        },
    )
    assert response.status_code == 404
