"""HTTP error mapping parity between the hosted /run and the helper /run."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inference.api.app import create_app
from inference.helper.app import create_helper_app
from inference.helper.settings import get_helper_settings
from inference.settings import get_inference_settings
from PIL import Image
from inference.hub.artifacts import ArtifactIntegrityError

REPO_REGISTRY = Path(__file__).resolve().parents[3] / "inference" / "registry.yaml"


def _run_payload() -> dict:
    output = BytesIO()
    Image.new("L", (2, 2)).save(output, format="PNG")
    return {
        "task": "segment",
        "registry_model_id": "blla-segment",
        "registry_tag": "stable",
        "image_bytes": base64.b64encode(output.getvalue()).decode(),
    }


EXPECTED_STATUS_BY_ERROR = [
    (KeyError("unknown model"), 404),
    (FileNotFoundError("weights missing"), 503),
    (ArtifactIntegrityError("artifact SHA-256 mismatch"), 503),
    (ValueError("bad request payload"), 422),
    (RuntimeError("model artifact failed to load"), 503),
    (TypeError("unexpected runner failure"), 500),
]


@pytest.fixture
def hosted_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "run-mapping-test-secret")
    get_inference_settings.cache_clear()
    return TestClient(
        create_app(),
        headers={"X-Inference-Service-Secret": "run-mapping-test-secret"},
    )


@pytest.fixture
def helper_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.delenv("HELPER_REGISTRY_URL", raising=False)
    monkeypatch.delenv("HELPER_SECURE_MODE", raising=False)
    monkeypatch.delenv("HELPER_AUTH_SECRET", raising=False)
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    get_helper_settings.cache_clear()
    return TestClient(create_helper_app())


@pytest.mark.parametrize(("error", "expected_status"), EXPECTED_STATUS_BY_ERROR)
def test_hosted_run_maps_runner_errors_like_the_helper(
    hosted_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    def raise_error(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("inference.api.run.run_model", raise_error)
    response = hosted_client.post("/inference/v1/run", json=_run_payload())
    assert response.status_code == expected_status


@pytest.mark.parametrize(("error", "expected_status"), EXPECTED_STATUS_BY_ERROR)
def test_helper_run_maps_runner_errors(
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    def raise_error(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("inference.helper.routes.run.run_model", raise_error)
    response = helper_client.post("/inference/v1/run", json=_run_payload())
    assert response.status_code == expected_status


def test_integrity_failures_are_service_errors_not_client_errors(
    hosted_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHA-256 mismatch subclasses ValueError but must never surface as 422."""

    def raise_integrity_error(**_kwargs: object) -> None:
        raise ArtifactIntegrityError("artifact SHA-256 mismatch for weights")

    monkeypatch.setattr("inference.api.run.run_model", raise_integrity_error)
    response = hosted_client.post("/inference/v1/run", json=_run_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "Model weights failed integrity verification"


def test_helper_and_hosted_share_catch_all_internal_detail(
    hosted_client: TestClient,
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_type_error(**_kwargs: object) -> None:
        raise TypeError("unexpected runner failure")

    monkeypatch.setattr("inference.api.run.run_model", raise_type_error)
    monkeypatch.setattr("inference.helper.routes.run.run_model", raise_type_error)

    hosted = hosted_client.post("/inference/v1/run", json=_run_payload())
    helper = helper_client.post("/inference/v1/run", json=_run_payload())
    assert hosted.status_code == 500
    assert helper.status_code == 500
    assert hosted.json()["detail"] == helper.json()["detail"] == "Internal inference error"
