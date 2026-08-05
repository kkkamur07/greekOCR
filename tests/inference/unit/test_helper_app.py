"""Inference helper HTTP surface (no ML weights required)."""

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inference.contracts.transcribe import TranscribeRunResponse
from inference.helper.app import create_helper_app
from inference.helper.settings import HELPER_VERSION, HelperSettings, get_helper_settings
from pydantic import ValidationError
from tests.fixtures.paths import TRANSCRIBE_LINE

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


def test_helper_info_identifies_the_service(helper_client: TestClient):
    """Discovery hinges on this: a foreign server on :8001 must not look like us."""
    response = helper_client.get("/inference/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "nomicous-inference-helper"
    assert body["version"] == HELPER_VERSION
    assert set(body) == {"service", "version", "models"}


def test_helper_info_lists_models_with_capabilities(helper_client: TestClient):
    response = helper_client.get("/inference/v1/info")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) >= 2
    model_ids = {item["registry_model_id"] for item in models}
    assert "greek-calamari-v1" not in model_ids

    syriac = next(item for item in models if item["registry_model_id"] == "syriac-calamari-v1")
    assert set(syriac) == {
        "registry_model_id",
        "task",
        "host_eligibility",
        "tags",
        "cached",
    }
    assert syriac["task"] == "transcribe"
    assert syriac["host_eligibility"] == "local"
    assert syriac["tags"] == ["stable"]
    assert isinstance(syriac["cached"], bool)

    segment = next(item for item in models if item["registry_model_id"] == "blla-segment")
    assert segment["task"] == "segment"


def test_helper_info_reports_uncached_weights_without_network(
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """An empty cache root must yield cached=false, never a Hub download."""
    monkeypatch.setenv("HF_CACHE_ROOT", str(tmp_path / "empty-cache"))

    response = helper_client.get("/inference/v1/info")
    assert response.status_code == 200
    assert all(item["cached"] is False for item in response.json()["models"])


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


def test_helper_always_dispatches_onnx_only(
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    received: dict[str, object] = {}

    def fake_run_model(**kwargs: object) -> TranscribeRunResponse:
        received.update(kwargs)
        return TranscribeRunResponse(text="", confidence=1.0, character_confidences=[])

    monkeypatch.setattr("inference.helper.routes.run.run_model", fake_run_model)
    response = helper_client.post(
        "/inference/v1/run",
        json={
            "task": "transcribe",
            "registry_model_id": "syriac-calamari-v1",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(TRANSCRIBE_LINE.read_bytes()).decode(),
        },
    )

    assert response.status_code == 200
    assert received["onnx_only"] is True


def test_helper_allows_only_configured_browser_origin(helper_client: TestClient):
    allowed_origin = "https://app.nomicous.com"
    preflight = helper_client.options(
        "/inference/v1/run",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == allowed_origin
    assert preflight.headers["access-control-allow-private-network"] == "true"
    assert "access-control-allow-credentials" not in preflight.headers

    local = helper_client.options(
        "/inference/v1/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in local.headers

    blocked = helper_client.options(
        "/inference/v1/run",
        headers={
            "Origin": "https://untrusted.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in blocked.headers


def test_helper_unhandled_errors_still_include_cors_headers(
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Bare Starlette 500s omit ACAO; UnhandledErrorMiddleware must prevent that."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("registry exploded")

    monkeypatch.setattr("inference.helper.routes.info.load_registry", boom)
    response = helper_client.get(
        "/inference/v1/info",
        headers={"Origin": "https://app.nomicous.com"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal helper error"
    assert response.headers["access-control-allow-origin"] == "https://app.nomicous.com"


def test_helper_mapped_run_errors_still_include_cors_headers(
    helper_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_type_error(**_kwargs: object) -> None:
        raise TypeError("unexpected runner failure")

    monkeypatch.setattr("inference.helper.routes.run.run_model", raise_type_error)
    response = helper_client.post(
        "/inference/v1/run",
        headers={"Origin": "https://app.nomicous.com"},
        json={
            "task": "transcribe",
            "registry_model_id": "syriac-calamari-v1",
            "registry_tag": "stable",
            "image_bytes": base64.b64encode(TRANSCRIBE_LINE.read_bytes()).decode(),
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal inference error"
    assert response.headers["access-control-allow-origin"] == "https://app.nomicous.com"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com"])
def test_helper_refuses_to_bind_off_loopback(host: str):
    """The helper is unauthenticated; loopback is the only thing containing it."""
    with pytest.raises(ValidationError, match="HELPER_HOST must be a loopback address"):
        HelperSettings(HELPER_HOST=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2"])
def test_helper_accepts_loopback_hosts(host: str):
    assert HelperSettings(HELPER_HOST=host).helper_host == host


def test_helper_rejects_non_loopback_binding_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HELPER_HOST", "0.0.0.0")
    get_helper_settings.cache_clear()

    with pytest.raises(ValidationError, match="HELPER_HOST must be a loopback address"):
        get_helper_settings()

    get_helper_settings.cache_clear()


def test_helper_has_no_auth_secret_settings():
    """Secure mode is gone: stray env vars must not resurrect it."""
    assert not hasattr(HelperSettings(), "helper_secure_mode")
    assert not hasattr(HelperSettings(), "helper_auth_secret")


def test_helper_ignores_secure_mode_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Leftover HELPER_SECURE_MODE from an old install must not lock users out."""
    monkeypatch.setenv("HELPER_SECURE_MODE", "true")
    monkeypatch.setenv("HELPER_AUTH_SECRET", "secure-helper-test-secret-0123456789")
    monkeypatch.setenv("INFERENCE_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(REPO_REGISTRY))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(tmp_path / "registry.yaml"))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(tmp_path / "registry.etag"))
    monkeypatch.setenv("HF_CACHE_ROOT", str(tmp_path / "hf-cache"))
    monkeypatch.delenv("HELPER_REGISTRY_URL", raising=False)
    get_helper_settings.cache_clear()
    client = TestClient(create_helper_app())

    assert client.get("/health").status_code == 200
    assert client.get("/inference/v1/info").status_code == 200


def test_helper_rejects_plain_http_registry_url_off_host(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("HELPER_REGISTRY_URL", raising=False)

    with pytest.raises(ValidationError, match="HELPER_REGISTRY_URL must use https"):
        HelperSettings(HELPER_REGISTRY_URL="http://api.test/inference/v1/registry")

    assert (
        HelperSettings(
            HELPER_REGISTRY_URL="http://127.0.0.1:8000/inference/v1/registry"
        ).helper_registry_url
        == "http://127.0.0.1:8000/inference/v1/registry"
    )
    assert (
        HelperSettings(
            HELPER_REGISTRY_URL="https://api.nomicous.com/inference/v1/registry"
        ).helper_registry_url
        == "https://api.nomicous.com/inference/v1/registry"
    )


def test_helper_no_longer_serves_replaced_routes(helper_client: TestClient):
    assert helper_client.get("/inference/v1/catalog").status_code == 404
    assert (
        helper_client.get("/inference/v1/cache-status?registry_model_id=blla-segment").status_code
        == 404
    )
