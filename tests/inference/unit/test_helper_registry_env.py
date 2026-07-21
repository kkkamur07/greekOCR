"""Helper environment applies synced registry path."""

from pathlib import Path

import httpx

from inference.helper.settings import apply_helper_environment, get_helper_settings

VALID_REGISTRY = """\
models:
  demo-calamari-v1:
    task: transcribe
    architecture: calamari
    device: cpu
    host_eligibility: local
    versions:
      stable:
        weights_source: hf://example/demo@stable
"""


def test_apply_helper_environment_uses_synced_registry(tmp_path: Path, monkeypatch):
    cached = tmp_path / "registry.yaml"
    etag_file = tmp_path / "registry.etag"
    bundled = tmp_path / "bundled.yaml"
    bundled.write_text(VALID_REGISTRY, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=VALID_REGISTRY, headers={"ETag": '"remote"'})

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setattr("inference.helper.registry_sync.httpx.Client", client_factory)
    monkeypatch.setenv("HELPER_REGISTRY_URL", "https://api.test/inference/v1/registry")
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_PATH", str(cached))
    monkeypatch.setenv("HELPER_CACHED_REGISTRY_ETAG_PATH", str(etag_file))
    monkeypatch.setenv("HELPER_BUNDLED_REGISTRY_PATH", str(bundled))
    get_helper_settings.cache_clear()

    settings = apply_helper_environment()
    assert settings.inference_registry_path == cached
    get_helper_settings.cache_clear()
