"""Registry sync from hosted API into ~/.nomicous/registry.yaml."""

from pathlib import Path

import httpx
import pytest

from inference.helper.registry_sync import sync_registry_from_url
from inference.registry import load_registry

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


@pytest.fixture
def mock_registry_transport(monkeypatch):
    def install(handler):
        real_client = httpx.Client

        def client_factory(*args, **kwargs):
            return real_client(
                transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout")
            )

        monkeypatch.setattr("inference.helper.registry_sync.httpx.Client", client_factory)

    return install


def test_sync_registry_writes_validated_cache(tmp_path: Path, mock_registry_transport):
    cached = tmp_path / "registry.yaml"
    etag_file = tmp_path / "registry.etag"
    fallback = tmp_path / "bundled.yaml"
    fallback.write_text("models: {}\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/inference/v1/registry"
        return httpx.Response(
            200,
            text=VALID_REGISTRY,
            headers={"ETag": '"abc123"'},
        )

    mock_registry_transport(handler)

    resolved = sync_registry_from_url(
        "http://api.test/inference/v1/registry",
        cached_path=cached,
        etag_path=etag_file,
        fallback_path=fallback,
    )
    assert resolved == cached
    assert etag_file.read_text(encoding="utf-8") == '"abc123"'
    load_registry(cached)


def test_sync_registry_replays_weak_etag_verbatim(tmp_path: Path, mock_registry_transport):
    cached = tmp_path / "registry.yaml"
    cached.write_text(VALID_REGISTRY, encoding="utf-8")
    etag_file = tmp_path / "registry.etag"
    etag_file.write_text('W/"weak123"', encoding="utf-8")
    fallback = tmp_path / "bundled.yaml"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == 'W/"weak123"'
        return httpx.Response(304)

    mock_registry_transport(handler)

    resolved = sync_registry_from_url(
        "http://api.test/inference/v1/registry",
        cached_path=cached,
        etag_path=etag_file,
        fallback_path=fallback,
    )
    assert resolved == cached


def test_sync_registry_uses_cache_on_304(tmp_path: Path, mock_registry_transport):
    cached = tmp_path / "registry.yaml"
    cached.write_text(VALID_REGISTRY, encoding="utf-8")
    etag_file = tmp_path / "registry.etag"
    etag_file.write_text("abc123", encoding="utf-8")
    fallback = tmp_path / "bundled.yaml"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == '"abc123"'
        return httpx.Response(304)

    mock_registry_transport(handler)

    resolved = sync_registry_from_url(
        "http://api.test/inference/v1/registry",
        cached_path=cached,
        etag_path=etag_file,
        fallback_path=fallback,
    )
    assert resolved == cached


def test_sync_registry_falls_back_when_fetch_fails(tmp_path: Path, mock_registry_transport):
    cached = tmp_path / "registry.yaml"
    etag_file = tmp_path / "registry.etag"
    fallback = tmp_path / "bundled.yaml"
    fallback.write_text(VALID_REGISTRY, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    mock_registry_transport(handler)

    resolved = sync_registry_from_url(
        "http://api.test/inference/v1/registry",
        cached_path=cached,
        etag_path=etag_file,
        fallback_path=fallback,
    )
    assert resolved == fallback


def test_sync_registry_rejects_invalid_payload(tmp_path: Path, mock_registry_transport):
    cached = tmp_path / "registry.yaml"
    cached.write_text(VALID_REGISTRY, encoding="utf-8")
    etag_file = tmp_path / "registry.etag"
    fallback = tmp_path / "bundled.yaml"
    fallback.write_text(VALID_REGISTRY, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not: valid: yaml: [[[")

    mock_registry_transport(handler)

    resolved = sync_registry_from_url(
        "http://api.test/inference/v1/registry",
        cached_path=cached,
        etag_path=etag_file,
        fallback_path=fallback,
    )
    assert resolved == cached
    assert cached.read_text(encoding="utf-8") == VALID_REGISTRY
