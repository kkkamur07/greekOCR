"""Regression tests for production runtime configuration safeguards."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from backend.core.app import create_app
from backend.core.settings import (
    AppSettings,
    AuthSettings,
    MLSettings,
    get_app_settings,
    reset_settings_caches,
)
from backend.users.api.rate_limit import _real_ip


def _clear_platform_settings() -> None:
    # Enumerating the accessors by hand is what let this list drift - it was
    # missing the device accessor. The registry cannot go stale: an accessor is
    # enrolled by the decorator that caches it.
    reset_settings_caches()


@pytest.fixture(autouse=True)
def clear_platform_settings_cache() -> None:
    _clear_platform_settings()
    yield
    _clear_platform_settings()


def _request(*, peer: str, forwarded_for: str | None = None) -> Request:
    headers = [(b"host", b"testserver")]
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "headers": headers,
            "client": (peer, 12345),
            "scheme": "http",
            "method": "POST",
            "path": "/auth/login",
            "query_string": b"",
            "server": ("testserver", 80),
        }
    )


@pytest.mark.parametrize("allowlist", ["*", "testclient", "192.0.2.1, *"])
def test_proxy_allowlist_rejects_wildcards_and_hostnames(allowlist: str) -> None:
    with pytest.raises(ValidationError, match="FORWARDED_ALLOW_IPS"):
        AppSettings(BEHIND_PROXY=True, FORWARDED_ALLOW_IPS=allowlist, _env_file=None)


def test_proxy_allowlist_normalizes_networks() -> None:
    settings = AppSettings(
        BEHIND_PROXY=True,
        FORWARDED_ALLOW_IPS="10.0.0.7, 2001:db8::1",
        _env_file=None,
    )

    assert settings.forwarded_allow_ips == "10.0.0.7/32,2001:db8::1/128"


def test_rate_limit_uses_forwarded_client_only_from_trusted_proxy(monkeypatch) -> None:
    monkeypatch.setenv("BEHIND_PROXY", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    get_app_settings.cache_clear()

    assert _real_ip(_request(peer="10.1.2.3", forwarded_for="203.0.113.10, 10.2.3.4")) == (
        "203.0.113.10"
    )
    assert _real_ip(_request(peer="198.51.100.9", forwarded_for="203.0.113.10")) == "198.51.100.9"


def test_rate_limit_ignores_malformed_forwarded_client(monkeypatch) -> None:
    monkeypatch.setenv("BEHIND_PROXY", "true")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    get_app_settings.cache_clear()

    assert _real_ip(_request(peer="10.1.2.3", forwarded_for="not-an-ip, 10.2.3.4")) == "10.1.2.3"


def test_platform_holds_no_outbound_inference_credentials() -> None:
    """ADR 0003: nothing on the platform calls the inference service any more."""
    field_names = set(MLSettings.model_fields)

    assert "inference_url" not in field_names
    assert "inference_service_secret" not in field_names


def test_platform_local_inference_mode_needs_no_cloud_credentials(monkeypatch) -> None:
    # CI injects inference secrets for other jobs; isolate this local-mode check.
    monkeypatch.delenv("INFERENCE_WEBHOOK_SECRET", raising=False)
    settings = MLSettings(ENVIRONMENT="production", _env_file=None)

    assert settings.cloud_inference_enabled is False
    assert settings.inference_webhook_secret is None


@pytest.mark.parametrize("secret", [None, "", "replace-me", "replace-with-a-secret"])
def test_platform_production_rejects_missing_or_placeholder_inference_secrets(
    secret: str | None,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="INFERENCE_WEBHOOK_SECRET"):
        MLSettings(
            ENVIRONMENT="production",
            INFERENCE_WEBHOOK_SECRET=secret,
            _env_file=None,
        ).require_callback_receiver_configuration()


@pytest.mark.parametrize("secret", ["replace-me", "replace-with-at-least-32-byte-secret"])
def test_platform_rejects_placeholder_jwt_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        AuthSettings(JWT_SECRET=secret, _env_file=None)


def test_platform_app_fails_fast_for_production_inference_configuration(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    # Must clear the production JWT_SECRET floor (>=32 bytes, high entropy) so
    # that this test still fails on the inference secret it is actually about.
    monkeypatch.setenv("JWT_SECRET", "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV")
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "replace-me")
    _clear_platform_settings()

    with pytest.raises(ValidationError, match="INFERENCE_WEBHOOK_SECRET"):
        create_app()
