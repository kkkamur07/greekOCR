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
    get_auth_settings,
    get_infrastructure_settings,
    get_job_settings,
    get_ml_settings,
    get_storage_settings,
)
from backend.jobs.worker_main import validate_worker_settings
from backend.users.api.rate_limit import _real_ip


def _clear_platform_settings() -> None:
    for get_settings in (
        get_app_settings,
        get_auth_settings,
        get_infrastructure_settings,
        get_job_settings,
        get_ml_settings,
        get_storage_settings,
    ):
        get_settings.cache_clear()


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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://inference.example.com", None),
        ("http://localhost:8001", "must use HTTPS"),
        ("http://127.0.0.1:8001", "must use HTTPS"),
        ("http://inference-api:8001", "must use HTTPS"),
        ("http://api:8000/internal/inference/job-complete", "must use HTTPS"),
        ("http://inference.example.com", "must use HTTPS"),
    ],
)
def test_platform_production_inference_url_requires_https(url: str, expected: str | None) -> None:
    values = {
        "ENVIRONMENT": "production",
        "INFERENCE_URL": url,
        "INFERENCE_WEBHOOK_SECRET": "webhook-secret",
        "INFERENCE_SERVICE_SECRET": "service-secret",
    }
    if expected:
        with pytest.raises(ValidationError, match=expected):
            MLSettings(_env_file=None, **values)
    else:
        assert MLSettings(_env_file=None, **values).inference_url == url


def test_platform_local_inference_mode_needs_no_cloud_credentials() -> None:
    settings = MLSettings(ENVIRONMENT="production", _env_file=None)

    assert settings.cloud_inference_enabled is False


@pytest.mark.parametrize("secret", [None, "", "replace-me", "replace-with-a-secret"])
def test_platform_production_rejects_missing_or_placeholder_inference_secrets(
    secret: str | None,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="INFERENCE_WEBHOOK_SECRET"):
        MLSettings(
            ENVIRONMENT="production",
            INFERENCE_URL="https://inference.example.com",
            INFERENCE_WEBHOOK_SECRET=secret,
            INFERENCE_SERVICE_SECRET="service-secret",
            _env_file=None,
        ).require_callback_receiver_configuration()


@pytest.mark.parametrize("secret", ["replace-me", "replace-with-at-least-32-byte-secret"])
def test_platform_rejects_placeholder_jwt_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        AuthSettings(JWT_SECRET=secret, _env_file=None)


def test_platform_app_fails_fast_for_production_inference_configuration(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "valid-production-jwt-secret")
    monkeypatch.setenv("INFERENCE_URL", "https://inference.example.com")
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "replace-me")
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "service-secret")
    _clear_platform_settings()

    with pytest.raises(ValidationError, match="INFERENCE_WEBHOOK_SECRET"):
        create_app()


def test_platform_worker_validates_production_inference_configuration(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("INFERENCE_URL", "https://inference.example.com")
    monkeypatch.delenv("INFERENCE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("INFERENCE_SERVICE_SECRET", raising=False)
    _clear_platform_settings()

    with pytest.raises(ValueError, match="INFERENCE_SERVICE_SECRET"):
        validate_worker_settings()
