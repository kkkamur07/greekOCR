"""Regression tests for inference runtime configuration safeguards."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference.api.app import create_app
from inference.infrastructure.settings import InferenceSettings, get_inference_settings
from inference.jobs.worker import main


@pytest.fixture(autouse=True)
def clear_inference_settings_cache() -> None:
    get_inference_settings.cache_clear()
    yield
    get_inference_settings.cache_clear()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/internal/inference/job-complete", None),
        ("http://localhost:8000/internal/inference/job-complete", "must use HTTPS"),
        ("http://127.0.0.1:8000/internal/inference/job-complete", "must use HTTPS"),
        ("http://api:8000/internal/inference/job-complete", "must use HTTPS"),
        ("http://inference-api:8001/internal/inference/job-complete", "must use HTTPS"),
        ("http://api.example.com/internal/inference/job-complete", "must use HTTPS"),
    ],
)
def test_production_callback_url_requires_https(url: str, expected: str | None) -> None:
    values = {
        "ENVIRONMENT": "production",
        "INFERENCE_CALLBACK_URL": url,
        "INFERENCE_WEBHOOK_SECRET": "webhook-secret",
        "INFERENCE_SERVICE_SECRET": "service-secret",
    }
    if expected:
        with pytest.raises(ValidationError, match=expected):
            InferenceSettings(_env_file=None, **values)
    else:
        assert InferenceSettings(_env_file=None, **values).inference_callback_url == url


@pytest.mark.parametrize("secret", [None, "", "replace-me", "replace-with-a-secret"])
def test_production_worker_rejects_missing_or_placeholder_callback_secret(
    secret: str | None,
) -> None:
    with pytest.raises((ValidationError, ValueError), match="INFERENCE_WEBHOOK_SECRET"):
        InferenceSettings(
            ENVIRONMENT="production",
            INFERENCE_CALLBACK_URL="https://api.example.com/internal/inference/job-complete",
            INFERENCE_WEBHOOK_SECRET=secret,
            INFERENCE_SERVICE_SECRET="service-secret",
            _env_file=None,
        ).require_callback_configuration()


def test_production_requires_callback_url() -> None:
    with pytest.raises(ValueError, match="INFERENCE_CALLBACK_URL"):
        InferenceSettings(
            ENVIRONMENT="production",
            INFERENCE_WEBHOOK_SECRET="webhook-secret",
            INFERENCE_SERVICE_SECRET="service-secret",
            _env_file=None,
        ).require_callback_configuration()


def test_production_inference_api_requires_explicit_database_url() -> None:
    with pytest.raises(ValueError, match="INFERENCE_DATABASE_URL"):
        InferenceSettings(
            ENVIRONMENT="production",
            INFERENCE_SERVICE_SECRET="service-secret",
            _env_file=None,
        ).require_service_endpoint_configuration()


def test_inference_api_fails_fast_for_production_placeholder_secret(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "INFERENCE_CALLBACK_URL", "https://api.example.com/internal/inference/job-complete"
    )
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "webhook-secret")
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "replace-me")
    get_inference_settings.cache_clear()

    with pytest.raises(ValidationError, match="INFERENCE_SERVICE_SECRET"):
        create_app()


def test_inference_worker_validates_before_waiting_for_schema(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "INFERENCE_CALLBACK_URL", "https://api.example.com/internal/inference/job-complete"
    )
    monkeypatch.setenv("INFERENCE_WEBHOOK_SECRET", "replace-me")
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "service-secret")
    get_inference_settings.cache_clear()

    with pytest.raises(ValidationError, match="INFERENCE_WEBHOOK_SECRET"):
        main()
