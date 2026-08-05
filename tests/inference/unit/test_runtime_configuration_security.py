"""Regression tests for inference runtime configuration safeguards."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference.api.app import create_app
from inference.settings import InferenceSettings, get_inference_settings


@pytest.fixture(autouse=True)
def clear_inference_settings_cache() -> None:
    get_inference_settings.cache_clear()
    yield
    get_inference_settings.cache_clear()


@pytest.mark.parametrize("secret", [None, "", "replace-me", "replace-with-a-secret"])
def test_production_service_endpoint_rejects_missing_or_placeholder_secret(
    secret: str | None,
) -> None:
    with pytest.raises(ValueError, match="INFERENCE_SERVICE_SECRET"):
        InferenceSettings(
            ENVIRONMENT="production",
            INFERENCE_SERVICE_SECRET=secret,
            _env_file=None,
        ).require_service_endpoint_configuration()


def test_development_service_endpoint_tolerates_missing_secret() -> None:
    InferenceSettings(
        ENVIRONMENT="development",
        _env_file=None,
    ).require_service_endpoint_configuration()


def test_inference_api_fails_fast_for_production_placeholder_secret(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("INFERENCE_SERVICE_SECRET", "replace-me")
    get_inference_settings.cache_clear()

    with pytest.raises(ValueError, match="INFERENCE_SERVICE_SECRET"):
        create_app()


def test_settings_carry_no_database_configuration() -> None:
    """ADR 0003: the inference service owns no queue, so it owns no database."""
    field_names = set(InferenceSettings.model_fields)

    assert not [name for name in field_names if "database" in name or name.startswith("db_")]
    assert "worker_notify_channel" not in field_names
