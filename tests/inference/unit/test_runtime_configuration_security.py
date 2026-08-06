"""Regression tests for inference runtime configuration safeguards."""

from __future__ import annotations

import pytest

from inference.settings import InferenceSettings, get_inference_settings


@pytest.fixture(autouse=True)
def clear_inference_settings_cache() -> None:
    get_inference_settings.cache_clear()
    yield
    get_inference_settings.cache_clear()


def test_settings_carry_no_database_configuration() -> None:
    """ADR 0003: the inference service owns no queue, so it owns no database."""
    field_names = set(InferenceSettings.model_fields)

    assert not [name for name in field_names if "database" in name or name.startswith("db_")]
    assert "worker_notify_channel" not in field_names


def test_settings_carry_no_listening_service_configuration() -> None:
    """ADR 0002: nothing here serves HTTP, so nothing here configures a server.

    The service secret and its production placeholder check existed only to
    authenticate callers of ``POST /inference/v1/run``. With that front door
    gone, a surviving knob would be the first foothold for growing it back.
    """
    field_names = set(InferenceSettings.model_fields)

    assert "inference_service_secret" not in field_names
    assert not [name for name in field_names if "host" in name or "port" in name]
    assert not hasattr(InferenceSettings, "require_service_endpoint_configuration")
