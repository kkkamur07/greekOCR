"""Environment settings for the ML inference service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from inference.admission import AdmissionSettings

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER_SECRET_VALUES = {
    "change-me",
    "change-me-in-production",
    "replace-me",
    "replace-with-a-secret",
}


def _is_placeholder_secret(value: str | None) -> bool:
    normalized = (value or "").strip().casefold()
    return not normalized or normalized in _PLACEHOLDER_SECRET_VALUES or normalized.startswith(
        "replace-with-"
    )


def _validate_service_url(*, value: str, name: str, environment: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an absolute http(s) URL without credentials, query, or fragment")
    if environment.casefold() == "production" and parsed.scheme != "https":
        raise ValueError(f"{name} must use HTTPS in production")


class InferenceSettings(AdmissionSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    inference_database_url: str = Field(
        default="postgresql://postgres@localhost:5433/kalamos",
        alias="INFERENCE_DATABASE_URL",
    )
    inference_callback_url: str | None = Field(default=None, alias="INFERENCE_CALLBACK_URL")
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")
    inference_registry_path: Path = Field(
        default=INFERENCE_ROOT / "registry.yaml",
        alias="INFERENCE_REGISTRY_PATH",
    )
    inference_weights_cache_dir: Path = Field(
        default=INFERENCE_ROOT / "weights" / "cache",
        alias="INFERENCE_WEIGHTS_CACHE_DIR",
    )
    worker_notify_channel: str = Field(
        default="inference_jobs",
        alias="INFERENCE_WORKER_NOTIFY_CHANNEL",
    )
    worker_running_job_timeout_seconds: float = Field(
        default=1800.0,
        alias="INFERENCE_WORKER_RUNNING_JOB_TIMEOUT_SECONDS",
    )
    inference_service_secret: str | None = Field(default=None, alias="INFERENCE_SERVICE_SECRET")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=1800, alias="DB_POOL_RECYCLE")

    @model_validator(mode="after")
    def _validate_production_runtime(self) -> "InferenceSettings":
        if self.inference_callback_url:
            _validate_service_url(
                value=self.inference_callback_url,
                name="INFERENCE_CALLBACK_URL",
                environment=self.environment,
            )
        if self.environment.casefold() != "production":
            return self

        invalid = [
            name
            for name, value in (
                ("INFERENCE_CALLBACK_URL", self.inference_callback_url),
                ("INFERENCE_WEBHOOK_SECRET", self.inference_webhook_secret),
                ("INFERENCE_SERVICE_SECRET", self.inference_service_secret),
            )
            if _is_placeholder_secret(value)
        ]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be set to non-placeholder values in production")
        return self


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
