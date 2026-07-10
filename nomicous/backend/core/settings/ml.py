"""ML service integration settings."""

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._env import REPO_ROOT, env_settings_config

_PLACEHOLDER_SECRET_VALUES = {
    "change-me",
    "change-me-in-production",
    "replace-me",
    "replace-with-a-secret",
}


def _default_inference_registry_path() -> Path:
    app_root = Path(os.environ.get("NOMICOUS_APP_ROOT", REPO_ROOT.parent))
    return app_root / "inference" / "registry.yaml"


def _is_placeholder_secret(value: str | None) -> bool:
    normalized = (value or "").strip().casefold()
    return (
        not normalized
        or normalized in _PLACEHOLDER_SECRET_VALUES
        or normalized.startswith("replace-with-")
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
        raise ValueError(
            f"{name} must be an absolute http(s) URL without credentials, query, or fragment"
        )
    if environment.casefold() == "production" and parsed.scheme != "https":
        raise ValueError(f"{name} must use HTTPS in production")


class MLSettings(BaseSettings):
    model_config = env_settings_config()

    environment: str = Field(default="development", alias="ENVIRONMENT")
    cloud_inference_enabled: bool = Field(default=False, alias="CLOUD_INFERENCE_ENABLED")
    inference_url: str = Field(default="http://localhost:8001", alias="INFERENCE_URL")
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")
    inference_service_secret: str | None = Field(default=None, alias="INFERENCE_SERVICE_SECRET")
    inference_registry_path: Path = Field(
        default_factory=_default_inference_registry_path,
        alias="INFERENCE_REGISTRY_PATH",
    )

    @model_validator(mode="after")
    def _validate_production_runtime(self) -> "MLSettings":
        if (
            self.environment.casefold() != "production"
            or self.cloud_inference_enabled
            or self.inference_service_secret is not None
        ):
            _validate_service_url(
                value=self.inference_url,
                name="INFERENCE_URL",
                environment=self.environment,
            )
        if self.environment.casefold() != "production":
            return self

        invalid = [
            name
            for name, value in (
                ("INFERENCE_WEBHOOK_SECRET", self.inference_webhook_secret),
                ("INFERENCE_SERVICE_SECRET", self.inference_service_secret),
            )
            if value is not None and _is_placeholder_secret(value)
        ]
        if invalid:
            raise ValueError(
                f"{', '.join(invalid)} must be set to non-placeholder secrets in production"
            )
        return self

    def require_callback_receiver_configuration(self) -> None:
        """Fail closed when an API process accepts inference completion callbacks."""
        if self.environment.casefold() != "production":
            return
        if _is_placeholder_secret(self.inference_webhook_secret):
            raise ValueError(
                "INFERENCE_WEBHOOK_SECRET must be set to a non-placeholder secret in production"
            )

    def require_job_dispatcher_configuration(self) -> None:
        """Fail closed when a worker process submits jobs to inference."""
        if self.environment.casefold() != "production":
            return
        if _is_placeholder_secret(self.inference_service_secret):
            raise ValueError(
                "INFERENCE_SERVICE_SECRET must be set to a non-placeholder secret in production"
            )


@lru_cache
def get_inference_settings() -> MLSettings:
    return MLSettings()


get_ml_settings = get_inference_settings
