"""Environment settings for the ML inference service.

Nothing here reaches a database. The inference service reads a registry file and
runs models; the only job queue is the platform's (ADR 0003).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from inference.admission import AdmissionSettings

INFERENCE_ROOT = Path(__file__).resolve().parent
_PLACEHOLDER_SECRET_VALUES = {
    "change-me",
    "change-me-in-production",
    "replace-me",
    "replace-with-a-secret",
}


def _is_placeholder_secret(value: str | None) -> bool:
    normalized = (value or "").strip().casefold()
    return (
        not normalized
        or normalized in _PLACEHOLDER_SECRET_VALUES
        or normalized.startswith("replace-with-")
    )


class InferenceSettings(AdmissionSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")
    inference_registry_path: Path = Field(
        default=INFERENCE_ROOT / "registry.yaml",
        alias="INFERENCE_REGISTRY_PATH",
    )
    inference_service_secret: str | None = Field(default=None, alias="INFERENCE_SERVICE_SECRET")

    def require_service_endpoint_configuration(self) -> None:
        """Fail closed when the HTTP inference API cannot authenticate callers."""
        if self.environment.casefold() != "production":
            return
        if _is_placeholder_secret(self.inference_service_secret):
            raise ValueError(
                "INFERENCE_SERVICE_SECRET must be set to a non-placeholder value in production"
            )


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
