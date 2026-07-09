"""ML service integration settings."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import REPO_ROOT, env_settings_config


def _default_inference_registry_path() -> Path:
    app_root = Path(os.environ.get("NOMICOUS_APP_ROOT", REPO_ROOT.parent))
    return app_root / "inference" / "registry.yaml"


class MLSettings(BaseSettings):
    model_config = env_settings_config()

    inference_url: str = Field(default="http://localhost:8001", alias="INFERENCE_URL")
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")
    inference_service_secret: str | None = Field(default=None, alias="INFERENCE_SERVICE_SECRET")
    inference_registry_path: Path = Field(
        default_factory=_default_inference_registry_path,
        alias="INFERENCE_REGISTRY_PATH",
    )


@lru_cache
def get_inference_settings() -> MLSettings:
    return MLSettings()


get_ml_settings = get_inference_settings
