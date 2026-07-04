"""ML service integration settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class MLSettings(BaseSettings):
    model_config = env_settings_config()

    ml_service_url: str = Field(default="http://localhost:8001", alias="ML_SERVICE_URL")
    ml_webhook_secret: str | None = Field(default=None, alias="ML_WEBHOOK_SECRET")


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()
