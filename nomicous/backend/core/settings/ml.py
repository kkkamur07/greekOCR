"""ML service integration settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class MLSettings(BaseSettings):
    model_config = env_settings_config()

    inference_url: str = Field(default="http://localhost:8001", alias="INFERENCE_URL")
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")


@lru_cache
def get_inference_settings() -> MLSettings:
    return MLSettings()


get_ml_settings = get_inference_settings
