"""ML model paths and defaults (populated in inference issues)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class ModelSettings(BaseSettings):
    model_config = env_settings_config()

    # Placeholder — issue 005 will wire Kraken/TrOCR artifact paths.
    default_segment_model: str | None = Field(default=None, alias="DEFAULT_SEGMENT_MODEL")
    default_transcribe_model: str | None = Field(default=None, alias="DEFAULT_TRANSCRIBE_MODEL")


@lru_cache
def get_model_settings() -> ModelSettings:
    return ModelSettings()
