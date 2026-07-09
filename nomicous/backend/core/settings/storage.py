"""Object storage settings (local filesystem or Supabase Storage)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config

StorageBackend = Literal["local", "supabase"]


class StorageSettings(BaseSettings):
    model_config = env_settings_config()

    storage_backend: StorageBackend = Field(default="local", alias="STORAGE_BACKEND")
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(
        default=None, alias="SUPABASE_SERVICE_ROLE_KEY"
    )
    supabase_storage_bucket: str = Field(
        default="document-media", alias="SUPABASE_STORAGE_BUCKET"
    )
    media_webp_lossless: bool = Field(default=True, alias="MEDIA_WEBP_LOSSLESS")
    media_webp_quality: int = Field(default=95, alias="MEDIA_WEBP_QUALITY")


@lru_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()
