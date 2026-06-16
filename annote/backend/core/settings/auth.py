"""JWT and authentication settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class AuthSettings(BaseSettings):
    model_config = env_settings_config()

    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60 * 24 * 7, alias="JWT_EXPIRE_MINUTES")


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
