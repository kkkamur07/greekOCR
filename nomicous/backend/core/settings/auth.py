"""JWT and authentication settings."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class AuthSettings(BaseSettings):
    model_config = env_settings_config()

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")
    auth_rate_limit_requests: int = Field(default=10, alias="AUTH_RATE_LIMIT_REQUESTS")
    auth_rate_limit_window_seconds: int = Field(
        default=60,
        alias="AUTH_RATE_LIMIT_WINDOW_SECONDS",
    )

    @model_validator(mode="after")
    def _validate_secret(self) -> "AuthSettings":
        if not self.jwt_secret or self.jwt_secret == "change-me-in-production":
            raise ValueError("JWT_SECRET must be set to a secret value")
        return self


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
