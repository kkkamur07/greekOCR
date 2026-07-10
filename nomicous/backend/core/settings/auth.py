"""JWT and browser-session authentication settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class AuthSettings(BaseSettings):
    model_config = env_settings_config()

    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=15, alias="JWT_EXPIRE_MINUTES")
    session_expire_days: int = Field(default=14, alias="AUTH_SESSION_EXPIRE_DAYS", ge=1, le=90)
    session_cookie_name: str = Field(
        default="__Host-greekocr-session", alias="AUTH_SESSION_COOKIE_NAME", min_length=1
    )
    csrf_cookie_name: str = Field(
        default="greekocr-csrf", alias="AUTH_CSRF_COOKIE_NAME", min_length=1
    )
    csrf_cookie_domain: str | None = Field(default=None, alias="AUTH_CSRF_COOKIE_DOMAIN")
    cookie_same_site: Literal["lax", "strict"] = Field(default="lax", alias="AUTH_COOKIE_SAME_SITE")
    auth_rate_limit_requests: int = Field(default=10, alias="AUTH_RATE_LIMIT_REQUESTS")
    auth_rate_limit_window_seconds: int = Field(
        default=60,
        alias="AUTH_RATE_LIMIT_WINDOW_SECONDS",
    )

    @model_validator(mode="after")
    def _validate_secret(self) -> "AuthSettings":
        normalized = self.jwt_secret.strip().casefold()
        if (
            not normalized
            or normalized in {"change-me", "change-me-in-production", "replace-me"}
            or normalized.startswith("replace-with-")
        ):
            raise ValueError("JWT_SECRET must be set to a non-placeholder secret value")
        return self


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
