"""JWT and browser-session authentication settings."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
from zxcvbn import zxcvbn

from backend.core.settings._cache import settings_cache
from backend.core.settings._env import env_settings_config

_PLACEHOLDER_SECRET_VALUES = {"change-me", "change-me-in-production", "replace-me"}

#: A signing key shorter than the HMAC-SHA256 block-equivalent output is padded
#: by the algorithm, so it never contributes more than 256 bits regardless. 32
#: bytes is the smallest key that cannot be brute-forced faster than the digest.
MIN_JWT_SECRET_BYTES = 32

#: Floor on ``zxcvbn``'s log10 estimate of the guesses needed to reach the secret.
#:
#: Measured rather than derived. Over 5000 draws each, the weakest generator worth
#: admitting bottomed out at 27.45 (``secrets.token_hex(16)``); ``token_urlsafe(32)``
#: at 40.49. The strongest *memorable* strings of the same length top out far below
#: that - ``"correcthorsebatterystaple1234567"`` scores 16.9 and
#: ``"change-me-in-production-abcdefgh"`` 16.8. 22 sits in the middle of that gap,
#: with ~5 decades of margin before a legitimate secret is rejected, which matters:
#: a false rejection here is a failed production boot.
#:
#: Deliberately NOT zxcvbn's 0-4 ``score``. That saturates at 4 for anything past
#: ~10^10 guesses, which both of the memorable strings above clear - it is tuned for
#: "is this an acceptable human password", a much lower bar than a signing key.
MIN_JWT_SECRET_GUESSES_LOG10 = 22.0


def _is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().casefold()
    return (
        not normalized
        or normalized in _PLACEHOLDER_SECRET_VALUES
        or normalized.startswith("replace-with-")
    )


def secret_guesses_log10(value: str) -> float:
    """log10 of the guesses zxcvbn estimates an attacker needs to reach ``value``.

    zxcvbn is a real estimator rather than a character-frequency proxy: it matches
    against common passwords, names, dates, keyboard walks, l33t substitutions and
    repeats, so it sees the structure that a distribution-only measure cannot. That
    is the whole reason to use it here instead of counting characters - the failure
    mode this gate exists to catch is a human-chosen secret that *looks* varied.
    """
    if not value:
        return 0.0
    return zxcvbn(value)["guesses_log10"]


class AuthSettings(BaseSettings):
    model_config = env_settings_config()

    environment: str = Field(default="development", alias="ENVIRONMENT")
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
        # Placeholders are rejected everywhere: a shipped example value must never
        # sign a token, not even in development.
        if _is_placeholder_secret(self.jwt_secret):
            raise ValueError("JWT_SECRET must be set to a non-placeholder secret value")
        if self.environment.casefold() != "production":
            return self

        secret = self.jwt_secret.strip()
        if len(secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
            raise ValueError(
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes in production"
            )
        if secret_guesses_log10(secret) < MIN_JWT_SECRET_GUESSES_LOG10:
            raise ValueError(
                "JWT_SECRET is too guessable for production; generate it with "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`'
            )
        return self


@settings_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
