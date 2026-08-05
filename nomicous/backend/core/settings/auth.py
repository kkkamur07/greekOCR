"""JWT and browser-session authentication settings."""

import math
from collections import Counter
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config

_PLACEHOLDER_SECRET_VALUES = {"change-me", "change-me-in-production", "replace-me"}

#: A signing key shorter than the HMAC-SHA256 block-equivalent output is padded
#: by the algorithm, so it never contributes more than 256 bits regardless. 32
#: bytes is the smallest key that cannot be brute-forced faster than the digest.
MIN_JWT_SECRET_BYTES = 32

#: Estimated entropy floor, in bits, for the whole secret.
#:
#: The previous value, 128.0, came from the arithmetic that 32 hex characters
#: span 32 * log2(16) = 128 bits. That is the *ceiling* for a 32-character hex
#: string, reached only when all 16 symbols appear exactly twice - which a real
#: draw never does. ``secrets.token_hex(16)`` carries a genuine 128 bits but
#: scores ~115 on average, peaked at 126.5 over 200k samples and dipped to 89.
#: So the gate rejected every correctly generated 32-character secret - 200k out
#: of 200k - while waving through patterned ones such as ``"0123456789" * 4``,
#: which scored 133.
#:
#: 80 bits sits ~9 below the worst draw observed for the weakest secret worth
#: admitting, and far above any patterned string of the same length:
#: ``"password" * 4`` scores 22, ``"secret" * 6`` scores 14, ``"a" * 32`` scores
#: 0. It is also the conventional floor below which offline brute force stops
#: being a thought experiment.
MIN_JWT_SECRET_ENTROPY_BITS = 80.0


def _is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().casefold()
    return (
        not normalized
        or normalized in _PLACEHOLDER_SECRET_VALUES
        or normalized.startswith("replace-with-")
    )


def _shortest_period(value: str) -> int:
    """Length of the shortest block ``value`` is a repetition of.

    The longest-border table from Knuth-Morris-Pratt: ``"abcabcab"`` has border
    ``"abcab"`` and therefore period 3. A value with no internal repetition has a
    period equal to its own length, so this only ever shortens a string that
    literally repeats itself.
    """
    length = len(value)
    border = [0] * length
    matched = 0
    for index in range(1, length):
        while matched and value[index] != value[matched]:
            matched = border[matched - 1]
        if value[index] == value[matched]:
            matched += 1
        border[index] = matched
    return length - border[-1]


def estimated_entropy_bits(value: str) -> float:
    """Estimated entropy of ``value`` in bits.

    Two factors, multiplied: how evenly the value spreads over its own alphabet
    (Shannon entropy per character), and how much of it is not a repeat of what
    came before (its shortest period). The second factor is what a
    distribution-only estimate misses - character counts are blind to order, so
    ``"0123456789abcdef" * 2`` scores a flawless 4 bits/char and would be graded
    at 128 bits while carrying at most 64.

    Still an upper bound on real entropy - it cannot see that a value came out of
    a dictionary - but a hard ceiling for anything built from a small alphabet or
    a repeated block, which is what the production gate exists to reject.
    """
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    per_character = -sum((count / length) * math.log2(count / length) for count in counts.values())
    # A value that is `block * n` cannot carry more unpredictability than one
    # block does, however uniform its character counts happen to look.
    return per_character * _shortest_period(value)


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
        if estimated_entropy_bits(secret) < MIN_JWT_SECRET_ENTROPY_BITS:
            raise ValueError(
                "JWT_SECRET is too low-entropy for production; generate it with "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`'
            )
        return self


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
