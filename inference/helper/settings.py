"""Environment settings for the Inference helper sidecar."""

from __future__ import annotations

import os
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from inference.admission import AdmissionSettings

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_ROOT = Path.home() / ".nomicous" / "hf" / "cache"
DEFAULT_NOMICOUS_HOME = Path.home() / ".nomicous"
DEFAULT_BUNDLED_REGISTRY_PATH = INFERENCE_ROOT / "registry.yaml"
DEFAULT_CACHED_REGISTRY_PATH = DEFAULT_NOMICOUS_HOME / "registry.yaml"
DEFAULT_CACHED_REGISTRY_ETAG_PATH = DEFAULT_NOMICOUS_HOME / "registry.etag"


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_non_placeholder_secret(secret: str | None) -> bool:
    if secret is None or len(secret.strip()) < 32:
        return False
    normalized = secret.strip().lower()
    return not any(
        marker in normalized for marker in ("change", "example", "placeholder", "replace")
    )


class HelperSettings(AdmissionSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    helper_host: str = Field(default="127.0.0.1", alias="HELPER_HOST")
    helper_port: int = Field(default=8001, alias="HELPER_PORT")
    helper_secure_mode: bool = Field(default=False, alias="HELPER_SECURE_MODE")
    helper_auth_secret: str | None = Field(default=None, alias="HELPER_AUTH_SECRET")
    bundled_registry_path: Path = Field(
        default=DEFAULT_BUNDLED_REGISTRY_PATH,
        alias="HELPER_BUNDLED_REGISTRY_PATH",
    )
    cached_registry_path: Path = Field(
        default=DEFAULT_CACHED_REGISTRY_PATH,
        alias="HELPER_CACHED_REGISTRY_PATH",
    )
    cached_registry_etag_path: Path = Field(
        default=DEFAULT_CACHED_REGISTRY_ETAG_PATH,
        alias="HELPER_CACHED_REGISTRY_ETAG_PATH",
    )
    helper_registry_url: str | None = Field(default=None, alias="HELPER_REGISTRY_URL")
    inference_registry_path: Path = Field(
        default=DEFAULT_BUNDLED_REGISTRY_PATH,
        alias="INFERENCE_REGISTRY_PATH",
    )
    hf_cache_root: Path = Field(default=DEFAULT_HF_CACHE_ROOT, alias="HF_CACHE_ROOT")
    helper_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "https://app.nomicous.com",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        alias="HELPER_CORS_ORIGINS",
    )

    @field_validator("helper_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return value
        return []

    @field_validator("helper_cors_origins")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        if not origins:
            raise ValueError("HELPER_CORS_ORIGINS must contain at least one explicit origin")

        normalized_origins: list[str] = []
        for origin in origins:
            if not isinstance(origin, str) or origin == "*":
                raise ValueError("HELPER_CORS_ORIGINS must not contain wildcard origins")
            parsed = urlparse(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "HELPER_CORS_ORIGINS entries must be absolute http(s) origins without paths"
                )
            normalized = f"{parsed.scheme}://{parsed.netloc}"
            if normalized not in normalized_origins:
                normalized_origins.append(normalized)
        return normalized_origins

    @model_validator(mode="after")
    def validate_exposure(self) -> HelperSettings:
        secret_is_safe = _is_non_placeholder_secret(self.helper_auth_secret)
        if self.helper_secure_mode and not secret_is_safe:
            raise ValueError("HELPER_SECURE_MODE requires a non-placeholder HELPER_AUTH_SECRET")
        if not _is_loopback_host(self.helper_host) and not (
            self.helper_secure_mode and secret_is_safe
        ):
            raise ValueError(
                "HELPER_HOST must be loopback unless secure mode has a non-placeholder auth secret"
            )
        return self


@lru_cache
def get_helper_settings() -> HelperSettings:
    return HelperSettings()


def _resolve_registry_path(settings: HelperSettings) -> Path:
    if settings.helper_registry_url:
        from inference.helper.registry_sync import sync_registry_from_url

        return sync_registry_from_url(
            settings.helper_registry_url,
            cached_path=settings.cached_registry_path,
            etag_path=settings.cached_registry_etag_path,
            fallback_path=settings.bundled_registry_path,
        )
    if settings.inference_registry_path.exists():
        return settings.inference_registry_path
    return settings.bundled_registry_path


def apply_helper_environment() -> HelperSettings:
    """Set process env used by weight resolution and sync registry before serving."""
    settings = get_helper_settings()
    settings.hf_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_CACHE_ROOT", str(settings.hf_cache_root))

    registry_path = _resolve_registry_path(settings)
    os.environ["INFERENCE_REGISTRY_PATH"] = str(registry_path)
    get_helper_settings.cache_clear()
    refreshed = get_helper_settings()
    refreshed.hf_cache_root.mkdir(parents=True, exist_ok=True)
    return refreshed
