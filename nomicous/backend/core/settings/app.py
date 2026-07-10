"""API application settings (CORS, media paths)."""

from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from backend.core.settings._env import REPO_ROOT, env_settings_config


class AppSettings(BaseSettings):
    model_config = env_settings_config()

    media_root: Path = Field(default=REPO_ROOT / "backend" / "media", alias="MEDIA_ROOT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )
    behind_proxy: bool = Field(default=False, alias="BEHIND_PROXY")
    forwarded_allow_ips: str | None = Field(default=None, alias="FORWARDED_ALLOW_IPS")

    @field_validator("forwarded_allow_ips")
    @classmethod
    def _validate_forwarded_allow_ips(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None

        entries = [entry.strip() for entry in value.split(",") if entry.strip()]
        if not entries:
            return None
        if "*" in entries:
            raise ValueError(
                "FORWARDED_ALLOW_IPS must contain explicit proxy IPs or CIDRs, never '*'"
            )

        networks: list[str] = []
        for entry in entries:
            try:
                networks.append(str(ip_network(entry, strict=False)))
            except ValueError as exc:
                raise ValueError(
                    "FORWARDED_ALLOW_IPS must contain only explicit proxy IPs or CIDRs"
                ) from exc
        return ",".join(dict.fromkeys(networks))

    @model_validator(mode="after")
    def _require_proxy_allowlist(self) -> "AppSettings":
        if self.behind_proxy and not self.forwarded_allow_ips:
            raise ValueError(
                "BEHIND_PROXY=true requires FORWARDED_ALLOW_IPS with explicit proxy IPs or CIDRs"
            )
        return self

    @field_validator("cors_origins")
    @classmethod
    def _validate_cors_origins(cls, value: str) -> str:
        origins = [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        if not origins:
            raise ValueError("CORS_ORIGINS must include at least one explicit origin")
        for origin in origins:
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
                or "*" in origin
            ):
                raise ValueError("CORS_ORIGINS must contain explicit http(s) origins only")
        return ",".join(dict.fromkeys(origins))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()
