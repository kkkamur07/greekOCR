"""API application settings (CORS, media paths)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()
