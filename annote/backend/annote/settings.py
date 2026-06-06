"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_ROOT = _BACKEND_DIR.parent / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANNOTE_",
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Field(
        default=_DEFAULT_DATA_ROOT,
        description="Root directory for annote filesystem data",
    )
    host: str = Field(default="127.0.0.1", description="API bind host")
    port: int = Field(default=5050, description="API bind port")
    reload: bool = Field(default=True, description="Uvicorn auto-reload (local dev)")
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated allowed CORS origins",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
