"""Application settings loaded from environment / backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="development", alias="ENVIRONMENT")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:dev@localhost:5433/greekocr",
        alias="DATABASE_URL",
    )
    sync_database_url: str = Field(
        default="postgresql://postgres:dev@localhost:5433/greekocr",
        alias="SYNC_DATABASE_URL",
    )
    jwt_secret: str = Field(default="change-me-in-production", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60 * 24 * 7, alias="JWT_EXPIRE_MINUTES")
    media_root: Path = Field(default=REPO_ROOT / "backend" / "media", alias="MEDIA_ROOT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
