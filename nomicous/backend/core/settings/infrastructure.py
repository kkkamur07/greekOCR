"""Database and runtime environment settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class InfrastructureSettings(BaseSettings):
    model_config = env_settings_config()

    environment: str = Field(default="development", alias="ENVIRONMENT")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:dev@localhost:5433/kalamos",
        alias="DATABASE_URL",
    )
    sync_database_url: str = Field(
        default="postgresql://postgres:dev@localhost:5433/kalamos",
        alias="SYNC_DATABASE_URL",
    )
    migrator_database_url: str = Field(
        default="postgresql://postgres:dev@localhost:5433/kalamos",
        alias="MIGRATOR_DATABASE_URL",
    )
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=1800, alias="DB_POOL_RECYCLE")


@lru_cache
def get_infrastructure_settings() -> InfrastructureSettings:
    return InfrastructureSettings()
