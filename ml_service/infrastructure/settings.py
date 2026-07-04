"""Environment settings for the ML inference service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ML_ROOT = Path(__file__).resolve().parents[1]


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://postgres:dev@localhost:5433/kalamos",
        alias="DATABASE_URL",
    )
    ml_callback_url: str | None = Field(default=None, alias="ML_CALLBACK_URL")
    ml_webhook_secret: str | None = Field(default=None, alias="ML_WEBHOOK_SECRET")
    ml_registry_path: Path = Field(
        default=ML_ROOT / "registry.yaml",
        alias="ML_REGISTRY_PATH",
    )
    ml_weights_cache_dir: Path = Field(
        default=ML_ROOT / "weights" / "cache",
        alias="ML_WEIGHTS_CACHE_DIR",
    )
    worker_notify_channel: str = Field(
        default="ml_jobs",
        alias="ML_WORKER_NOTIFY_CHANNEL",
    )
    worker_running_job_timeout_seconds: float = Field(
        default=1800.0,
        alias="ML_WORKER_RUNNING_JOB_TIMEOUT_SECONDS",
    )


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()
