"""Environment settings for the ML inference service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

INFERENCE_ROOT = Path(__file__).resolve().parents[1]


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    inference_database_url: str = Field(
        default="postgresql://postgres:dev@localhost:5433/kalamos",
        alias="INFERENCE_DATABASE_URL",
    )
    inference_callback_url: str | None = Field(default=None, alias="INFERENCE_CALLBACK_URL")
    inference_webhook_secret: str | None = Field(default=None, alias="INFERENCE_WEBHOOK_SECRET")
    inference_registry_path: Path = Field(
        default=INFERENCE_ROOT / "registry.yaml",
        alias="INFERENCE_REGISTRY_PATH",
    )
    inference_weights_cache_dir: Path = Field(
        default=INFERENCE_ROOT / "weights" / "cache",
        alias="INFERENCE_WEIGHTS_CACHE_DIR",
    )
    worker_notify_channel: str = Field(
        default="inference_jobs",
        alias="INFERENCE_WORKER_NOTIFY_CHANNEL",
    )
    worker_running_job_timeout_seconds: float = Field(
        default=1800.0,
        alias="INFERENCE_WORKER_RUNNING_JOB_TIMEOUT_SECONDS",
    )
    inference_service_secret: str | None = Field(default=None, alias="INFERENCE_SERVICE_SECRET")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=1800, alias="DB_POOL_RECYCLE")


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
