"""Background job worker settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from backend.core.settings._env import env_settings_config


class JobSettings(BaseSettings):
    model_config = env_settings_config()

    enable_test_job_routes: bool = Field(default=False, alias="ENABLE_TEST_JOB_ROUTES")
    job_worker_enabled: bool = Field(default=True, alias="JOB_WORKER_ENABLED")
    job_poll_interval_seconds: float = Field(default=0.25, alias="JOB_POLL_INTERVAL_SECONDS")
    job_poll_max_interval_seconds: float = Field(default=2.0, alias="JOB_POLL_MAX_INTERVAL_SECONDS")
    job_worker_claim_test_only: bool | None = Field(
        default=None,
        alias="JOB_WORKER_CLAIM_TEST_ONLY",
        description="True=only test payloads; False=exclude test; None=claim any pending job",
    )
    transcribe_adapter: str = Field(default="mock:transcribe", alias="TRANSCRIBE_ADAPTER")
    segment_adapter: str = Field(default="kraken", alias="SEGMENT_ADAPTER")


@lru_cache
def get_job_settings() -> JobSettings:
    return JobSettings()
