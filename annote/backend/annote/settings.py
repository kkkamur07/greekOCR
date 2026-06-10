"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data"


class PageLockSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANNOTE_PAGE_LOCK_", extra="ignore")

    prompt_at_full_pairing: bool = Field(
        default=True,
        description="Show lock prompt when pairing progress reaches 100%",
    )


class HistorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANNOTE_HISTORY_", extra="ignore")

    snapshot_interval_minutes: int = Field(default=5, ge=0)
    max_timed_snapshots: int = Field(default=5, ge=1)
    pairing_milestones: list[int] = Field(default=[50, 100])


class TranscriptionPdfSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANNOTE_TRANSCRIPTION_PDF_", extra="ignore")

    share_dir: str = Field(
        default="manuscripts/share",
        description="Directory under data_root for frozen share PDFs",
    )
    share_filename_pattern: str = Field(
        default="{stem}_transcription.pdf",
        description="Filename pattern for share PDFs; {stem} is replaced",
    )


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

    @field_validator("data_root", mode="before")
    @classmethod
    def resolve_data_root(cls, value: Path | str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        return path

    port: int = Field(default=5050, description="API bind port")
    reload: bool = Field(default=True, description="Uvicorn auto-reload (local dev)")
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated allowed CORS origins",
    )
    page_lock: PageLockSettings = Field(default_factory=PageLockSettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    transcription_pdf: TranscriptionPdfSettings = Field(default_factory=TranscriptionPdfSettings)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
