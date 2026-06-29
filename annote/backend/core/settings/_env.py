"""Shared env file path for all settings classes."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

CORE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CORE_DIR.parents[1]
ENV_FILE = CORE_DIR / ".env"


def env_settings_config() -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )
