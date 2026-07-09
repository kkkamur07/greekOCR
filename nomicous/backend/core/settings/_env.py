"""Shared env file path for all settings classes."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

CORE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CORE_DIR.parents[1]
ENV_FILE = CORE_DIR / ".env"
SUPABASE_ENV_FILE = CORE_DIR / ".env.supabase"


def resolved_env_file() -> Path:
    """Prefer backend/core/.env; fall back to .env.supabase when .env is absent."""
    if ENV_FILE.is_file():
        return ENV_FILE
    if SUPABASE_ENV_FILE.is_file():
        return SUPABASE_ENV_FILE
    return ENV_FILE


def env_settings_config() -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=resolved_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
