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
    """The env-loading contract shared by every settings class.

    ``env_ignore_empty`` is what makes ``KNOB=`` mean "not set" rather than the
    empty string. Without it, an entry left blank in a .env file - the ordinary
    way to write down a knob you are not using - reaches pydantic as ``''`` and
    fails every field that is not a string. The shipped ``.env.example`` carried
    exactly that for ``JOB_WORKER_CLAIM_TEST_ONLY`` (a ``bool | None``), so the
    template four documents tell readers to copy raised ``ValidationError``
    inside ``create_app()`` and the API never came up.
    """
    return SettingsConfigDict(
        env_file=resolved_env_file(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
