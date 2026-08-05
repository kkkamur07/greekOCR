"""Guards on the destructive platform scripts (schema reset, dev corpus seed).

Both guards were bypassable from ambient shell state, so the regressions here are
about *provenance*: the reset must trust only the env file it names, and the seed
must go through the shared development guard instead of asserting its own
environment.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from backend.core.settings import get_infrastructure_settings
from tests.fixtures.paths import REPO_ROOT

PLATFORM_SCRIPTS = REPO_ROOT / "scripts" / "platform"
RESET_SCRIPT = PLATFORM_SCRIPTS / "reset_supabase_nonprod.sh"
SEED_SCRIPT = PLATFORM_SCRIPTS / "seed_dev_annotated_data.py"


@pytest.fixture(scope="module")
def seed_module() -> Iterator[ModuleType]:
    """Import the standalone seed script; it resolves `_bootstrap` as a sibling."""
    sys.path.insert(0, str(PLATFORM_SCRIPTS))
    try:
        yield importlib.import_module("seed_dev_annotated_data")
    finally:
        sys.path.remove(str(PLATFORM_SCRIPTS))


def test_seed_refuses_to_run_outside_development(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_infrastructure_settings.cache_clear()
    try:
        # Must raise before the data-root SystemExit: the guard is the first thing
        # run_seed does, so no filesystem or database work happens first.
        with pytest.raises(RuntimeError, match="not 'development'"):
            asyncio.run(seed_module.run_seed(force=False, import_history=False))
    finally:
        get_infrastructure_settings.cache_clear()


def test_seed_never_falls_back_to_a_committed_password(
    seed_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEV_USER_PASSWORD", raising=False)
    first, first_generated = seed_module._resolve_dev_password()
    second, second_generated = seed_module._resolve_dev_password()
    assert first_generated and second_generated
    assert first != second
    assert len(first) >= 24

    monkeypatch.setenv("DEV_USER_PASSWORD", "supplied-by-the-operator")
    assert seed_module._resolve_dev_password() == ("supplied-by-the-operator", False)

    source = SEED_SCRIPT.read_text(encoding="utf-8")
    assert "dev-pass-123" not in source
    # Claiming production to quieten SQL echo also disarmed the dev guard.
    assert 'setdefault("ENVIRONMENT"' not in source
    assert "require_development_environment" in source


def test_reset_guards_run_before_the_env_file_is_merged() -> None:
    script = RESET_SCRIPT.read_text(encoding="utf-8")
    guards, marker, remainder = script.partition("\nset -a\n")
    assert marker, "expected the env-file merge to still be present"

    # Every value a guard decides on is parsed out of the named file, before the
    # merge, so an exported SUPABASE_NON_PRODUCTION cannot satisfy a check whose
    # error message points at the file.
    assert 'CONFIRM_INPUT="${CONFIRM_SUPABASE_RESET:-}"' in guards
    assert (
        'FILE_NON_PRODUCTION="$(lowercase "$(env_file_value SUPABASE_NON_PRODUCTION)")"' in guards
    )
    assert 'FILE_ENVIRONMENT="$(lowercase "$(env_file_value ENVIRONMENT)")"' in guards
    assert '[[ "$FILE_NON_PRODUCTION" != "true" ]]' in guards
    assert '[[ "$FILE_ENVIRONMENT" == "production" ]]' in guards
    assert '[[ "$CONFIRM_INPUT" != "$TARGET" ]]' in guards

    # The destructive statement uses the parsed URL, not the merged environment.
    assert 'psql "$FILE_MIGRATOR_URL"' in remainder

    # The exact bypassed forms must not come back.
    assert '"${SUPABASE_NON_PRODUCTION:-}"' not in script
    assert '"${CONFIRM_SUPABASE_RESET:-}" != "RESET"' not in script
