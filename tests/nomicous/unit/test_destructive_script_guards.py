"""Guards on the destructive platform scripts (schema reset, dev corpus seed).

Both guards were bypassable from ambient shell state, so the regressions here are
about *provenance*: the reset must trust only the env file it names, and the seed
must go through the shared development guard instead of asserting its own
environment.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
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


# ---------------------------------------------------------------------------
# The reset script's guards, run rather than read
# ---------------------------------------------------------------------------
#
# These used to be substring assertions over the script's source. Every one of
# them was satisfied by the text appearing anywhere above `set -a` - in a
# comment, in a function nothing calls, or in a branch whose `exit 1` had been
# deleted - so the guards could stop guarding with the test still green. The
# script is executed instead, against a throwaway env file, with a stub `psql`
# first on PATH standing in for the irreversible statement. What is asserted is
# the only thing that matters at a shell prompt: it exits non-zero, and `psql`
# was never reached.
#
# Deliberately nothing here touches the DROP lists. Those change as the schema
# does; the guards do not.

#: A database the guards must never let the script reach.
SCRATCH_URL = "postgresql://postgres:pw@localhost:5432/scratch"
#: What ``database_target`` makes of it - the string an operator has to type.
SCRATCH_TARGET = "localhost:5432/scratch"


@pytest.fixture
def psql_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A `psql` that records having been called and then fails.

    Recording is what makes "the guard stopped it" checkable. Failing is what
    keeps a *broken* guard from carrying the run onwards into
    ``migrate_supabase.sh`` and the dev seed against whatever database the env
    file named: with ``set -e``, a non-zero psql ends the script right there.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sentinel = tmp_path / "psql-was-called"
    stub = bin_dir / "psql"
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"{sentinel}"\nexit 1\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    # `uv` and `alembic` are only reachable past psql, but a stub that cannot be
    # found would fall through to the real one if psql ever stopped failing.
    for name in ("uv", "alembic"):
        fallback = bin_dir / name
        fallback.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        fallback.chmod(0o755)
    return bin_dir, sentinel


def _run_reset(
    env_file: Path,
    *,
    bin_dir: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    child = dict(os.environ)
    # The guards exist to be immune to ambient shell state, so the ambient state
    # a developer's profile might carry is cleared and then set explicitly.
    for inherited in (
        "SUPABASE_NON_PRODUCTION",
        "ENVIRONMENT",
        "MIGRATOR_DATABASE_URL",
        "SUPABASE_URL",
        "CONFIRM_SUPABASE_RESET",
        "SUPABASE_PRODUCTION_PROJECT_REFS",
        "SUPABASE_RESET_ASSUME_YES",
    ):
        child.pop(inherited, None)
    child["SUPABASE_ENV_FILE"] = str(env_file)
    child["PATH"] = f"{bin_dir}{os.pathsep}{child.get('PATH', '')}"
    child.update(environment or {})
    return subprocess.run(
        ["bash", str(RESET_SCRIPT), "--yes"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=REPO_ROOT,
        env=child,
        timeout=120,
    )


def _env_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".env.supabase"
    path.write_text(body, encoding="utf-8")
    return path


def test_reset_reaches_psql_when_every_guard_is_satisfied(
    tmp_path: Path, psql_stub: tuple[Path, Path]
) -> None:
    """The positive control, and the reason the refusals below mean anything.

    Without it, "psql was never called" would also be satisfied by a harness
    that could never call psql at all - which is the shape of failure this whole
    file was rewritten to stop believing.
    """
    bin_dir, sentinel = psql_stub
    env_file = _env_file(
        tmp_path,
        "SUPABASE_NON_PRODUCTION=true\n"
        "ENVIRONMENT=development\n"
        f"MIGRATOR_DATABASE_URL={SCRATCH_URL}\n",
    )

    result = _run_reset(
        env_file,
        bin_dir=bin_dir,
        environment={"CONFIRM_SUPABASE_RESET": SCRATCH_TARGET},
    )

    assert sentinel.exists(), f"the run never reached psql: {result.stderr}"
    assert "Guard failed" not in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("name", "body", "environment", "expected"),
    [
        (
            "the env file declares production",
            "SUPABASE_NON_PRODUCTION=true\n"
            "ENVIRONMENT=production\n"
            f"MIGRATOR_DATABASE_URL={SCRATCH_URL}\n",
            {"CONFIRM_SUPABASE_RESET": SCRATCH_TARGET},
            "ENVIRONMENT=production",
        ),
        (
            # The regression this script was rewritten for: an export in a shell
            # profile satisfying a guard whose message points at the file.
            "an exported flag cannot stand in for the file's",
            f"ENVIRONMENT=development\nMIGRATOR_DATABASE_URL={SCRATCH_URL}\n",
            {
                "SUPABASE_NON_PRODUCTION": "true",
                "CONFIRM_SUPABASE_RESET": SCRATCH_TARGET,
            },
            "SUPABASE_NON_PRODUCTION is not 'true'",
        ),
        (
            "the environment is missing entirely",
            f"SUPABASE_NON_PRODUCTION=true\nMIGRATOR_DATABASE_URL={SCRATCH_URL}\n",
            {"CONFIRM_SUPABASE_RESET": SCRATCH_TARGET},
            "ENVIRONMENT is not set",
        ),
        (
            "the confirmation names a different target",
            "SUPABASE_NON_PRODUCTION=true\n"
            "ENVIRONMENT=development\n"
            f"MIGRATOR_DATABASE_URL={SCRATCH_URL}\n",
            {"CONFIRM_SUPABASE_RESET": "some-other-database"},
            "confirmation did not match",
        ),
        (
            # An env file confirming its own destruction. CONFIRM_INPUT is read
            # before the merge precisely so this cannot work.
            "the env file tries to confirm itself",
            "SUPABASE_NON_PRODUCTION=true\n"
            "ENVIRONMENT=development\n"
            f"MIGRATOR_DATABASE_URL={SCRATCH_URL}\n"
            f"CONFIRM_SUPABASE_RESET={SCRATCH_TARGET}\n",
            {},
            "confirmation did not match",
        ),
        (
            "the operator denylist names the target",
            "SUPABASE_NON_PRODUCTION=true\n"
            "ENVIRONMENT=staging\n"
            "MIGRATOR_DATABASE_URL="
            "postgresql://postgres:pw@db.abcdefghijklmnop.supabase.co:5432/postgres\n",
            {
                "CONFIRM_SUPABASE_RESET": "abcdefghijklmnop",
                "SUPABASE_PRODUCTION_PROJECT_REFS": "zzz,abcdefghijklmnop",
            },
            "SUPABASE_PRODUCTION_PROJECT_REFS",
        ),
        (
            "the file mixes two Supabase projects",
            "SUPABASE_NON_PRODUCTION=true\n"
            "ENVIRONMENT=staging\n"
            "MIGRATOR_DATABASE_URL="
            "postgresql://postgres:pw@db.abcdefghijklmnop.supabase.co:5432/postgres\n"
            "SUPABASE_URL=https://qrstuvwxyzabcdef.supabase.co\n",
            {"CONFIRM_SUPABASE_RESET": "abcdefghijklmnop"},
            "mixes Supabase projects",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and " " in value else None,
)
def test_reset_refuses_before_it_can_drop_anything(
    tmp_path: Path,
    psql_stub: tuple[Path, Path],
    name: str,
    body: str,
    environment: dict[str, str],
    expected: str,
) -> None:
    bin_dir, sentinel = psql_stub
    env_file = _env_file(tmp_path, body)

    result = _run_reset(env_file, bin_dir=bin_dir, environment=environment)

    assert result.returncode != 0, f"{name}: the reset was allowed to proceed"
    assert "Guard failed" in result.stderr, result.stderr
    assert expected in result.stderr, result.stderr
    assert not sentinel.exists(), f"{name}: psql ran anyway - {sentinel.read_text()}"
