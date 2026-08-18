"""The suite refuses to wipe a database that is not on this machine.

``conftest._truncate_database`` runs before every test in this directory and
TRUNCATEs every mapped table with ``RESTART IDENTITY CASCADE``. Nothing about it
is reversible, and the environment defaults in that module are ``setdefault``,
which means an exported ``SYNC_DATABASE_URL`` wins. That is the whole exposure:
a developer with a Supabase URL live in their shell running pytest.

Nothing here opens a connection. ``create_engine`` resolves a dialect and parses
a URL; it does not dial anything until a connection is asked for, and every
assertion below is about a refusal that happens strictly before that point.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from tests.nomikos.integration import conftest
from tests.nomikos.integration.conftest import require_local_database

pytestmark = pytest.mark.integration

REMOTE_URL = "postgresql://postgres:hunter2@db.abcdefghijkl.supabase.co:5432/postgres"


def _engine(url: str) -> Engine:
    return create_engine(url)


def test_a_remote_host_is_refused_by_name() -> None:
    with pytest.raises(RuntimeError) as refusal:
        require_local_database(_engine(REMOTE_URL))

    message = str(refusal.value)
    # The host has to appear, or the developer cannot tell which database they
    # were one fixture away from truncating.
    assert "db.abcdefghijkl.supabase.co" in message
    assert "SYNC_DATABASE_URL" in message
    # The refusal quotes the target back, so it must not quote the password with it.
    assert "hunter2" not in message


def test_a_url_without_a_host_is_refused_rather_than_assumed_local() -> None:
    """Fail closed. A host that cannot be read cannot be shown to be loopback."""
    with pytest.raises(RuntimeError, match="no host"):
        require_local_database(_engine("postgresql://postgres:dev@/kalamos"))


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://postgres:dev@localhost:5433/kalamos",
        "postgresql://postgres:dev@127.0.0.1:5433/kalamos",
        "postgresql://postgres:dev@[::1]:5433/kalamos",
        "postgresql://postgres:dev@LocalHost:5433/kalamos",
    ],
)
def test_loopback_targets_are_allowed(url: str) -> None:
    require_local_database(_engine(url))


def test_the_engine_this_suite_actually_truncates_is_local() -> None:
    require_local_database(conftest._truncate_engine)


def test_truncating_re_checks_the_engine_instead_of_trusting_the_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard lives inside the helper, not only at module import.

    An import-time check alone would be bypassed by anything that calls
    ``_truncate_database`` after the engine has been repointed - which is exactly
    the shape of the accident being guarded against.
    """
    monkeypatch.setattr(conftest, "_truncate_engine", _engine(REMOTE_URL))

    with pytest.raises(RuntimeError, match="Refusing to wipe"):
        conftest._truncate_database()
