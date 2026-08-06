"""The migration chain must reproduce ``Base.metadata`` exactly.

This is the ``alembic check`` the project was missing. 001_initial_schema used to
build the baseline with ``Base.metadata.create_all``, which made a missing
migration structurally undetectable: a fresh database migrated to head always
matched the ORM because head *was* the ORM. The revisions that had to be written
afterwards to add ``jobs.claimed_by`` and ``document_parts.width`` are what that
cost us - schema changes that reached production before anyone noticed they had
no migration. They are folded back into the baseline now, but only because this
test exists to catch the next one.

So this test migrates a *scratch* database from empty to head and asserts that
alembic's autogenerate comparison against ``Base.metadata`` finds nothing. Add a
column to an ORM model without a migration and this goes red.

The scratch database is created and dropped per run rather than reusing the test
database: the point is to exercise the chain from nothing, and running it against
the shared database would collide with every other integration test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import urlsplit, urlunsplit

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

import infrastructure.models  # noqa: F401 - register all ORM tables
from backend.core.settings import get_infrastructure_settings
from infrastructure.db import Base

pytestmark = pytest.mark.integration

_SCRATCH_DATABASE = "nomicous_migration_check"

_ALEMBIC_INI = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../nomicous/infrastructure/alembic.ini",
)

# Tables the chain creates that are deliberately absent from ``Base.metadata``.
# ``alembic_version`` is alembic's own bookkeeping. ``inference_jobs`` used to be
# listed here because the inference service owned its own mapper; ADR 0003
# collapsed that queue into ``jobs`` and the squashed baseline never creates the
# table, so the exemption is gone.
_UNMANAGED_TABLES = frozenset({"alembic_version"})


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{database}", query=""))


def _maintenance_url(url: str) -> str:
    """A connection that is not to the database being created or dropped."""
    return _with_database(url, "postgres")


def _run_ddl(url: str, statement: str) -> None:
    # CREATE/DROP DATABASE cannot run inside a transaction block.
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(text(statement))
    finally:
        engine.dispose()


@pytest.fixture
def migrated_scratch_database() -> Iterator[str]:
    """Create an empty database, migrate it to head, yield its URL, drop it."""
    base_url = get_infrastructure_settings().migrator_database_url
    maintenance_url = _maintenance_url(base_url)
    scratch_url = _with_database(base_url, _SCRATCH_DATABASE)

    _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')
    _run_ddl(maintenance_url, f'CREATE DATABASE "{_SCRATCH_DATABASE}"')

    # alembic's env.py resolves its URL from the cached settings object, so the
    # override has to go through the environment and the cache has to be cleared
    # on both sides of the run or the rest of the session keeps the scratch URL.
    previous = os.environ.get("MIGRATOR_DATABASE_URL")
    os.environ["MIGRATOR_DATABASE_URL"] = scratch_url
    get_infrastructure_settings.cache_clear()
    try:
        command.upgrade(Config(_ALEMBIC_INI), "head")
        yield scratch_url
    finally:
        if previous is None:
            os.environ.pop("MIGRATOR_DATABASE_URL", None)
        else:
            os.environ["MIGRATOR_DATABASE_URL"] = previous
        get_infrastructure_settings.cache_clear()
        _run_ddl(maintenance_url, f'DROP DATABASE IF EXISTS "{_SCRATCH_DATABASE}" WITH (FORCE)')


def test_migration_chain_matches_orm_metadata(migrated_scratch_database: str) -> None:
    """An empty database migrated to head has no autogenerate diff vs the ORM.

    A non-empty diff means a model changed without a migration. Read the printed
    operations: ``add_column``/``remove_column``/``add_table`` name exactly what
    the missing revision has to do.
    """

    def include_name(name, type_, parent_names) -> bool:
        return type_ != "table" or name not in _UNMANAGED_TABLES

    engine = create_engine(migrated_scratch_database)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "include_name": include_name,
                    "compare_type": True,
                    "compare_server_default": True,
                },
            )
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], (
        "The migration chain no longer reproduces Base.metadata. "
        "Add a migration for these changes instead of editing 001_initial_schema:\n"
        + "\n".join(repr(entry) for entry in diff)
    )
