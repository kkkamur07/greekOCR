"""Alembic migration environment - sync engine via SYNC_DATABASE_URL."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from backend.core.settings import get_infrastructure_settings
from infrastructure.db import Base
from infrastructure import models  # noqa: F401 - register all ORM tables

config = context.config


def _migrator_database_url() -> str:
    return get_infrastructure_settings().migrator_database_url


# ConfigParser treats % as interpolation; escape when storing in alembic.ini section.
config.set_main_option("sqlalchemy.url", _migrator_database_url().replace("%", "%%"))

if config.config_file_name is not None:
    # disable_existing_loggers=False, and it is not cosmetic. fileConfig defaults
    # to True, which switches off every logger that already exists rather than
    # merely reconfiguring the ones alembic.ini names.
    #
    # In-process that is destructive: this module is imported by the migration
    # test, and from that point on the application's own loggers are dead for the
    # rest of the session. Five unit tests asserting on log output saw an empty
    # caplog and failed, in a run where they had done nothing wrong - and passed
    # when their directory ran alone, which is what made it look like flake.
    #
    # It is also wrong outside the tests: a migration run inside any process that
    # had already configured logging would silently take that logging with it.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = _migrator_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_migrator_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
