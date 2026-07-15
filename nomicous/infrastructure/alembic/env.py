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
    fileConfig(config.config_file_name)

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
