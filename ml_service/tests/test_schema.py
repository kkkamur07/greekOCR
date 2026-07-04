"""Schema parity checks for the ML-owned job queue."""

from __future__ import annotations

import importlib

from ml_service.infrastructure.orm_models import MLJob


def test_ml_job_metadata_includes_claim_order_index():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in MLJob.__table__.indexes
    }

    assert indexes["ix_ml_jobs_claim_order"] == ("status", "created_at", "id")


def test_migration_013_handles_precreated_ml_jobs_table(monkeypatch):
    migration = importlib.import_module("infrastructure.alembic.versions.013_ml_jobs_queue")
    calls: list[tuple[str, bool | None]] = []
    bind = object()

    class DummyEnum:
        def create(self, _bind, *, checkfirst: bool) -> None:
            assert _bind is bind
            calls.append(("enum", checkfirst))

    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration, "ml_task", DummyEnum())
    monkeypatch.setattr(migration, "ml_job_status", DummyEnum())
    monkeypatch.setattr(migration, "_ml_jobs_table_exists", lambda _bind: True)
    monkeypatch.setattr(migration, "_create_ml_jobs_table", lambda: calls.append(("table", None)))
    monkeypatch.setattr(
        migration,
        "_create_missing_ml_jobs_indexes",
        lambda _bind: calls.append(("indexes", None)),
    )

    migration.upgrade()

    assert ("table", None) not in calls
    assert ("indexes", None) in calls
    assert calls.count(("enum", True)) == 2


def test_migration_013_index_helper_creates_missing_indexes(monkeypatch):
    migration = importlib.import_module("infrastructure.alembic.versions.013_ml_jobs_queue")
    monkeypatch.setattr(migration.op, "f", lambda name: name)
    engine = migration.sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.execute(
            migration.sa.text(
                "CREATE TABLE ml_jobs ("
                "id TEXT, product_job_id TEXT, status TEXT, created_at TEXT)"
            )
        )

        migration._create_missing_ml_jobs_indexes(connection)
        migration._create_missing_ml_jobs_indexes(connection)

        index_names = {
            row[1] for row in connection.execute(migration.sa.text("PRAGMA index_list(ml_jobs)"))
        }

    assert {
        "ix_ml_jobs_product_job_id",
        "ix_ml_jobs_status",
        "ix_ml_jobs_claim_order",
    }.issubset(index_names)

