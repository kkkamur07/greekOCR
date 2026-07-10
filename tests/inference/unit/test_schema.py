"""Schema parity checks for the ML-owned job queue."""

from __future__ import annotations

import importlib

from inference.infrastructure.orm_models import InferenceJob


# --- ORM schema ---
# Tests claim-order index on inference_jobs. Does not run migrations against Postgres.


def test_ml_job_metadata_includes_claim_order_index():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in InferenceJob.__table__.indexes
    }

    assert indexes["ix_inference_jobs_claim_order"] == ("status", "created_at", "id")


# --- Alembic migration 013 ---
# Tests idempotent upgrade when table already exists. Does not apply migrations to a real DB.


def test_migration_013_handles_precreated_inference_jobs_table(monkeypatch):
    migration = importlib.import_module("infrastructure.alembic.versions.013_ml_jobs_queue")
    calls: list[tuple[str, bool | None]] = []
    bind = object()

    class DummyEnum:
        def create(self, _bind, *, checkfirst: bool) -> None:
            assert _bind is bind
            calls.append(("enum", checkfirst))

    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration, "inference_task", DummyEnum())
    monkeypatch.setattr(migration, "inference_job_status", DummyEnum())
    monkeypatch.setattr(migration, "_inference_jobs_table_exists", lambda _bind: True)
    monkeypatch.setattr(
        migration, "_create_inference_jobs_table", lambda: calls.append(("table", None))
    )
    monkeypatch.setattr(
        migration,
        "_create_missing_inference_jobs_indexes",
        lambda _bind: calls.append(("indexes", None)),
    )

    migration.upgrade()

    assert ("table", None) not in calls
    assert ("indexes", None) in calls
    assert calls.count(("enum", True)) == 2


# --- Migration index helper ---
# Tests missing indexes are created idempotently on SQLite. Does not run full Alembic upgrade.


def test_migration_013_index_helper_creates_missing_indexes(monkeypatch):
    migration = importlib.import_module("infrastructure.alembic.versions.013_ml_jobs_queue")
    monkeypatch.setattr(migration.op, "f", lambda name: name)
    engine = migration.sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        connection.execute(
            migration.sa.text(
                "CREATE TABLE inference_jobs ("
                "id TEXT, product_job_id TEXT, status TEXT, created_at TEXT)"
            )
        )

        migration._create_missing_inference_jobs_indexes(connection)
        migration._create_missing_inference_jobs_indexes(connection)

        index_names = {
            row[1]
            for row in connection.execute(migration.sa.text("PRAGMA index_list(inference_jobs)"))
        }

    assert {
        "ix_inference_jobs_product_job_id",
        "ix_inference_jobs_status",
        "ix_inference_jobs_claim_order",
    }.issubset(index_names)
