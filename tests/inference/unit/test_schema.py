"""Schema parity checks for the ML-owned job queue and Alembic head."""

from __future__ import annotations

import importlib

from inference.infrastructure.orm_models import InferenceJob


def test_ml_job_metadata_includes_claim_order_index():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in InferenceJob.__table__.indexes
    }

    assert indexes["ix_inference_jobs_claim_order"] == ("status", "created_at", "id")


def test_squashed_migration_chain_has_one_schema_revision_and_role_revision():
    schema = importlib.import_module("infrastructure.alembic.versions.001_initial_schema")
    roles = importlib.import_module("infrastructure.alembic.versions.002_service_roles")

    assert schema.revision == "001_initial_schema"
    assert schema.down_revision is None
    assert roles.revision == "002_service_roles"
    assert roles.down_revision == schema.revision
