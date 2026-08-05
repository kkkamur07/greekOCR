"""Alembic head parity checks."""

from __future__ import annotations

import importlib


def test_squashed_migration_chain_has_one_schema_revision_and_role_revision():
    schema = importlib.import_module("infrastructure.alembic.versions.001_initial_schema")
    roles = importlib.import_module("infrastructure.alembic.versions.002_service_roles")

    assert schema.revision == "001_initial_schema"
    assert schema.down_revision is None
    assert roles.revision == "002_service_roles"
    assert roles.down_revision == schema.revision
