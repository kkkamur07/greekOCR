"""Alembic head parity checks."""

from __future__ import annotations

import importlib


def test_squashed_migration_chain_is_three_linear_revisions():
    schema = importlib.import_module("infrastructure.alembic.versions.001_initial_schema")
    roles = importlib.import_module("infrastructure.alembic.versions.002_service_roles")
    devices = importlib.import_module("infrastructure.alembic.versions.003_helper_devices")

    assert schema.revision == "001_initial_schema"
    assert schema.down_revision is None
    assert roles.revision == "002_service_roles"
    assert roles.down_revision == schema.revision
    assert devices.revision == "003_helper_devices"
    assert devices.down_revision == roles.revision


def test_the_collapsed_queue_leaves_no_enum_behind():
    """ADR 0003 left one queue, so ``inference_job_status`` is never created.

    Asserted on the enum tuple rather than the source text: the docstring names
    the revision that used to drop it, and a substring search would match that.
    """
    schema = importlib.import_module("infrastructure.alembic.versions.001_initial_schema")

    names = {enum_type.name for enum_type in schema._ENUM_TYPES}
    assert "inference_job_status" not in names
    assert "execution_target" in names
