"""Create the baseline application schema.

This is the squashed replacement for the pre-production migration history.
Application authorization remains in FastAPI; this schema intentionally does not
enable PostgreSQL row-level security.

Service role grants live in 002_service_roles.

THE DDL BELOW IS FROZEN. It is a historical record of the schema as it stood
immediately before 002, and it must never be regenerated from live ORM metadata
(``Base.metadata.create_all``) again. It previously was, and that cost us two
things:

* the migration stopped being replayable - running 001 produced *today's* ORM
  schema rather than the schema this revision actually created, so migration
  history could not be audited or replayed to any point in time; and
* it made a missing migration structurally undetectable - a fresh database
  migrated to head always matched the ORM because head *was* the ORM, so an
  ``alembic check`` style diff could never fail. 003_job_lifecycle and
  004_document_part_dimensions are the scar tissue from schema changes that
  slipped through and had to be patched after the fact.

So: every schema change after this revision gets its own migration. Columns and
tables that later revisions add are deliberately absent here - jobs.claimed_by /
jobs.heartbeat_at (003), document_parts.width / height (004), and
helper_devices / helper_pairings (005).

``tests/nomicous/integration/test_migrations.py`` is what keeps this honest: it
migrates a scratch database to head and asserts the autogenerate diff against
``Base.metadata`` is empty.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Enum types are declared with ``create_type=False`` and created explicitly, once,
# by ``_create_enum_types``. Left to itself SQLAlchemy emits a CREATE TYPE from
# whichever CREATE TABLE happens to mention the type first, which breaks the
# moment two tables share one (inference_task is used by inference_models and by
# inference_jobs) and makes the drop order in downgrade() guesswork.
def _enum(name: str, *values: str) -> postgresql.ENUM:
    return postgresql.ENUM(*values, name=name, create_type=False)


INFERENCE_TASK = _enum("inference_task", "segment", "transcribe", "binarize")
BINDING_TASK = _enum("binding_task", "segment", "transcribe", "binarize")
DOCUMENT_WORKFLOW = _enum("document_workflow", "draft", "published", "archived")
TRANSCRIPTION_KIND = _enum("transcription_kind", "ground_truth", "model")
LINE_GEOMETRY_KIND = _enum("line_geometry_kind", "polygon", "rectangle")
LINE_SOURCE = _enum("line_source", "manual", "kraken", "model")
JOB_TYPE = _enum("job_type", "segment", "transcribe", "binarize", "pipeline")
JOB_STATUS = _enum("job_status", "pending", "waiting", "running", "done", "failed", "cancelled")
INFERENCE_JOB_STATUS = _enum("inference_job_status", "pending", "running", "done", "failed")

_ENUM_TYPES = (
    INFERENCE_TASK,
    BINDING_TASK,
    DOCUMENT_WORKFLOW,
    TRANSCRIPTION_KIND,
    LINE_GEOMETRY_KIND,
    LINE_SOURCE,
    JOB_TYPE,
    JOB_STATUS,
)


def _create_enum_types() -> None:
    bind = op.get_bind()
    for enum_type in _ENUM_TYPES:
        enum_type.create(bind, checkfirst=True)


def _create_users() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("username", sa.String(length=150), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("csrf_token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_auth_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_sessions")),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"])

    # BIGSERIAL, not UUID: this is a high-churn append-and-sweep counter table,
    # never referenced by anything, so a monotonic key keeps the index tight.
    op.create_table(
        "auth_rate_limit_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_rate_limit_attempts")),
    )
    op.create_index(
        "ix_auth_rate_limit_key_time",
        "auth_rate_limit_attempts",
        ["key", "attempted_at"],
    )


def _create_projects() -> None:
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("slug", sa.String(length=512), nullable=False),
        sa.Column("guidelines", sa.Text(), nullable=True),
        # SET NULL, not CASCADE: deleting a user must not delete the corpus they
        # own; the project is re-assignable.
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_projects_owner_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
    )
    op.create_index(op.f("ix_projects_owner_id"), "projects", ["owner_id"])
    op.create_index(op.f("ix_projects_slug"), "projects", ["slug"], unique=True)

    op.create_table(
        "project_shared_users",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_project_shared_users_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_project_shared_users_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "user_id", name=op.f("pk_project_shared_users")),
    )


def _create_documents() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("workflow", DOCUMENT_WORKFLOW, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_documents_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_index(op.f("ix_documents_project_id"), "documents", ["project_id"])

    # width/height are NOT here - they arrive in 004_document_part_dimensions.
    op.create_table(
        "document_parts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("image_key", sa.String(length=1024), nullable=False),
        sa.Column("reviewed", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_parts_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_parts")),
        sa.UniqueConstraint("document_id", "order", name="uq_document_parts_document_order"),
    )
    op.create_index(op.f("ix_document_parts_document_id"), "document_parts", ["document_id"])
    op.create_index("ix_document_parts_document_order", "document_parts", ["document_id", "order"])

    # Durable outbox for object-store deletions: the object lives outside the
    # transaction, so the intent is committed with the row delete and swept later.
    op.create_table(
        "media_deletion_intents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_key", sa.String(length=1024), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_media_deletion_intents")),
        sa.UniqueConstraint("image_key", name=op.f("uq_media_deletion_intents_image_key")),
    )
    # Partial: the sweeper only ever scans the unfinished tail.
    op.create_index(
        "ix_media_deletion_intents_pending",
        "media_deletion_intents",
        ["created_at"],
        postgresql_where=sa.text("completed_at IS NULL"),
    )


def _create_layout() -> None:
    op.create_table(
        "blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("box", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("manual_geometry", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["document_parts.id"],
            name=op.f("fk_blocks_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_blocks")),
    )
    op.create_index(op.f("ix_blocks_part_id"), "blocks", ["part_id"])

    op.create_table(
        "lines",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("block_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("baseline", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("mask", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("kind", LINE_GEOMETRY_KIND, server_default="polygon", nullable=False),
        sa.Column(
            "points",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("source", LINE_SOURCE, server_default="manual", nullable=False),
        sa.Column("source_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("kraken_ceiling", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("manual_geometry", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # SET NULL: unassigning a block must not take its lines with it.
        sa.ForeignKeyConstraint(
            ["block_id"],
            ["blocks.id"],
            name=op.f("fk_lines_block_id_blocks"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["document_parts.id"],
            name=op.f("fk_lines_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lines")),
    )
    op.create_index(op.f("ix_lines_block_id"), "lines", ["block_id"])
    op.create_index(op.f("ix_lines_part_id"), "lines", ["part_id"])
    # created_at is the tiebreaker: "order" is not unique while an edit is in flight.
    op.create_index("ix_lines_part_order", "lines", ["part_id", "order", "created_at"])

    op.create_table(
        "annotation_history_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("paired_line_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["document_parts.id"],
            name=op.f("fk_annotation_history_snapshots_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_annotation_history_snapshots")),
    )
    op.create_index(
        op.f("ix_annotation_history_snapshots_part_id"),
        "annotation_history_snapshots",
        ["part_id"],
    )
    op.create_index(
        "ix_annotation_history_snapshots_part_created",
        "annotation_history_snapshots",
        ["part_id", "created_at"],
    )


def _create_ml() -> None:
    op.create_table(
        "inference_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("task", INFERENCE_TASK, nullable=False),
        sa.Column("artifact_ref", sa.String(length=1024), nullable=False),
        sa.Column(
            "default_params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_models")),
        sa.UniqueConstraint("name", name=op.f("uq_inference_models_name")),
    )

    # A binding attaches a model to exactly one scope (project / document / part);
    # the nullable FKs are the scope discriminator, resolved most-specific-first.
    op.create_table(
        "model_bindings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task", BINDING_TASK, nullable=False),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_part_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "overrides",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_model_bindings_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_part_id"],
            ["document_parts.id"],
            name=op.f("fk_model_bindings_document_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["inference_models.id"],
            name=op.f("fk_model_bindings_model_id_inference_models"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_model_bindings_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_bindings")),
    )
    op.create_index(op.f("ix_model_bindings_document_id"), "model_bindings", ["document_id"])
    op.create_index(
        op.f("ix_model_bindings_document_part_id"), "model_bindings", ["document_part_id"]
    )
    op.create_index(op.f("ix_model_bindings_model_id"), "model_bindings", ["model_id"])
    op.create_index(op.f("ix_model_bindings_project_id"), "model_bindings", ["project_id"])


def _create_jobs() -> None:
    # claimed_by/heartbeat_at are NOT here - they arrive in 003_job_lifecycle.
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        # Deliberately not an FK: inference_jobs lives in the inference service's
        # own schema and may be a different database entirely.
        sa.Column("inference_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("type", JOB_TYPE, nullable=False),
        sa.Column("status", JOB_STATUS, nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        # Every reference is SET NULL: a job row is an audit record and outlives
        # the model, binding, user or page it ran against.
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("binding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_part_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("callback_claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["binding_id"],
            ["model_bindings.id"],
            name=op.f("fk_jobs_binding_id_model_bindings"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_jobs_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_part_id"],
            ["document_parts.id"],
            name=op.f("fk_jobs_document_part_id_document_parts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["inference_models.id"],
            name=op.f("fk_jobs_model_id_inference_models"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_jobs_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index(op.f("ix_jobs_document_id"), "jobs", ["document_id"])
    op.create_index(op.f("ix_jobs_inference_job_id"), "jobs", ["inference_job_id"])
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"])
    # GIN: the worker filters jobs by keys *inside* payload.
    op.create_index("ix_jobs_payload_gin", "jobs", ["payload"], postgresql_using="gin")
    # Partial: the claim query only ever looks at the pending head of the queue.
    op.create_index(
        "ix_jobs_claim_pending",
        "jobs",
        ["created_at", "id"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def _create_transcriptions() -> None:
    op.create_table(
        "transcriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", TRANSCRIPTION_KIND, nullable=False),
        sa.Column("created_by_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by_job_id"],
            ["jobs.id"],
            name=op.f("fk_transcriptions_created_by_job_id_jobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_transcriptions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcriptions")),
    )
    op.create_index(op.f("ix_transcriptions_document_id"), "transcriptions", ["document_id"])
    # Partial unique: a document has at most one ground-truth layer, but any
    # number of model layers.
    op.create_index(
        "uq_transcriptions_one_ground_truth",
        "transcriptions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'ground_truth'"),
    )

    op.create_table(
        "line_transcriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("line_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transcription_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["line_id"],
            ["lines.id"],
            name=op.f("fk_line_transcriptions_line_id_lines"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["transcription_id"],
            ["transcriptions.id"],
            name=op.f("fk_line_transcriptions_transcription_id_transcriptions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_line_transcriptions")),
        sa.UniqueConstraint(
            "line_id", "transcription_id", name="uq_line_transcriptions_line_layer"
        ),
    )
    op.create_index(op.f("ix_line_transcriptions_line_id"), "line_transcriptions", ["line_id"])
    op.create_index(
        op.f("ix_line_transcriptions_transcription_id"),
        "line_transcriptions",
        ["transcription_id"],
    )

    # Page-level transcription: free text typed before (or without) any line
    # geometry, optionally paired 1:1 with a segmented line once one exists.
    op.create_table(
        "page_transcription_lines",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("paired_line_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["paired_line_id"],
            ["lines.id"],
            name=op.f("fk_page_transcription_lines_paired_line_id_lines"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["document_parts.id"],
            name=op.f("fk_page_transcription_lines_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_page_transcription_lines")),
        sa.UniqueConstraint("part_id", "order", name="uq_page_transcription_lines_part_order"),
        sa.UniqueConstraint("paired_line_id", name="uq_page_transcription_lines_paired_line"),
    )
    op.create_index(
        op.f("ix_page_transcription_lines_paired_line_id"),
        "page_transcription_lines",
        ["paired_line_id"],
    )
    op.create_index(
        op.f("ix_page_transcription_lines_part_id"),
        "page_transcription_lines",
        ["part_id"],
    )


def _create_inference_jobs() -> None:
    bind = op.get_bind()
    inference_task = postgresql.ENUM(
        "segment",
        "transcribe",
        "binarize",
        name="inference_task",
        create_type=False,
    )
    inference_job_status = postgresql.ENUM(
        "pending",
        "running",
        "done",
        "failed",
        name="inference_job_status",
        create_type=False,
    )
    inference_job_status.create(bind, checkfirst=True)
    op.create_table(
        "inference_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("product_job_id", sa.UUID(), nullable=False),
        sa.Column("task", inference_task, nullable=False),
        sa.Column("registry_model_id", sa.Text(), nullable=False),
        sa.Column("registry_tag", sa.Text(), nullable=False),
        sa.Column("status", inference_job_status, nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column(
            "params", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inference_jobs")),
    )
    op.create_index(op.f("ix_inference_jobs_product_job_id"), "inference_jobs", ["product_job_id"])
    op.create_index(op.f("ix_inference_jobs_status"), "inference_jobs", ["status"])
    op.create_index(
        "ix_inference_jobs_claim_order",
        "inference_jobs",
        ["status", "created_at", "id"],
    )


def upgrade() -> None:
    _create_enum_types()
    _create_users()
    _create_projects()
    _create_documents()
    _create_layout()
    _create_ml()
    _create_jobs()
    _create_transcriptions()
    _create_inference_jobs()


def downgrade() -> None:
    bind = op.get_bind()
    # Reverse dependency order; indexes go with their table.
    for table in (
        "inference_jobs",
        "page_transcription_lines",
        "line_transcriptions",
        "transcriptions",
        "jobs",
        "model_bindings",
        "inference_models",
        "annotation_history_snapshots",
        "lines",
        "blocks",
        "media_deletion_intents",
        "document_parts",
        "documents",
        "project_shared_users",
        "projects",
        "auth_rate_limit_attempts",
        "auth_sessions",
        "users",
    ):
        op.drop_table(table)
    for enum_type in (*_ENUM_TYPES, INFERENCE_JOB_STATUS):
        enum_type.drop(bind, checkfirst=True)
