"""inference domain (jobs) — ModelBinding, Job

Revision ID: 005_inference_jobs
Revises: 004_document_layout
Create Date: 2026-05-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_inference_jobs"
down_revision: Union[str, None] = "004_document_layout"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

binding_task = postgresql.ENUM(
    "segment", "transcribe", "binarize", name="binding_task", create_type=False
)
job_type = postgresql.ENUM(
    "segment", "transcribe", "binarize", "pipeline", name="job_type", create_type=False
)
job_status = postgresql.ENUM("pending", "running", "done", "failed", name="job_status", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    binding_task.create(bind, checkfirst=True)
    job_type.create(bind, checkfirst=True)
    job_status.create(bind, checkfirst=True)

    op.create_table(
        "model_bindings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("task", binding_task, nullable=False),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=True),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("document_part_id", sa.UUID(), nullable=True),
        sa.Column(
            "overrides",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
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
    op.create_index(op.f("ix_model_bindings_document_id"), "model_bindings", ["document_id"], unique=False)
    op.create_index(
        op.f("ix_model_bindings_document_part_id"), "model_bindings", ["document_part_id"], unique=False
    )
    op.create_index(op.f("ix_model_bindings_model_id"), "model_bindings", ["model_id"], unique=False)
    op.create_index(op.f("ix_model_bindings_project_id"), "model_bindings", ["project_id"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("type", job_type, nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model_id", sa.UUID(), nullable=True),
        sa.Column("binding_id", sa.UUID(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("document_part_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_model_bindings_project_id"), table_name="model_bindings")
    op.drop_index(op.f("ix_model_bindings_model_id"), table_name="model_bindings")
    op.drop_index(op.f("ix_model_bindings_document_part_id"), table_name="model_bindings")
    op.drop_index(op.f("ix_model_bindings_document_id"), table_name="model_bindings")
    op.drop_table("model_bindings")
    job_status.drop(bind, checkfirst=True)
    job_type.drop(bind, checkfirst=True)
    binding_task.drop(bind, checkfirst=True)
