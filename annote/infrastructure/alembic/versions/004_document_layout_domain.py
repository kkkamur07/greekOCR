"""document domain (layout) — Document, DocumentPart, Block, Line

Revision ID: 004_document_layout
Revises: 003_inference_models
Create Date: 2026-05-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004_document_layout"
down_revision: Union[str, None] = "003_inference_models"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

document_workflow = postgresql.ENUM(
    "draft", "published", "archived", name="document_workflow", create_type=False
)


def upgrade() -> None:
    document_workflow.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("workflow", document_workflow, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_documents_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_index(op.f("ix_documents_project_id"), "documents", ["project_id"], unique=False)
    op.create_table(
        "document_parts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("image_key", sa.String(length=1024), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_parts_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_parts")),
    )
    op.create_index(op.f("ix_document_parts_document_id"), "document_parts", ["document_id"], unique=False)
    op.create_table(
        "blocks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("part_id", sa.UUID(), nullable=False),
        sa.Column("box", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("manual_geometry", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["part_id"],
            ["document_parts.id"],
            name=op.f("fk_blocks_part_id_document_parts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_blocks")),
    )
    op.create_index(op.f("ix_blocks_part_id"), "blocks", ["part_id"], unique=False)
    op.create_table(
        "lines",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("part_id", sa.UUID(), nullable=False),
        sa.Column("block_id", sa.UUID(), nullable=True),
        sa.Column("baseline", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("mask", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("manual_geometry", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
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
    op.create_index(op.f("ix_lines_block_id"), "lines", ["block_id"], unique=False)
    op.create_index(op.f("ix_lines_part_id"), "lines", ["part_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_lines_part_id"), table_name="lines")
    op.drop_index(op.f("ix_lines_block_id"), table_name="lines")
    op.drop_table("lines")
    op.drop_index(op.f("ix_blocks_part_id"), table_name="blocks")
    op.drop_table("blocks")
    op.drop_index(op.f("ix_document_parts_document_id"), table_name="document_parts")
    op.drop_table("document_parts")
    op.drop_index(op.f("ix_documents_project_id"), table_name="documents")
    op.drop_table("documents")
    document_workflow.drop(op.get_bind(), checkfirst=True)
