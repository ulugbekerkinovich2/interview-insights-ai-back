"""Add retrain_jobs table for RAG re-embedding progress tracking.

Revision ID: c003_retrain_jobs
"""
from alembic import op
import sqlalchemy as sa

revision = "c003_retrain_jobs"
down_revision = "c002_notifications"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "retrain_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("status", sa.String(), server_default="pending", nullable=False),
        sa.Column("triggered_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("total_docs", sa.Integer(), server_default="0"),
        sa.Column("processed", sa.Integer(), server_default="0"),
        sa.Column("succeeded", sa.Integer(), server_default="0"),
        sa.Column("failed", sa.Integer(), server_default="0"),
        sa.Column("chunks_total", sa.Integer(), server_default="0"),
        sa.Column("current_doc_id", sa.Integer(), nullable=True),
        sa.Column("failed_ids", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_retrain_jobs_status", "retrain_jobs", ["status"])


def downgrade():
    op.drop_index("ix_retrain_jobs_status", table_name="retrain_jobs")
    op.drop_table("retrain_jobs")
