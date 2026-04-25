"""Add job_records table for Celery task audit logging.

Revision ID: c005_job_records
"""
from alembic import op
import sqlalchemy as sa

revision = "c005_job_records"
down_revision = "c004_candidate_filters"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "job_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column(
            "candidate_id",
            sa.Integer(),
            sa.ForeignKey("candidates.id"),
            nullable=True,
        ),
        sa.Column("status", sa.String(), server_default="queued", nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_unique_constraint("uq_job_records_task_id", "job_records", ["task_id"])
    op.create_index("ix_job_records_task_id", "job_records", ["task_id"])
    op.create_index("ix_job_records_task_name", "job_records", ["task_name"])
    op.create_index("ix_job_records_candidate_id", "job_records", ["candidate_id"])
    op.create_index("ix_job_records_status", "job_records", ["status"])
    op.create_index("ix_job_records_created_at", "job_records", ["created_at"])


def downgrade():
    op.drop_index("ix_job_records_created_at", table_name="job_records")
    op.drop_index("ix_job_records_status", table_name="job_records")
    op.drop_index("ix_job_records_candidate_id", table_name="job_records")
    op.drop_index("ix_job_records_task_name", table_name="job_records")
    op.drop_index("ix_job_records_task_id", table_name="job_records")
    op.drop_constraint("uq_job_records_task_id", "job_records", type_="unique")
    op.drop_table("job_records")
