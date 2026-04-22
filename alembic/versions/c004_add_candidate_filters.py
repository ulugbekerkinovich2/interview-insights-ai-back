"""Add per-candidate filters JSON column.

Revision ID: c004_candidate_filters
"""
from alembic import op
import sqlalchemy as sa

revision = "c004_candidate_filters"
down_revision = "c003_retrain_jobs"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "candidates",
        sa.Column("filters", sa.JSON(), nullable=True),
    )
    op.execute("UPDATE candidates SET filters = '[]'::json WHERE filters IS NULL")


def downgrade():
    op.drop_column("candidates", "filters")
