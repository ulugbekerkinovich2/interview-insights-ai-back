"""Add login_count, last_login, created_at to users table

Revision ID: c001_user_activity
"""
from alembic import op
import sqlalchemy as sa

revision = "c001_user_activity"
down_revision = "be6245ec74a1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("login_count", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("users", sa.Column("last_login", sa.DateTime(), nullable=True))
    op.add_column("users", sa.Column("created_at", sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column("users", "login_count")
    op.drop_column("users", "last_login")
    op.drop_column("users", "created_at")
