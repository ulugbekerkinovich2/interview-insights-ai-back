"""Add chat_query_logs for psychologist chat analytics.

Revision ID: c007_chat_query_logs
"""
from alembic import op
import sqlalchemy as sa

revision = "c007_chat_query_logs"
down_revision = "c006_fk_cascades"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chat_query_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Integer(), nullable=True),
        sa.Column("chunks_used", sa.Integer(), server_default="0"),
        sa.Column("citations_count", sa.Integer(), server_default="0"),
        sa.Column("backend", sa.String(), nullable=True),
        sa.Column("feedback", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("streamed", sa.Boolean(), server_default="false"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_chat_query_logs_user_id", "chat_query_logs", ["user_id"])
    op.create_index("ix_chat_query_logs_role", "chat_query_logs", ["role"])
    op.create_index("ix_chat_query_logs_feedback", "chat_query_logs", ["feedback"])
    op.create_index("ix_chat_query_logs_created_at", "chat_query_logs", ["created_at"])


def downgrade():
    op.drop_index("ix_chat_query_logs_created_at", table_name="chat_query_logs")
    op.drop_index("ix_chat_query_logs_feedback", table_name="chat_query_logs")
    op.drop_index("ix_chat_query_logs_role", table_name="chat_query_logs")
    op.drop_index("ix_chat_query_logs_user_id", table_name="chat_query_logs")
    op.drop_table("chat_query_logs")
