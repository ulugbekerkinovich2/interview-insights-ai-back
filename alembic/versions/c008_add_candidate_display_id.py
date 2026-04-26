"""add Candidate.display_id column for user-friendly IDs (YYMMNNNN format).

Revision ID: c008_display_id
Revises: c007_chat_query_logs
Create Date: 2026-04-27

Foydalanuvchi-do'st ID format: YYMMNNNN (masalan 26040001 — 2026-yil 04-oy
0001-nomzod). Yaratilish paytida avtomatik to'ldiradi (main.py:create_candidate).

Eski yozuvlar uchun lazy backfill: column nullable, agar UI'da display_id YO'Q
bo'lsa hozirgi `id` numerik ko'rinadi (frontend fallback).
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c008_display_id"
down_revision = "c007_chat_query_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "candidates",
        sa.Column("display_id", sa.String(length=8), nullable=True),
    )
    op.create_index(
        "ix_candidates_display_id",
        "candidates",
        ["display_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_candidates_display_id", table_name="candidates")
    op.drop_column("candidates", "display_id")
