"""Add ON DELETE cascade/set null rules to Foreign Keys.

Revision ID: c006_fk_cascades
down_revision: c005_job_records

Muammo
------
Avval FK larda ON DELETE aniqlanmagan edi — DB default ``NO ACTION`` ishlatar.
Natijada:
* User o'chirilsa, uning candidates/notifications/knowledge_documents orfan qoladi
* Candidate o'chirilsa, uning visual_records DB da qoladi (storage va RAM oqimi)
* Bu GDPR "o'ng'aytirish huquqi" uchun muhim muammo

Qarorlar
--------
* ``candidates.owner_id`` → SET NULL (nomzod saqlanadi, owner audit yo'qoladi)
* ``visual_records.candidate_id`` → CASCADE (kandidat bilan birga kadrlar ham)
* ``notifications.user_id`` → CASCADE (shaxsiy notifikatsiya foydasiz)
* ``knowledge_documents.created_by/approved_by`` → SET NULL (hujjat audit)
* ``retrain_jobs.triggered_by`` → SET NULL (job audit)
* ``job_records.candidate_id`` → SET NULL (Celery audit)
"""
from alembic import op
import sqlalchemy as sa

revision = "c006_fk_cascades"
down_revision = "c005_job_records"
branch_labels = None
depends_on = None


# Har FK uchun: (table, column, referenced_table, referenced_column, constraint_name, ondelete)
_CASCADES = [
    ("candidates", "owner_id", "users", "id", "fk_candidates_owner_id_users", "SET NULL"),
    ("visual_records", "candidate_id", "candidates", "id", "fk_visual_records_candidate_id_candidates", "CASCADE"),
    ("notifications", "user_id", "users", "id", "fk_notifications_user_id_users", "CASCADE"),
    ("knowledge_documents", "created_by", "users", "id", "fk_knowledge_documents_created_by_users", "SET NULL"),
    ("knowledge_documents", "approved_by", "users", "id", "fk_knowledge_documents_approved_by_users", "SET NULL"),
    ("retrain_jobs", "triggered_by", "users", "id", "fk_retrain_jobs_triggered_by_users", "SET NULL"),
    ("job_records", "candidate_id", "candidates", "id", "fk_job_records_candidate_id_candidates", "SET NULL"),
]


def _drop_existing_fks(table: str, column: str) -> None:
    """Mavjud FK constrainti nomi noaniq bo'lishi mumkin — inspectorda topamiz."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys(table):
        if fk.get("constrained_columns") == [column]:
            name = fk.get("name")
            if name:
                try:
                    op.drop_constraint(name, table, type_="foreignkey")
                except Exception:
                    pass


def upgrade():
    # SQLite FK lar ALTER bilan o'zgarmaydi — bu migratsiya faqat PostgreSQL uchun
    # xavfsiz. SQLite dev da ``init_db()`` modellardan to'g'ri schema yaratadi.
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite: schema recreate kerak — dev uchun init_db() yetarli
        return

    for table, column, ref_table, ref_col, new_name, ondelete in _CASCADES:
        _drop_existing_fks(table, column)
        op.create_foreign_key(
            new_name, table, ref_table, [column], [ref_col],
            ondelete=ondelete,
        )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    for table, column, ref_table, ref_col, new_name, _ondelete in _CASCADES:
        try:
            op.drop_constraint(new_name, table, type_="foreignkey")
        except Exception:
            pass
        # ondelete yo'q holatga qaytarish
        op.create_foreign_key(
            f"{new_name}_no_cascade", table, ref_table, [column], [ref_col],
        )
