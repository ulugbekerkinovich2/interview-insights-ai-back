"""Role-based Knowledge Base (RAG) API.

Roles
-----
* **SuperAdmin** — full control: add/approve/train/delete, testing-mode chat
  (sources + confidence + used_chunks in the chat response).
* **Psychologist** — adds knowledge in *draft* state, sees own drafts.
  Chat results are restricted to ``approved=True`` chunks.
* **User** — chat only. No knowledge management.

Qdrant is optional at runtime: if the server is unreachable we still accept
drafts/approvals (the SQL row is the source of truth) and re-index on approval
once Qdrant becomes available via ``POST /knowledge/reindex/{doc_id}``.
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

import database
import schemas
from utils import rag_knowledge as kb
from utils import notifications as notif_svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}

# Reuse the same OAuth2 scheme token URL as the main app — the JWT payload is
# decoded here directly to avoid a circular import with ``main``.
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
_JWT_ALGORITHM = "HS256"


def _get_db():
    from database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _current_user(
    token: Optional[str] = Depends(_oauth2),
    db: Session = Depends(_get_db),
) -> Optional[database.User]:
    if not token:
        return None
    secret = os.getenv("SECRET_KEY") or "DEV_DEBUG_SECRET_ONLY_DO_NOT_USE_IN_PROD"
    try:
        payload = jwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
        email = payload.get("sub")
        if not email:
            return None
    except JWTError:
        return None
    return db.query(database.User).filter_by(email=email).first()


def _require_authenticated(user: Optional[database.User] = Depends(_current_user)) -> database.User:
    if not user:
        raise HTTPException(status_code=401, detail="Сессия истекла или вы не авторизованы")
    return user


def _require_knowledge_writer(user: database.User = Depends(_require_authenticated)):
    if not kb.can_add_knowledge(user.role):
        raise HTTPException(status_code=403, detail="Добавлять знания могут только психолог или SuperAdmin")
    return user


def _require_super_admin(user: database.User = Depends(_require_authenticated)):
    if not kb.is_super_admin(user.role):
        raise HTTPException(status_code=403, detail="Доступно только SuperAdmin")
    return user


# --- Helpers -----------------------------------------------------------------

def _serialize(doc: database.KnowledgeDocument) -> schemas.KnowledgeDocSchema:
    return schemas.KnowledgeDocSchema.model_validate(doc)


def _reindex(doc: database.KnowledgeDocument, *, db: Session) -> None:
    """(Re)index a document into Qdrant and persist the resulting flags."""
    ok, n = kb.index_document(
        doc_id=doc.id,
        title=doc.title,
        content=doc.content,
        approved=bool(doc.approved),
        category=doc.category,
        language=doc.language or "uz",
        source_type=doc.source_type or "text",
        source_name=doc.source_name,
        created_by=doc.created_by,
    )
    doc.qdrant_indexed = bool(ok)
    doc.chunks_count = int(n)
    db.commit()


def _notify_draft_submitted(db: Session, doc: database.KnowledgeDocument, author: database.User) -> None:
    notif_svc.notify_role(
        db,
        role=kb.ROLE_SUPER_ADMIN,
        title="Новый черновик базы знаний",
        message=f"{author.name or author.email}: «{doc.title}»",
        type="info",
        meta={"event": "knowledge.draft.created", "doc_id": doc.id, "author_id": author.id},
    )


def _notify_approved(db: Session, doc: database.KnowledgeDocument) -> None:
    if not doc.created_by:
        return
    notif_svc.notify_user(
        db,
        doc.created_by,
        title="Черновик подтверждён",
        message=f"«{doc.title}» подтверждён и добавлен в базу знаний.",
        type="success",
        meta={"event": "knowledge.approved", "doc_id": doc.id},
    )


def _notify_rejected(db: Session, doc: database.KnowledgeDocument, reason: Optional[str] = None) -> None:
    if not doc.created_by:
        return
    msg = f"«{doc.title}» отклонён или удалён."
    if reason:
        msg += f" Причина: {reason}"
    notif_svc.notify_user(
        db,
        doc.created_by,
        title="Черновик отклонён",
        message=msg,
        type="warning",
        meta={"event": "knowledge.rejected", "doc_id": doc.id},
    )


# --- Endpoints ---------------------------------------------------------------

@router.post("/", response_model=schemas.KnowledgeDocSchema, status_code=201)
def create_knowledge_text(
    payload: schemas.KnowledgeDocCreate,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_knowledge_writer),
):
    """Create a new knowledge document from raw text (draft state)."""
    doc = database.KnowledgeDocument(
        title=payload.title.strip(),
        content=payload.content.strip(),
        source_type="text",
        source_name=None,
        category=(payload.category or None),
        language=(payload.language or "uz"),
        approved=False,
        created_by=user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    _notify_draft_submitted(db, doc, user)
    # SuperAdmin authors can optionally get instant approval via the dedicated
    # approve endpoint — we keep create/approve as two explicit steps.
    return _serialize(doc)


@router.post("/upload", response_model=schemas.KnowledgeDocSchema, status_code=201)
async def upload_knowledge_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    language: Optional[str] = Form("uz"),
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_knowledge_writer),
):
    """Upload a .txt/.md/.pdf/.docx document. Content is parsed into text and stored as a draft."""
    filename = (file.filename or "").strip()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail=f"Неподдерживаемый формат: {ext or '?'}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой (>10 МБ)")

    text = kb.extract_text_from_upload(filename, data).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Не удалось извлечь текст из файла")

    doc = database.KnowledgeDocument(
        title=(title or filename).strip()[:300],
        content=text,
        source_type="file",
        source_name=filename,
        category=(category or None),
        language=(language or "uz"),
        approved=False,
        created_by=user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    _notify_draft_submitted(db, doc, user)
    return _serialize(doc)


@router.get("/", response_model=List[schemas.KnowledgeDocSchema])
def list_knowledge(
    approved: Optional[bool] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """List documents. Psychologists see their own drafts + all approved docs.
    SuperAdmin sees everything. Regular users are forbidden from listing."""
    if (user.role or "").lower() == kb.ROLE_USER.lower():
        raise HTTPException(status_code=403, detail="Для обычных пользователей доступ закрыт")

    q = db.query(database.KnowledgeDocument)

    if not kb.is_super_admin(user.role):
        # Psychologist: own drafts + any approved
        q = q.filter(
            (database.KnowledgeDocument.approved.is_(True))
            | (database.KnowledgeDocument.created_by == user.id)
        )

    if approved is not None:
        q = q.filter(database.KnowledgeDocument.approved.is_(bool(approved)))
    if category:
        q = q.filter(database.KnowledgeDocument.category == category)

    q = q.order_by(database.KnowledgeDocument.created_at.desc()).offset(max(0, offset)).limit(min(500, max(1, limit)))
    return [_serialize(doc) for doc in q.all()]


@router.get("/stats", response_model=schemas.KnowledgeStats)
def knowledge_stats(
    db: Session = Depends(_get_db),
    _admin: database.User = Depends(_require_super_admin),
):
    return _collect_stats(db)


@router.get("/metrics")
def knowledge_metrics(
    _admin: database.User = Depends(_require_super_admin),
):
    """RAG retrieval + embedding metrics (in-memory, process-local)."""
    from utils.rag_metrics import metrics as rag_metrics
    from utils.rag_service import embed_bucket_stats

    return {"metrics": rag_metrics.snapshot(), "rate_limit": embed_bucket_stats()}


# DIQQAT: /analytics route /{doc_id} dan OLDIN bo'lishi kerak
# (FastAPI route order — aks holda "analytics" string'i doc_id deb qabul qilinadi
# va Pydantic int_parsing xato beradi).
@router.get("/analytics")
def chat_analytics(
    days: int = 7,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Psixologik chat foydalanish statistikasi.

    Foydalanuvchi: faqat o'z so'rovlarini ko'radi.
    SuperAdmin: barcha so'rovlar (umumiy panel).
    """
    import datetime
    from sqlalchemy import func

    since = datetime.datetime.utcnow() - datetime.timedelta(days=max(1, min(days, 90)))
    q = db.query(database.ChatQueryLog).filter(database.ChatQueryLog.created_at >= since)

    if not kb.is_super_admin(user.role):
        q = q.filter(database.ChatQueryLog.user_id == user.id)

    total = q.count()
    if total == 0:
        return {
            "period_days": days,
            "total_queries": 0,
            "avg_confidence": None,
            "avg_latency_ms": None,
            "feedback": {"positive": 0, "negative": 0, "none": 0},
            "by_day": [],
            "top_queries": [],
        }

    avg_confidence = q.with_entities(func.avg(database.ChatQueryLog.confidence)).scalar()
    avg_latency = q.with_entities(func.avg(database.ChatQueryLog.latency_ms)).scalar()

    pos = q.filter(database.ChatQueryLog.feedback == "positive").count()
    neg = q.filter(database.ChatQueryLog.feedback == "negative").count()
    none = total - pos - neg

    # Sutkalik tarqalish
    by_day_rows = (
        q.with_entities(
            func.date(database.ChatQueryLog.created_at).label("day"),
            func.count(database.ChatQueryLog.id).label("count"),
            func.avg(database.ChatQueryLog.confidence).label("avg_conf"),
        )
        .group_by(func.date(database.ChatQueryLog.created_at))
        .order_by("day")
        .all()
    )
    by_day = [
        {
            "date": str(r.day) if r.day else None,
            "count": int(r.count),
            "avg_confidence": round(float(r.avg_conf), 1) if r.avg_conf else None,
        }
        for r in by_day_rows
    ]

    # Eng tez-tez beriladigan savollar (oddiy LOWER LIKE bo'yicha guruhlash)
    top_queries_rows = (
        q.with_entities(
            database.ChatQueryLog.query,
            func.count(database.ChatQueryLog.id).label("count"),
        )
        .group_by(database.ChatQueryLog.query)
        .order_by(func.count(database.ChatQueryLog.id).desc())
        .limit(10)
        .all()
    )
    top_queries = [
        {"query": r.query[:120], "count": int(r.count)} for r in top_queries_rows
    ]

    return {
        "period_days": days,
        "total_queries": total,
        "avg_confidence": round(float(avg_confidence), 1) if avg_confidence else None,
        "avg_latency_ms": int(avg_latency) if avg_latency else None,
        "feedback": {"positive": pos, "negative": neg, "none": none},
        "by_day": by_day,
        "top_queries": top_queries,
    }


@router.get("/{doc_id}", response_model=schemas.KnowledgeDocSchema)
def get_knowledge(
    doc_id: int,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    if not kb.is_super_admin(user.role):
        if (user.role or "").lower() == kb.ROLE_USER.lower():
            raise HTTPException(status_code=403, detail="Нет доступа")
        # Psychologist: own drafts + approved only
        if not doc.approved and doc.created_by != user.id:
            raise HTTPException(status_code=403, detail="К этому документу нет доступа")

    return _serialize(doc)


@router.patch("/{doc_id}/approve", response_model=schemas.KnowledgeDocSchema)
def approve_knowledge(
    doc_id: int,
    db: Session = Depends(_get_db),
    admin: database.User = Depends(_require_super_admin),
):
    """SuperAdmin approves a draft and triggers Qdrant indexing (training)."""
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    doc.approved = True
    doc.approved_by = admin.id
    doc.approved_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(doc)

    # Training step: (re)index into Qdrant. If Qdrant is down we keep the row
    # approved — reindex can be retried later.
    _reindex(doc, db=db)
    _notify_approved(db, doc)
    return _serialize(doc)


@router.patch("/{doc_id}/unapprove", response_model=schemas.KnowledgeDocSchema)
def unapprove_knowledge(
    doc_id: int,
    db: Session = Depends(_get_db),
    _admin: database.User = Depends(_require_super_admin),
):
    """Revoke approval. Keeps chunks in Qdrant but flips their ``approved`` payload."""
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    doc.approved = False
    doc.approved_by = None
    doc.approved_at = None
    db.commit()
    db.refresh(doc)

    kb.update_document_approval(doc.id, approved=False)
    _notify_rejected(db, doc, reason="Подтверждение отменено")
    return _serialize(doc)


@router.post("/reindex/{doc_id}", response_model=schemas.KnowledgeDocSchema)
def reindex_knowledge(
    doc_id: int,
    db: Session = Depends(_get_db),
    _admin: database.User = Depends(_require_super_admin),
):
    """Force re-chunk + re-embed + re-upsert. Useful after content edits or Qdrant restore."""
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")
    _reindex(doc, db=db)
    return _serialize(doc)


def _retrain_all(db: Session) -> schemas.KnowledgeRetrainReport:
    """Re-chunk + re-embed + re-upsert every approved document (synchronous)."""
    docs = (
        db.query(database.KnowledgeDocument)
        .filter(database.KnowledgeDocument.approved.is_(True))
        .order_by(database.KnowledgeDocument.id.asc())
        .all()
    )
    succeeded = 0
    failed_ids: List[int] = []
    chunks_total = 0
    for doc in docs:
        _reindex(doc, db=db)
        if doc.qdrant_indexed:
            succeeded += 1
            chunks_total += int(doc.chunks_count or 0)
        else:
            failed_ids.append(doc.id)

    return schemas.KnowledgeRetrainReport(
        attempted=len(docs),
        succeeded=succeeded,
        failed=len(failed_ids),
        chunks_total=chunks_total,
        failed_ids=failed_ids,
    )


def _job_to_schema(job: database.RetrainJob) -> schemas.RetrainJobSchema:
    total = int(job.total_docs or 0)
    processed = int(job.processed or 0)
    pct = round((processed / total) * 100, 1) if total > 0 else 0.0
    data = schemas.RetrainJobSchema.model_validate(job)
    data.progress_pct = pct
    data.failed_ids = list(job.failed_ids or [])
    return data


def _run_retrain_job(job_id: int, admin_id: int) -> None:
    """Background worker — re-embeds every approved document, updating the job row."""
    from database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(database.RetrainJob).get(job_id)
        if not job:
            return
        docs = (
            db.query(database.KnowledgeDocument)
            .filter(database.KnowledgeDocument.approved.is_(True))
            .order_by(database.KnowledgeDocument.id.asc())
            .all()
        )
        job.status = "running"
        job.total_docs = len(docs)
        job.processed = 0
        job.succeeded = 0
        job.failed = 0
        job.chunks_total = 0
        job.failed_ids = []
        db.commit()

        failed_ids: List[int] = []
        try:
            for doc in docs:
                job.current_doc_id = doc.id
                db.commit()
                try:
                    _reindex(doc, db=db)
                except Exception as exc:
                    logger.warning("retrain: reindex failed for doc_id=%s: %s", doc.id, exc)
                    doc.qdrant_indexed = False

                if doc.qdrant_indexed:
                    job.succeeded = int(job.succeeded or 0) + 1
                    job.chunks_total = int(job.chunks_total or 0) + int(doc.chunks_count or 0)
                else:
                    failed_ids.append(doc.id)
                    job.failed = len(failed_ids)
                    job.failed_ids = list(failed_ids)
                job.processed = int(job.processed or 0) + 1
                db.commit()

            job.status = "completed"
            job.current_doc_id = None
            job.finished_at = datetime.datetime.utcnow()
            db.commit()
        except Exception as exc:
            logger.exception("retrain job %s crashed", job_id)
            job.status = "failed"
            job.error = str(exc)[:2000]
            job.finished_at = datetime.datetime.utcnow()
            db.commit()

        try:
            notif_svc.notify_user(
                db,
                admin_id,
                title="Переобучение завершено" if job.status == "completed" else "Переобучение прервано",
                message=(
                    f"Попыток: {job.total_docs} • Успешно: {job.succeeded} "
                    f"• Ошибок: {job.failed} • Всего фрагментов: {job.chunks_total}"
                ),
                type="success" if (job.status == "completed" and job.failed == 0) else "warning",
                meta={
                    "event": "knowledge.retrained",
                    "job_id": job.id,
                    "status": job.status,
                    "attempted": job.total_docs,
                    "succeeded": job.succeeded,
                    "failed": job.failed,
                    "chunks_total": job.chunks_total,
                    "failed_ids": list(job.failed_ids or []),
                },
            )
        except Exception:
            logger.exception("retrain job %s: notification failed", job_id)
    finally:
        db.close()


def _active_retrain_job(db: Session) -> Optional[database.RetrainJob]:
    return (
        db.query(database.RetrainJob)
        .filter(database.RetrainJob.status.in_(("pending", "running")))
        .order_by(database.RetrainJob.id.desc())
        .first()
    )


@router.post("/reindex-all", response_model=schemas.RetrainJobSchema, status_code=202)
def reindex_all(
    background_tasks: BackgroundTasks,
    db: Session = Depends(_get_db),
    admin: database.User = Depends(_require_super_admin),
):
    """Queue a full re-embedding job. Returns the job row immediately — poll
    ``GET /knowledge/retrain/{id}`` or ``/retrain/latest`` for progress.
    """
    active = _active_retrain_job(db)
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Переобучение уже выполняется (job #{active.id}, status={active.status})",
        )

    job = database.RetrainJob(
        status="pending",
        triggered_by=admin.id,
        started_at=datetime.datetime.utcnow(),
        failed_ids=[],
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_retrain_job, job.id, admin.id)
    return _job_to_schema(job)


@router.get("/retrain/latest", response_model=Optional[schemas.RetrainJobSchema])
def retrain_latest(
    db: Session = Depends(_get_db),
    _admin: database.User = Depends(_require_super_admin),
):
    job = db.query(database.RetrainJob).order_by(database.RetrainJob.id.desc()).first()
    return _job_to_schema(job) if job else None


@router.get("/retrain/{job_id}", response_model=schemas.RetrainJobSchema)
def retrain_status(
    job_id: int,
    db: Session = Depends(_get_db),
    _admin: database.User = Depends(_require_super_admin),
):
    job = db.query(database.RetrainJob).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job не найден")
    return _job_to_schema(job)


@router.put("/{doc_id}", response_model=schemas.KnowledgeDocSchema)
def update_knowledge(
    doc_id: int,
    payload: schemas.KnowledgeDocUpdate,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Edit a document. SuperAdmin can edit anything; Psychologist only own drafts.
    If content changes on an approved doc, Qdrant is re-indexed automatically."""
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    if not kb.is_super_admin(user.role):
        if not kb.can_add_knowledge(user.role):
            raise HTTPException(status_code=403, detail="Нет доступа")
        if doc.created_by != user.id or doc.approved:
            raise HTTPException(status_code=403, detail="Редактировать можно только свои черновики")

    changed_content = False
    if payload.title is not None and payload.title.strip() != doc.title:
        doc.title = payload.title.strip()[:300]
    if payload.content is not None and payload.content.strip() != (doc.content or ""):
        doc.content = payload.content.strip()
        changed_content = True
    if payload.category is not None:
        doc.category = payload.category or None
    if payload.language is not None:
        doc.language = payload.language or "uz"

    db.commit()
    db.refresh(doc)

    # If an approved doc's content changed we must re-train its chunks;
    # otherwise only update the title/category/approved payload in Qdrant.
    if doc.approved:
        if changed_content:
            _reindex(doc, db=db)
        else:
            kb.update_document_approval(doc.id, approved=True)

    return _serialize(doc)


@router.delete("/bulk")
def bulk_delete_knowledge(
    payload: schemas.KnowledgeBulkDelete,
    db: Session = Depends(_get_db),
    admin: database.User = Depends(_require_super_admin),
):
    """SuperAdmin: delete many documents in one call. Drops Qdrant chunks too."""
    if not payload.ids:
        raise HTTPException(status_code=400, detail="Не указаны ID")

    deleted = 0
    not_found: List[int] = []
    affected_authors: Dict[int, List[Dict[str, Any]]] = {}

    for doc_id in payload.ids:
        doc = db.query(database.KnowledgeDocument).get(doc_id)
        if not doc:
            not_found.append(doc_id)
            continue

        kb.delete_document_points(doc.id)
        snap = {"doc_id": doc.id, "title": doc.title}
        author_id = doc.created_by
        db.delete(doc)
        if author_id and author_id != admin.id:
            affected_authors.setdefault(author_id, []).append(snap)
        deleted += 1
    db.commit()

    for author_id, docs in affected_authors.items():
        titles = ", ".join(f"«{d['title']}»" for d in docs[:3])
        more = f" и ещё {len(docs) - 3}" if len(docs) > 3 else ""
        notif_svc.notify_user(
            db,
            author_id,
            title="Ваши документы удалены",
            message=f"SuperAdmin удалил: {titles}{more}",
            type="warning",
            meta={"event": "knowledge.bulk_deleted", "docs": docs},
        )

    return {"deleted": deleted, "not_found": not_found}


def _collect_stats(db: Session) -> schemas.KnowledgeStats:
    from sqlalchemy import func
    from utils.rag_service import get_collection_info

    total = db.query(database.KnowledgeDocument).count()
    approved = db.query(database.KnowledgeDocument).filter(database.KnowledgeDocument.approved.is_(True)).count()
    indexed = db.query(database.KnowledgeDocument).filter(database.KnowledgeDocument.qdrant_indexed.is_(True)).count()
    chunks = db.query(func.coalesce(func.sum(database.KnowledgeDocument.chunks_count), 0)).scalar() or 0

    by_category = dict(
        db.query(database.KnowledgeDocument.category, func.count(database.KnowledgeDocument.id))
        .group_by(database.KnowledgeDocument.category)
        .all()
    )
    by_language = dict(
        db.query(database.KnowledgeDocument.language, func.count(database.KnowledgeDocument.id))
        .group_by(database.KnowledgeDocument.language)
        .all()
    )

    return schemas.KnowledgeStats(
        total=total,
        approved=approved,
        drafts=total - approved,
        indexed_in_qdrant=indexed,
        chunks_total=int(chunks),
        by_category={(k or "umumiy"): v for k, v in by_category.items()},
        by_language={(k or "?"): v for k, v in by_language.items()},
        qdrant=get_collection_info(),
    )


@router.delete("/{doc_id}", status_code=204)
def delete_knowledge(
    doc_id: int,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Документ не найден")

    # SuperAdmin can delete anything; Psychologist only their own *drafts*.
    if not kb.is_super_admin(user.role):
        if not kb.can_add_knowledge(user.role):
            raise HTTPException(status_code=403, detail="Нет доступа")
        if doc.created_by != user.id or doc.approved:
            raise HTTPException(status_code=403, detail="Удалить можно только свои черновики")

    kb.delete_document_points(doc.id)
    # Snapshot primitives before deletion so the notification still has values
    # to reference on the detached instance.
    snap_title = doc.title
    snap_author = doc.created_by
    snap_id = doc.id
    db.delete(doc)
    db.commit()

    if kb.is_super_admin(user.role) and snap_author and snap_author != user.id:
        notif_svc.notify_user(
            db,
            snap_author,
            title="Черновик отклонён",
            message=f"«{snap_title}» удалён SuperAdmin.",
            type="warning",
            meta={"event": "knowledge.rejected", "doc_id": snap_id},
        )
    return None


def _chat_save(
    db: Session, user: database.User, title: str, content: str
) -> schemas.KnowledgeChatResponse:
    doc = database.KnowledgeDocument(
        title=title[:300],
        content=content,
        source_type="text",
        approved=False,
        created_by=user.id,
        language="uz",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    _notify_draft_submitted(db, doc, user)
    answer = (
        f"✅ Черновик сохранён (ID: {doc.id}, «{doc.title}»). "
        "Ожидайте подтверждения от SuperAdmin."
    )
    return schemas.KnowledgeChatResponse(
        answer=answer, role_seen=user.role or "", action="save", data={"doc_id": doc.id}
    )


def _chat_list_drafts(
    db: Session, user: database.User, *, only_mine: bool
) -> schemas.KnowledgeChatResponse:
    q = db.query(database.KnowledgeDocument).filter(database.KnowledgeDocument.approved.is_(False))
    if only_mine:
        q = q.filter(database.KnowledgeDocument.created_by == user.id)
    drafts = q.order_by(database.KnowledgeDocument.created_at.desc()).limit(50).all()

    if not drafts:
        text = "Нет черновиков."
    else:
        lines = [f"#{d.id} — «{d.title}» ({d.category or 'общее'})" for d in drafts]
        text = f"Ожидают проверки ({len(drafts)}):\n" + "\n".join(lines)

    return schemas.KnowledgeChatResponse(
        answer=text,
        role_seen=user.role or "",
        action="my_drafts" if only_mine else "list_drafts",
        data=[
            {"doc_id": d.id, "title": d.title, "category": d.category, "created_at": d.created_at.isoformat() if d.created_at else None}
            for d in drafts
        ],
    )


def _chat_approve(db: Session, admin: database.User, doc_id: int) -> schemas.KnowledgeChatResponse:
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        return schemas.KnowledgeChatResponse(
            answer=f"❌ #{doc_id} не найден.", role_seen=admin.role or "", action="approve"
        )
    if doc.approved:
        return schemas.KnowledgeChatResponse(
            answer=f"ℹ️ #{doc.id} уже подтверждён.",
            role_seen=admin.role or "",
            action="approve",
            data={"doc_id": doc.id},
        )

    doc.approved = True
    doc.approved_by = admin.id
    doc.approved_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(doc)
    _reindex(doc, db=db)
    _notify_approved(db, doc)

    extra = ""
    if doc.qdrant_indexed:
        extra = f" В Qdrant загружено {doc.chunks_count} фрагментов."
    else:
        extra = " (Qdrant офлайн — потребуется повторная индексация.)"
    return schemas.KnowledgeChatResponse(
        answer=f"✅ #{doc.id} «{doc.title}» подтверждён.{extra}",
        role_seen=admin.role or "",
        action="approve",
        data={"doc_id": doc.id, "qdrant_indexed": doc.qdrant_indexed, "chunks": doc.chunks_count},
    )


def _chat_reject(db: Session, admin: database.User, doc_id: int) -> schemas.KnowledgeChatResponse:
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        return schemas.KnowledgeChatResponse(
            answer=f"❌ #{doc_id} не найден.", role_seen=admin.role or "", action="reject"
        )

    snap_title, snap_author, snap_id = doc.title, doc.created_by, doc.id
    kb.delete_document_points(doc.id)
    db.delete(doc)
    db.commit()

    if snap_author and snap_author != admin.id:
        notif_svc.notify_user(
            db,
            snap_author,
            title="Черновик отклонён",
            message=f"«{snap_title}» отклонён SuperAdmin.",
            type="warning",
            meta={"event": "knowledge.rejected", "doc_id": snap_id},
        )
    return schemas.KnowledgeChatResponse(
        answer=f"🗑 #{snap_id} «{snap_title}» удалён.",
        role_seen=admin.role or "",
        action="reject",
        data={"doc_id": snap_id},
    )


def _chat_reindex(db: Session, admin: database.User, doc_id: int) -> schemas.KnowledgeChatResponse:
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        return schemas.KnowledgeChatResponse(
            answer=f"❌ #{doc_id} не найден.", role_seen=admin.role or "", action="reindex"
        )
    _reindex(doc, db=db)
    status = "успешно" if doc.qdrant_indexed else "не удалось (Qdrant офлайн)"
    return schemas.KnowledgeChatResponse(
        answer=f"🔁 #{doc.id} переиндексация {status}. Фрагментов: {doc.chunks_count}.",
        role_seen=admin.role or "",
        action="reindex",
        data={"doc_id": doc.id, "qdrant_indexed": doc.qdrant_indexed, "chunks": doc.chunks_count},
    )


def _chat_edit(
    db: Session, admin: database.User, doc_id: int, title: Optional[str], content: Optional[str]
) -> schemas.KnowledgeChatResponse:
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        return schemas.KnowledgeChatResponse(
            answer=f"❌ #{doc_id} не найден.", role_seen=admin.role or "", action="edit"
        )

    changed_content = False
    if title:
        doc.title = title.strip()[:300]
    if content:
        new_content = content.strip()
        if new_content != (doc.content or ""):
            doc.content = new_content
            changed_content = True
    db.commit()
    db.refresh(doc)

    if doc.approved:
        if changed_content:
            _reindex(doc, db=db)
            note = f"Переиндексировано в Qdrant ({doc.chunks_count} фрагментов)."
        else:
            kb.update_document_approval(doc.id, approved=True)
            note = "Содержимое не изменилось — повторная индексация не требуется."
    else:
        note = "В статусе черновика — для публикации используйте команду `подтвердить`."

    return schemas.KnowledgeChatResponse(
        answer=f"✏️ #{doc.id} «{doc.title}» обновлён. {note}",
        role_seen=admin.role or "",
        action="edit",
        data={"doc_id": doc.id, "qdrant_indexed": doc.qdrant_indexed, "chunks": doc.chunks_count},
    )


def _chat_retrain(
    db: Session, admin: database.User, background_tasks: BackgroundTasks
) -> schemas.KnowledgeChatResponse:
    active = _active_retrain_job(db)
    if active:
        return schemas.KnowledgeChatResponse(
            answer=(
                f"⏳ Переобучение уже выполняется (job #{active.id}, "
                f"{active.processed}/{active.total_docs}). "
                f"Повторный запуск невозможен."
            ),
            role_seen=admin.role or "",
            action="retrain",
            data=_job_to_schema(active).model_dump(),
        )

    job = database.RetrainJob(
        status="pending",
        triggered_by=admin.id,
        started_at=datetime.datetime.utcnow(),
        failed_ids=[],
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(_run_retrain_job, job.id, admin.id)

    text = (
        f"🚂 Переобучение запущено (job #{job.id}). "
        f"Отслеживайте прогресс: GET /knowledge/retrain/{job.id}"
    )
    return schemas.KnowledgeChatResponse(
        answer=text, role_seen=admin.role or "", action="retrain", data=_job_to_schema(job).model_dump()
    )


def _chat_stats(db: Session, admin: database.User) -> schemas.KnowledgeChatResponse:
    s = _collect_stats(db)
    cat_lines = "\n".join(f"  • {k}: {v}" for k, v in sorted(s.by_category.items(), key=lambda x: -x[1])[:10])
    lang_lines = ", ".join(f"{k}:{v}" for k, v in s.by_language.items())
    text = (
        f"📊 База знаний:\n"
        f"Всего: {s.total}  |  Подтверждено: {s.approved}  |  Черновики: {s.drafts}\n"
        f"В Qdrant: {s.indexed_in_qdrant}  |  Фрагментов: {s.chunks_total}\n"
        f"Языки: {lang_lines or '—'}\n"
        f"Категории:\n{cat_lines or '  —'}"
    )
    return schemas.KnowledgeChatResponse(
        answer=text, role_seen=admin.role or "", action="stats", data=s.model_dump()
    )


def _chat_search(db: Session, user: database.User, query: str) -> schemas.KnowledgeChatResponse:
    pattern = f"%{query}%"
    q = db.query(database.KnowledgeDocument).filter(
        database.KnowledgeDocument.title.ilike(pattern)
        | database.KnowledgeDocument.content.ilike(pattern)
    )
    if not kb.is_super_admin(user.role):
        q = q.filter(
            (database.KnowledgeDocument.approved.is_(True))
            | (database.KnowledgeDocument.created_by == user.id)
        )
    docs = q.order_by(database.KnowledgeDocument.created_at.desc()).limit(25).all()
    if not docs:
        text = f"По запросу «{query}» ничего не найдено."
    else:
        lines = [f"#{d.id} — «{d.title}» ({'✅' if d.approved else '📝'})" for d in docs]
        text = f"Найдено: {len(docs)}\n" + "\n".join(lines)
    return schemas.KnowledgeChatResponse(
        answer=text,
        role_seen=user.role or "",
        action="search",
        data=[{"doc_id": d.id, "title": d.title, "approved": d.approved} for d in docs],
    )


def _chat_get(db: Session, user: database.User, doc_id: int) -> schemas.KnowledgeChatResponse:
    doc = db.query(database.KnowledgeDocument).get(doc_id)
    if not doc:
        return schemas.KnowledgeChatResponse(
            answer=f"❌ #{doc_id} не найден.", role_seen=user.role or "", action="get"
        )
    if not kb.is_super_admin(user.role):
        if (user.role or "").lower() == kb.ROLE_USER.lower():
            return schemas.KnowledgeChatResponse(
                answer="Нет доступа.", role_seen=user.role or "", action="get"
            )
        if not doc.approved and doc.created_by != user.id:
            return schemas.KnowledgeChatResponse(
                answer="К этому документу нет доступа.", role_seen=user.role or "", action="get"
            )

    status_icon = "✅" if doc.approved else "📝"
    preview = (doc.content or "")[:600] + ("…" if len(doc.content or "") > 600 else "")
    text = (
        f"{status_icon} #{doc.id} «{doc.title}»\n"
        f"Категория: {doc.category or 'общее'}  |  Язык: {doc.language or '?'}  |  "
        f"Фрагментов: {doc.chunks_count}\n\n{preview}"
    )
    return schemas.KnowledgeChatResponse(
        answer=text,
        role_seen=user.role or "",
        action="get",
        data={
            "doc_id": doc.id,
            "title": doc.title,
            "approved": doc.approved,
            "category": doc.category,
            "language": doc.language,
            "content": doc.content,
            "chunks_count": doc.chunks_count,
            "qdrant_indexed": doc.qdrant_indexed,
        },
    )


def _chat_delete_all_drafts(db: Session, admin: database.User) -> schemas.KnowledgeChatResponse:
    drafts = (
        db.query(database.KnowledgeDocument)
        .filter(database.KnowledgeDocument.approved.is_(False))
        .all()
    )
    if not drafts:
        return schemas.KnowledgeChatResponse(
            answer="Черновиков нет — удалять нечего.",
            role_seen=admin.role or "",
            action="delete_all_drafts",
        )

    affected: Dict[int, List[Dict[str, Any]]] = {}
    snapshots: List[Dict[str, Any]] = []
    for d in drafts:
        snapshots.append({"doc_id": d.id, "title": d.title})
        if d.created_by and d.created_by != admin.id:
            affected.setdefault(d.created_by, []).append({"doc_id": d.id, "title": d.title})
        kb.delete_document_points(d.id)
        db.delete(d)
    db.commit()

    for author_id, items in affected.items():
        titles = ", ".join(f"«{x['title']}»" for x in items[:3])
        more = f" и ещё {len(items) - 3}" if len(items) > 3 else ""
        notif_svc.notify_user(
            db,
            author_id,
            title="Ваши черновики удалены",
            message=f"SuperAdmin очистил все черновики: {titles}{more}",
            type="warning",
            meta={"event": "knowledge.drafts_purged", "docs": items},
        )

    return schemas.KnowledgeChatResponse(
        answer=f"🗑 Удалено черновиков: {len(drafts)}.",
        role_seen=admin.role or "",
        action="delete_all_drafts",
        data=snapshots,
    )


def _chat_status(admin: database.User) -> schemas.KnowledgeChatResponse:
    from utils.rag_service import get_collection_info
    from utils import rag_langchain as lc

    info = get_collection_info()
    backend = "LangChain" if lc.is_available() else "прямой HTTP"
    text = (
        f"Qdrant: {info.get('status')} — {info.get('collection', '?')}, "
        f"точек: {info.get('points_count', 0)}. Бэкенд чата: {backend}."
    )
    return schemas.KnowledgeChatResponse(
        answer=text, role_seen=admin.role or "", action="status", data=info
    )


@router.post("/chat", response_model=schemas.KnowledgeChatResponse)
def chat_knowledge(
    payload: schemas.KnowledgeChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Role-aware RAG chat with embedded admin commands.

    Regular questions go through the RAG pipeline. Privileged users can drive
    the knowledge base entirely from chat — see ``rag_knowledge.parse_intent``
    for supported commands (save: / approve N / reject N / reindex N / drafts /
    mydrafts / status).
    """
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Bo'sh so'rov")

    role = user.role or kb.ROLE_USER
    intent = kb.parse_intent(query, role=role)

    if intent is not None:
        if intent.action == "save":
            return _chat_save(db, user, intent.title or "", intent.content or "")
        if intent.action == "list_drafts":
            return _chat_list_drafts(db, user, only_mine=False)
        if intent.action == "my_drafts":
            return _chat_list_drafts(db, user, only_mine=True)
        if intent.action == "approve" and intent.doc_id is not None:
            return _chat_approve(db, user, intent.doc_id)
        if intent.action == "reject" and intent.doc_id is not None:
            return _chat_reject(db, user, intent.doc_id)
        if intent.action == "reindex" and intent.doc_id is not None:
            return _chat_reindex(db, user, intent.doc_id)
        if intent.action == "edit" and intent.doc_id is not None:
            return _chat_edit(db, user, intent.doc_id, intent.title, intent.content)
        if intent.action == "retrain":
            return _chat_retrain(db, user, background_tasks)
        if intent.action == "stats":
            return _chat_stats(db, user)
        if intent.action == "search" and intent.query:
            return _chat_search(db, user, intent.query)
        if intent.action == "get" and intent.doc_id is not None:
            return _chat_get(db, user, intent.doc_id)
        if intent.action == "delete_all_drafts":
            return _chat_delete_all_drafts(db, user)
        if intent.action == "status":
            return _chat_status(user)

    # Default: RAG question
    import time as _time
    t0 = _time.time()
    result = kb.run_chat(query, role=role, top_k=payload.top_k or int(os.getenv("RAG_DEFAULT_TOP_K", "8")))
    latency_ms = int((_time.time() - t0) * 1000)
    _log_chat_query(db, user, query, result, latency_ms=latency_ms, streamed=False)
    return schemas.KnowledgeChatResponse(**result)


# =============================================================================
# Streaming chat endpoint (SSE)
# =============================================================================

@router.post("/chat/stream")
async def chat_knowledge_stream(
    request: Request,
    payload: schemas.KnowledgeChatRequest,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Server-Sent Events orqali token-by-token streaming javob.

    #17 — endi client disconnect'ni aniqlaydi: foydalanuvchi Stop bossa va
    fetch'ni abort qilsa, server ham Mistral chaqiruvini to'xtatadi (token
    yoqib ketishni oldini oladi).
    """
    import json as _json
    import time as _time
    import asyncio as _asyncio
    from fastapi.responses import StreamingResponse

    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Bo'sh so'rov")

    role = user.role or kb.ROLE_USER
    intent = kb.parse_intent(query, role=role)
    if intent is not None:
        raise HTTPException(
            status_code=400,
            detail="Команды админа отправляйте через /chat (без streaming)",
        )

    user_id = user.id
    user_role = role

    async def _event_stream():
        t0 = _time.time()
        full_answer_parts: List[str] = []
        meta_info: Dict[str, Any] = {}
        done_payload: Dict[str, Any] = {}
        cancelled = False
        try:
            # Generator sync, lekin biz har iteratsiyada disconnect tekshiramiz.
            # asyncio.to_thread orqali bloklash yo'q — sync gen'ni async'da o'rab,
            # disconnect bo'lsa break qilamiz (Mistral connection avtomatik yopiladi).
            gen = kb.run_chat_stream(query, role=user_role, top_k=payload.top_k or int(os.getenv("RAG_DEFAULT_TOP_K", "8")))
            for event in gen:
                # Client disconnect tekshiruvi — har 50ms da ham bo'lsa cancel
                if await request.is_disconnected():
                    cancelled = True
                    logger.info("stream cancelled by client")
                    try:
                        gen.close()  # generator __close__ → Mistral connection yopiladi
                    except Exception:
                        pass
                    break
                if event.get("type") == "meta":
                    meta_info = event
                elif event.get("type") == "token":
                    full_answer_parts.append(event.get("text", ""))
                elif event.get("type") == "done":
                    done_payload = event
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
                # asyncio loop'ga nafas berish
                await _asyncio.sleep(0)
        except Exception as exc:
            logger.exception("stream chat failed")
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        # Stream tugagandan keyin DB ga log yozamiz
        latency_ms = int((_time.time() - t0) * 1000)
        try:
            db_session = database.SessionLocal()
            try:
                _conf_raw = meta_info.get("confidence")
                # 0.0 ham haqiqiy qiymat — None bilan aralashtirmaslik
                _conf = int(_conf_raw) if isinstance(_conf_raw, (int, float)) else None
                _persist_chat_log(
                    db_session,
                    user_id=user_id,
                    role=user_role,
                    query=query,
                    answer=done_payload.get("answer") or "".join(full_answer_parts),
                    confidence=_conf,
                    chunks_used=meta_info.get("chunks_found") or 0,
                    citations_count=len(done_payload.get("cited_indices") or []),
                    backend=meta_info.get("used_backend"),
                    latency_ms=latency_ms,
                    streamed=True,
                    error="cancelled_by_client" if cancelled else None,
                )
            finally:
                db_session.close()
        except Exception as exc:
            logger.warning(f"chat stream log save failed: {exc}")

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx buferlamasin
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# Chat query logging (analytics)
# =============================================================================

def _persist_chat_log(
    db: Session,
    *,
    user_id: Optional[int],
    role: str,
    query: str,
    answer: str,
    confidence: Optional[int],
    chunks_used: int,
    citations_count: int,
    backend: Optional[str],
    latency_ms: int,
    streamed: bool,
    error: Optional[str] = None,
) -> None:
    """ChatQueryLog jadvaliga audit yozadi (analytics uchun)."""
    try:
        log = database.ChatQueryLog(
            user_id=user_id,
            role=role,
            query=query[:5000],
            answer=(answer or "")[:10000],
            confidence=confidence,
            chunks_used=chunks_used,
            citations_count=citations_count,
            backend=backend,
            latency_ms=latency_ms,
            streamed=streamed,
            error=error,
        )
        db.add(log)
        db.commit()
    except Exception as exc:
        logger.warning(f"ChatQueryLog write failed: {exc}")
        try:
            db.rollback()
        except Exception:
            pass


def _log_chat_query(
    db: Session, user: database.User, query: str, result: Dict[str, Any],
    *, latency_ms: int, streamed: bool,
) -> None:
    """Sync chat (non-streaming) uchun audit yozish."""
    confidence = result.get("confidence")
    cited = result.get("cited_indices") or []
    chunks = result.get("used_chunks") or result.get("sources") or []
    backend = None
    if isinstance(result.get("sources"), list) and result["sources"]:
        backend = result["sources"][0].get("backend")
    _persist_chat_log(
        db,
        user_id=user.id,
        role=user.role or "User",
        query=query,
        answer=result.get("answer") or "",
        confidence=int(confidence) if confidence is not None else None,
        chunks_used=len(chunks),
        citations_count=len(cited),
        backend=backend,
        latency_ms=latency_ms,
        streamed=streamed,
    )


# =============================================================================
# Feedback endpoint — thumbs up/down
# =============================================================================


@router.post("/chat/feedback/{log_id}")
def chat_feedback(
    log_id: int,
    feedback: str,  # "positive" | "negative" | "clear"
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """ChatQueryLog yozuvi uchun foydalanuvchi feedback yangilash."""
    if feedback not in ("positive", "negative", "clear"):
        raise HTTPException(status_code=400, detail="feedback должен быть positive/negative/clear")
    log = db.query(database.ChatQueryLog).filter_by(id=log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Лог не найден")
    # Faqat o'z so'rovi yoki admin
    if log.user_id != user.id and not kb.is_super_admin(user.role):
        raise HTTPException(status_code=403, detail="Нет доступа")
    log.feedback = None if feedback == "clear" else feedback
    db.commit()
    return {"status": "ok", "feedback": log.feedback}


# =============================================================================
# Analytics endpoint /analytics yuqorida (route order — /{doc_id} dan oldin)
# =============================================================================

@router.get("/status/qdrant")
def qdrant_status(admin: database.User = Depends(_require_super_admin)):
    from utils.rag_service import get_collection_info
    return get_collection_info()
