"""Celery task signallari — ``database.JobRecord`` jadvaliga audit yozish.

Har task prerun, postrun, failure va retry signallariga ulangan. Bu Celery ning
Redis result backend idan alohida, DB da doimiy audit log beradi (foydalanuvchi
AI API xarajati va xatolarni kuzatish uchun muhim).
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Optional

from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
)

logger = logging.getLogger(__name__)


def _extract_candidate_id(args: Any, kwargs: Any) -> Optional[int]:
    """Task argumentlaridan candidate_id ni ajratib olish."""
    if isinstance(kwargs, dict) and "candidate_id" in kwargs:
        cid = kwargs.get("candidate_id")
        return int(cid) if cid is not None else None
    # Argumentlarda birinchi yoki ikkinchi pozitsiyada bo'lishi mumkin — har task
    # o'zining konvensiyasi bilan chaqiradi. Agar topilmasa None qaytaramiz.
    return None


def _safe_write(callback):
    """DB yozishni xavfsiz bajaradi — xato bo'lsa log yozadi, signal bloklamaydi."""
    try:
        callback()
    except Exception as exc:
        logger.warning(f"job_signals DB write failed: {exc}")


@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **_extra):
    from database import SessionLocal, JobRecord  # lazy import (circular oldini olish)

    def _write():
        with SessionLocal() as db:
            rec = db.query(JobRecord).filter_by(task_id=task_id).first()
            if rec is None:
                rec = JobRecord(
                    task_id=task_id,
                    task_name=getattr(task, "name", "") or "",
                    candidate_id=_extract_candidate_id(args, kwargs),
                    status="running",
                    payload={"args": list(args or []), "kwargs": dict(kwargs or {})},
                    attempts=1,
                    started_at=datetime.datetime.utcnow(),
                )
                db.add(rec)
            else:
                rec.status = "running"
                rec.started_at = datetime.datetime.utcnow()
                rec.attempts = (rec.attempts or 0) + 1
            db.commit()

    _safe_write(_write)


@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, retval=None, state=None, **_extra):
    from database import SessionLocal, JobRecord

    def _write():
        with SessionLocal() as db:
            rec = db.query(JobRecord).filter_by(task_id=task_id).first()
            if rec is None:
                return
            # state "SUCCESS" bo'lsa success deb yozamiz, aks holda failed (retry holatida
            # task_retry signal allaqachon yangilagan bo'ladi).
            if state == "SUCCESS":
                rec.status = "success"
                try:
                    rec.result = retval if isinstance(retval, (dict, list, str, int, float, bool, type(None))) else str(retval)
                except Exception:
                    rec.result = None
            rec.finished_at = datetime.datetime.utcnow()
            db.commit()

    _safe_write(_write)


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, einfo=None, **_extra):
    from database import SessionLocal, JobRecord

    def _write():
        with SessionLocal() as db:
            rec = db.query(JobRecord).filter_by(task_id=task_id).first()
            if rec is None:
                return
            rec.status = "failed"
            rec.error = f"{type(exception).__name__}: {exception}"[:2000]
            rec.finished_at = datetime.datetime.utcnow()
            db.commit()

    _safe_write(_write)


@task_retry.connect
def on_task_retry(sender=None, task_id=None, reason=None, einfo=None, **_extra):
    from database import SessionLocal, JobRecord

    def _write():
        with SessionLocal() as db:
            rec = db.query(JobRecord).filter_by(task_id=task_id).first()
            if rec is None:
                return
            rec.status = "retry"
            rec.error = f"retry: {reason}"[:2000]
            db.commit()

    _safe_write(_write)
