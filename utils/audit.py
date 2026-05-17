"""Audit log helper — har muhim amal uchun yozuv qo'shish.

Foydalanish (har router endpoint'da):

    from utils.audit import log_audit

    log_audit(
        db, user,
        action="delete",
        entity_type="candidate",
        entity_id=str(candidate.id),
        entity_label=candidate.name,
        request=request,
    )

Davlat regulyator talablariga ko'ra muhim har bir amal yoziladi
(login, logout, delete, update, create, role change, va h.k.).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Request
from sqlalchemy.orm import Session

import database

logger = logging.getLogger(__name__)


def log_audit(
    db: Session,
    user: Optional[database.User],
    *,
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    entity_label: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> None:
    """Audit log yozuvi qo'shish. Xato bo'lsa sukut (asosiy oqimni
    to'xtatmaslik kerak)."""
    try:
        ip = None
        if request is not None:
            # X-Forwarded-For (proxy orqali) — birinchi IP
            xff = request.headers.get("x-forwarded-for")
            if xff:
                ip = xff.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else None

        entry = database.AuditLog(
            user_id=user.id if user else None,
            user_email=user.email if user else None,
            user_role=getattr(user, "role", None),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_label=entity_label,
            details=details,
            ip_address=ip,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.warning("audit log failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
