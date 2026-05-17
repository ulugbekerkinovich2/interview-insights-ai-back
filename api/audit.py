"""Audit log API — SuperAdmin uchun amallar tarixi.

GET /audit-log
  ?action=delete (optional)
  &entity_type=candidate (optional)
  &user_id=42 (optional)
  &page=1&size=50
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database
from database import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audit-log", tags=["audit"])

_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
_JWT_ALGORITHM = "HS256"


def _get_db():
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


def _require_super_admin(user: Optional[database.User] = Depends(_current_user)) -> database.User:
    if not user:
        raise HTTPException(status_code=401, detail="Не авторизован")
    if (user.role or "").strip().lower() != "superadmin":
        raise HTTPException(status_code=403, detail="Доступно только SuperAdmin")
    return user


class AuditLogOut(BaseModel):
    id: int
    user_id: Optional[int]
    user_email: Optional[str]
    user_role: Optional[str]
    action: str
    entity_type: Optional[str]
    entity_id: Optional[str]
    entity_label: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    items: List[AuditLogOut]
    total: int
    page: int
    size: int


@router.get("", response_model=AuditLogResponse)
def list_audit_log(
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    user_id: Optional[int] = None,
    page: int = 1,
    size: int = 50,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """Audit log ro'yxati. Pagination + filter qo'llab-quvvatlangan."""
    if page < 1:
        page = 1
    if size < 1 or size > 200:
        size = 50

    qs = db.query(database.AuditLog)
    if action:
        qs = qs.filter(database.AuditLog.action == action)
    if entity_type:
        qs = qs.filter(database.AuditLog.entity_type == entity_type)
    if user_id:
        qs = qs.filter(database.AuditLog.user_id == user_id)

    total = qs.count()
    items = (
        qs.order_by(database.AuditLog.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return AuditLogResponse(items=items, total=total, page=page, size=size)
