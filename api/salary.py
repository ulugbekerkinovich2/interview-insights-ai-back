"""Salary management API.

3 ta asosiy ob'ekt:
  • SalaryGrade — lavozim × daraja → bazaviy maosh (SuperAdmin tahrirlaydi)
  • UserSalaryProfile — user'ning lavozimi (onboarding'da to'ldiriladi)
  • SalarySnapshot — har oy hisoblangan maosh (audit + tarix)

Formula (universitet o'qituvchilari):
    hourly_rate    = base_salary / 22 / 8
    percentage     = hours_worked / 176 * 100
    monthly_salary = hourly_rate * hours_worked * (hours_worked / 176)

Sarflangan soat avtomatik intervyu sessiyalaridan hisoblanadi
(candidates.answers[].duration_sec yig'indisi, oy bo'yicha).
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import database
from database import SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/salary", tags=["salary"])

_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
_JWT_ALGORITHM = "HS256"


# === DB + Auth deps =========================================================

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


def _require_authenticated(user: Optional[database.User] = Depends(_current_user)) -> database.User:
    if not user:
        raise HTTPException(status_code=401, detail="Сессия истекла или вы не авторизованы")
    return user


def _require_super_admin(user: database.User = Depends(_require_authenticated)) -> database.User:
    if (user.role or "").strip().lower() != "superadmin":
        raise HTTPException(status_code=403, detail="Доступно только SuperAdmin")
    return user


# === Pydantic schemas =======================================================

class SalaryGradeIn(BaseModel):
    position: str = Field(..., min_length=1, max_length=100)
    degree: str = Field(..., min_length=1, max_length=150)
    base_salary: int = Field(..., gt=0)


class SalaryGradeOut(BaseModel):
    id: int
    position: str
    degree: str
    base_salary: int
    updated_at: datetime.datetime

    class Config:
        from_attributes = True


class SalaryProfileIn(BaseModel):
    salary_grade_id: int = Field(..., gt=0)


class SalaryProfileOut(BaseModel):
    user_id: int
    salary_grade_id: Optional[int] = None
    onboarding_completed: bool
    position: Optional[str] = None
    degree: Optional[str] = None
    base_salary: Optional[int] = None


class SalaryCurrent(BaseModel):
    user_id: int
    user_name: str
    user_email: str
    position: str
    degree: str
    base_salary: int
    hourly_rate: float
    hours_worked: float
    percentage: float
    monthly_salary: float
    year: int
    month: int


class SalarySnapshotOut(BaseModel):
    id: int
    user_id: int
    year: int
    month: int
    position: str
    degree: str
    base_salary: int
    hours_worked: float
    hourly_rate: float
    percentage: float
    monthly_salary: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True


# === Hisoblash formulasi ====================================================

def calculate_salary(base_salary: int, hours_worked: float) -> dict:
    """Universitet o'qituvchilari formulasi."""
    hourly_rate = base_salary / 22 / 8 if base_salary > 0 else 0.0
    percentage = (hours_worked / 176.0) * 100.0 if hours_worked > 0 else 0.0
    monthly_salary = (
        hourly_rate * hours_worked * (hours_worked / 176.0) if hours_worked > 0 else 0.0
    )
    return {
        "hourly_rate": round(hourly_rate, 2),
        "percentage": round(percentage, 2),
        "monthly_salary": round(monthly_salary, 2),
    }


def _user_hours_in_month(db: Session, user_id: int, year: int, month: int) -> float:
    """Foydalanuvchining shu oydagi intervyu soatlari yig'indisi.

    Manba: candidates.answers JSON ichidagi har turn'ning duration_sec maydoni.
    """
    try:
        from sqlalchemy import text as _sa_text
        sql = _sa_text(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN ans->>'duration_sec' ~ '^[0-9.]+$' THEN (ans->>'duration_sec')::float
                    ELSE 0
                END
            ), 0) AS total_sec
            FROM candidates c,
                 jsonb_array_elements(COALESCE(c.answers, '[]'::json)::jsonb) AS ans
            WHERE c.owner_id = :uid
              AND EXTRACT(YEAR FROM c.created_at) = :y
              AND EXTRACT(MONTH FROM c.created_at) = :m
            """
        )
        row = db.execute(sql, {"uid": user_id, "y": year, "m": month}).first()
        total_sec = float(row[0] or 0)
        return round(total_sec / 3600.0, 2)
    except Exception as exc:
        logger.warning("_user_hours_in_month failed for user %s: %s", user_id, exc)
        return 0.0


def _get_or_create_profile(db: Session, user_id: int) -> database.UserSalaryProfile:
    prof = db.query(database.UserSalaryProfile).filter_by(user_id=user_id).first()
    if prof:
        return prof
    prof = database.UserSalaryProfile(user_id=user_id, onboarding_completed=False)
    db.add(prof)
    db.commit()
    db.refresh(prof)
    return prof


# === Grades CRUD ============================================================


@router.get("/grades", response_model=List[SalaryGradeOut])
def list_grades(
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_authenticated),
):
    """Barcha lavozim×daraja jadvali. Har auth user ko'ra oladi."""
    return (
        db.query(database.SalaryGrade)
        .order_by(database.SalaryGrade.position, database.SalaryGrade.degree)
        .all()
    )


@router.post("/grades", response_model=SalaryGradeOut)
def create_grade(
    payload: SalaryGradeIn,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """Yangi lavozim×daraja kombinatsiyasi."""
    grade = database.SalaryGrade(
        position=payload.position.strip(),
        degree=payload.degree.strip(),
        base_salary=payload.base_salary,
    )
    db.add(grade)
    try:
        db.commit()
        db.refresh(grade)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="Эта комбинация уже существует")
    return grade


@router.put("/grades/{grade_id}", response_model=SalaryGradeOut)
def update_grade(
    grade_id: int,
    payload: SalaryGradeIn,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    grade = db.query(database.SalaryGrade).filter_by(id=grade_id).first()
    if not grade:
        raise HTTPException(status_code=404, detail="Не найдено")
    grade.position = payload.position.strip()
    grade.degree = payload.degree.strip()
    grade.base_salary = payload.base_salary
    try:
        db.commit()
        db.refresh(grade)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="Конфликт уникальности")
    return grade


@router.delete("/grades/{grade_id}")
def delete_grade(
    grade_id: int,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    grade = db.query(database.SalaryGrade).filter_by(id=grade_id).first()
    if not grade:
        raise HTTPException(status_code=404, detail="Не найдено")
    db.delete(grade)
    db.commit()
    return {"status": "deleted"}


# === User salary profile (onboarding) =======================================


def _profile_response(db: Session, prof: database.UserSalaryProfile) -> SalaryProfileOut:
    grade = None
    if prof.salary_grade_id:
        grade = db.query(database.SalaryGrade).filter_by(id=prof.salary_grade_id).first()
    return SalaryProfileOut(
        user_id=prof.user_id,
        salary_grade_id=prof.salary_grade_id,
        onboarding_completed=bool(prof.onboarding_completed),
        position=grade.position if grade else None,
        degree=grade.degree if grade else None,
        base_salary=grade.base_salary if grade else None,
    )


@router.get("/me/profile", response_model=SalaryProfileOut)
def get_my_profile(
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Joriy user'ning salary profili. Yo'q bo'lsa avtomatik yaratiladi
    (onboarding_completed=False)."""
    prof = _get_or_create_profile(db, user.id)
    return _profile_response(db, prof)


@router.post("/me/profile", response_model=SalaryProfileOut)
def set_my_profile(
    payload: SalaryProfileIn,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_authenticated),
):
    """Onboarding: lavozim va darajani tanlash. Bir martalik majburiy qadam."""
    grade = db.query(database.SalaryGrade).filter_by(id=payload.salary_grade_id).first()
    if not grade:
        raise HTTPException(status_code=404, detail="Лавозим топилмади")
    prof = _get_or_create_profile(db, user.id)
    prof.salary_grade_id = grade.id
    prof.onboarding_completed = True
    db.commit()
    db.refresh(prof)
    return _profile_response(db, prof)


# === SuperAdmin: barcha userlarning maoshi ==================================


@router.get("/users/current", response_model=List[SalaryCurrent])
def list_current_salaries(
    year: Optional[int] = None,
    month: Optional[int] = None,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """Joriy oy uchun barcha onboarded user'larning maoshi.

    year/month berilmasa hozirgi oy ishlatiladi.
    """
    now = datetime.datetime.utcnow()
    y = year or now.year
    m = month or now.month

    profiles = (
        db.query(database.UserSalaryProfile, database.User, database.SalaryGrade)
        .join(database.User, database.User.id == database.UserSalaryProfile.user_id)
        .outerjoin(
            database.SalaryGrade,
            database.SalaryGrade.id == database.UserSalaryProfile.salary_grade_id,
        )
        .filter(database.UserSalaryProfile.onboarding_completed.is_(True))
        .all()
    )
    out: List[SalaryCurrent] = []
    for prof, u, grade in profiles:
        if not grade:
            continue
        hours = _user_hours_in_month(db, u.id, y, m)
        calc = calculate_salary(grade.base_salary, hours)
        out.append(
            SalaryCurrent(
                user_id=u.id,
                user_name=u.name or u.email,
                user_email=u.email,
                position=grade.position,
                degree=grade.degree,
                base_salary=grade.base_salary,
                hours_worked=hours,
                hourly_rate=calc["hourly_rate"],
                percentage=calc["percentage"],
                monthly_salary=calc["monthly_salary"],
                year=y,
                month=m,
            )
        )
    return out


@router.get("/users/{user_id}/current", response_model=SalaryCurrent)
def get_user_current(
    user_id: int,
    year: Optional[int] = None,
    month: Optional[int] = None,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """Bitta user uchun joriy oy hisob-kitobi."""
    u = db.query(database.User).filter_by(id=user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    prof = db.query(database.UserSalaryProfile).filter_by(user_id=user_id).first()
    if not prof or not prof.salary_grade_id:
        raise HTTPException(status_code=400, detail="Пользователь не прошёл onboarding")
    grade = db.query(database.SalaryGrade).filter_by(id=prof.salary_grade_id).first()
    if not grade:
        raise HTTPException(status_code=400, detail="Грейд удалён — обновите профиль")
    now = datetime.datetime.utcnow()
    y = year or now.year
    m = month or now.month
    hours = _user_hours_in_month(db, user_id, y, m)
    calc = calculate_salary(grade.base_salary, hours)
    return SalaryCurrent(
        user_id=u.id,
        user_name=u.name or u.email,
        user_email=u.email,
        position=grade.position,
        degree=grade.degree,
        base_salary=grade.base_salary,
        hours_worked=hours,
        hourly_rate=calc["hourly_rate"],
        percentage=calc["percentage"],
        monthly_salary=calc["monthly_salary"],
        year=y,
        month=m,
    )


# === Snapshots (tarix) ======================================================


@router.post("/snapshot", response_model=SalarySnapshotOut)
def create_snapshot(
    user_id: int,
    year: int,
    month: int,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """SuperAdmin oxirgi oy hisob-kitobini snapshot qiladi (audit/tarix uchun).

    Bir user, bir oy — bir snapshot (unique constraint). Qayta bossangiz
    409 qaytadi.
    """
    u = db.query(database.User).filter_by(id=user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    prof = db.query(database.UserSalaryProfile).filter_by(user_id=user_id).first()
    if not prof or not prof.salary_grade_id:
        raise HTTPException(status_code=400, detail="Пользователь без onboarding")
    grade = db.query(database.SalaryGrade).filter_by(id=prof.salary_grade_id).first()
    if not grade:
        raise HTTPException(status_code=400, detail="Грейд не найден")
    hours = _user_hours_in_month(db, user_id, year, month)
    calc = calculate_salary(grade.base_salary, hours)
    snap = database.SalarySnapshot(
        user_id=user_id,
        year=year,
        month=month,
        position=grade.position,
        degree=grade.degree,
        base_salary=grade.base_salary,
        hours_worked=hours,
        hourly_rate=calc["hourly_rate"],
        percentage=calc["percentage"],
        monthly_salary=calc["monthly_salary"],
    )
    db.add(snap)
    try:
        db.commit()
        db.refresh(snap)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="Snapshot уже существует")
    return snap


@router.get("/users/{user_id}/snapshots", response_model=List[SalarySnapshotOut])
def list_user_snapshots(
    user_id: int,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """User'ning tarixiy snapshot'lari (eng yangisi tepada)."""
    return (
        db.query(database.SalarySnapshot)
        .filter_by(user_id=user_id)
        .order_by(database.SalarySnapshot.year.desc(), database.SalarySnapshot.month.desc())
        .all()
    )


# === Seed function ==========================================================

SEED_GRADES = [
    ("Kafedra mudiri", "Fan doktori / Professor", 14_146_482),
    ("Kafedra mudiri", "Ph.D / Dotsent", 13_271_444),
    ("Kafedra mudiri", "Darajasiz", 11_703_662),
    ("Professor", "Fan doktori + Professor unvoni", 13_490_202),
    ("Professor", "Ph.D + Dotsent unvoni", 12_688_082),
    ("Professor", "Fan doktori yoki Professor", 12_104_722),
    ("Dotsent", "Fan doktori / Professor", 11_411_983),
    ("Dotsent", "Ph.D / Dotsent", 10_682_784),
    ("Dotsent", "Darajasiz", 9_552_521),
    ("Katta o'qituvchi", "Daraja va unvon", 9_953_583),
    ("Katta o'qituvchi", "Daraja yoki unvon", 9_370_224),
    ("Katta o'qituvchi", "Darajasiz", 8_568_102),
    ("Assistent", "Daraja va unvon", 8_786_864),
    ("Assistent", "Daraja yoki unvon", 8_203_501),
    ("Assistent", "Darajasiz", 7_620_142),
    ("O'qituvchi-stajyor", "—", 6_745_101),
]


def seed_salary_grades(db: Session) -> int:
    """Jadval bo'sh bo'lsa 16 yozuvni qo'shadi. Return: qo'shilgan soni."""
    existing = db.query(database.SalaryGrade).count()
    if existing > 0:
        return 0
    for position, degree, base_salary in SEED_GRADES:
        db.add(
            database.SalaryGrade(position=position, degree=degree, base_salary=base_salary)
        )
    db.commit()
    return len(SEED_GRADES)
