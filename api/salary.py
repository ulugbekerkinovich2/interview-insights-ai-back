"""Salary management API.

3 ta asosiy ob'ekt:
  • SalaryGrade — lavozim × daraja → bazaviy maosh (SuperAdmin tahrirlaydi)
  • UserSalaryProfile — user'ning lavozimi (onboarding'da to'ldiriladi)
  • SalarySnapshot — har oy hisoblangan maosh (audit + tarix)

Formula (universitet o'qituvchilari + data hajmi):
    hourly_rate    = base_salary / 22 / 8
    volume_ratio   = mb_input / 10.0           # 10 MB = 100%
    percentage     = volume_ratio * 100
    monthly_salary = hourly_rate * hours_worked * volume_ratio

Hisob faqat HAQIQIY PSIXOLOGIK chat'dan olinadi (chat_query_logs jadvali).
Oddiy yozishmalar (boshqa chat sessiyalari yoki kandidat intervyulari) hisobga
olinmaydi.

  • MB    = SUM(OCTET_LENGTH(query)) / 1048576 — user'ning psixologik
            so'rovlari hajmi (error IS NULL).
  • Hours = psixologik so'rovlar orasidagi vaqt yig'indisi, 30 daqiqalik
            tanaffus bilan sessiyalarga ajratiladi (har sessiya
            max 2 soat bilan cheklanadi, sessiyaning o'zi >0 bo'lsa min 5
            daqiqa hisoblanadi).
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
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
    # Shaxsiy ma'lumotlar (onboarding'da to'ldiriladi)
    phone: Optional[str] = Field(default=None, max_length=32)
    date_of_birth: Optional[str] = Field(default=None, max_length=20)  # YYYY-MM-DD
    gender: Optional[str] = Field(default=None, pattern="^(male|female)$")
    city: Optional[str] = Field(default=None, max_length=100)
    # Kasbiy ma'lumotlar
    specialization: Optional[str] = Field(default=None, max_length=200)
    years_of_experience: Optional[int] = Field(default=None, ge=0, le=70)
    education: Optional[str] = Field(default=None, max_length=2000)
    bio: Optional[str] = Field(default=None, max_length=2000)


class SalaryProfileOut(BaseModel):
    user_id: int
    salary_grade_id: Optional[int] = None
    onboarding_completed: bool
    position: Optional[str] = None
    degree: Optional[str] = None
    base_salary: Optional[int] = None
    # Shaxsiy
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    # Kasbiy
    specialization: Optional[str] = None
    years_of_experience: Optional[int] = None
    education: Optional[str] = None
    bio: Optional[str] = None


class SalaryCurrent(BaseModel):
    user_id: int
    user_name: str
    user_email: str
    position: str
    degree: str
    base_salary: int
    hourly_rate: float
    hours_worked: float
    mb_input: float
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
    mb_input: float
    percentage: float
    monthly_salary: float
    created_at: datetime.datetime

    class Config:
        from_attributes = True


# === Hisoblash formulasi ====================================================

def calculate_salary(base_salary: int, hours_worked: float, mb_input: float = 0.0) -> dict:
    """Universitet o'qituvchilari + data hajmi formulasi.

    10 MB kiritilgan data = 100%. Ya'ni:
        volume_ratio = mb_input / 10
        monthly_salary = hourly_rate * hours_worked * volume_ratio
    """
    hourly_rate = base_salary / 22 / 8 if base_salary > 0 else 0.0
    volume_ratio = (mb_input / 10.0) if mb_input > 0 else 0.0
    percentage = volume_ratio * 100.0
    monthly_salary = (
        hourly_rate * hours_worked * volume_ratio
        if hours_worked > 0 and volume_ratio > 0
        else 0.0
    )
    return {
        "hourly_rate": round(hourly_rate, 2),
        "percentage": round(percentage, 2),
        "monthly_salary": round(monthly_salary, 2),
    }


def _user_psych_activity_in_month(
    db: Session, user_id: int, year: int, month: int
) -> tuple[float, float]:
    """Psixologik chat aktivligi: (soatlar, MB).

    Manba: chat_query_logs (error IS NULL) — bu jadval faqat
    psixologik RAG so'rovlari uchun yoziladi, oddiy yozishmalar yo'q.

    Hours hisobi: so'rovlar created_at bo'yicha tartiblanadi, 30 daqiqadan
    katta tanaffus yangi sessiyani boshlaydi. Har sessiya min 5 daq /
    max 2 soat oralig'iga cheklanadi.
    """
    try:
        from sqlalchemy import text as _sa_text
        sql = _sa_text(
            """
            SELECT created_at, COALESCE(OCTET_LENGTH(query), 0) AS qbytes
            FROM chat_query_logs
            WHERE user_id = :uid
              AND error IS NULL
              AND EXTRACT(YEAR FROM created_at) = :y
              AND EXTRACT(MONTH FROM created_at) = :m
            ORDER BY created_at ASC
            """
        )
        rows = db.execute(sql, {"uid": user_id, "y": year, "m": month}).fetchall()
    except Exception as exc:
        logger.warning("_user_psych_activity_in_month (pg) failed: %s", exc)
        try:
            from sqlalchemy import text as _sa_text
            sql = _sa_text(
                """
                SELECT created_at, COALESCE(LENGTH(CAST(query AS BLOB)), 0) AS qbytes
                FROM chat_query_logs
                WHERE user_id = :uid
                  AND error IS NULL
                  AND strftime('%Y', created_at) = :y
                  AND strftime('%m', created_at) = :m
                ORDER BY created_at ASC
                """
            )
            rows = db.execute(sql, {
                "uid": user_id, "y": str(year), "m": f"{month:02d}",
            }).fetchall()
        except Exception as exc2:
            logger.warning("_user_psych_activity_in_month (sqlite) failed: %s", exc2)
            return 0.0, 0.0

    if not rows:
        return 0.0, 0.0

    total_bytes = 0
    timestamps: list[datetime.datetime] = []
    for r in rows:
        ts, qb = r[0], r[1]
        total_bytes += int(qb or 0)
        if isinstance(ts, str):
            try:
                ts = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
        if ts is not None:
            timestamps.append(ts)

    # Sessiyaga ajratish: 30 daq tanaffus → yangi sessiya
    GAP = datetime.timedelta(minutes=30)
    MIN_SESSION = datetime.timedelta(minutes=5)
    MAX_SESSION = datetime.timedelta(hours=2)
    total_sec = 0.0
    if timestamps:
        sess_start = timestamps[0]
        sess_end = timestamps[0]
        for ts in timestamps[1:]:
            if ts - sess_end > GAP:
                dur = max(sess_end - sess_start, MIN_SESSION)
                dur = min(dur, MAX_SESSION)
                total_sec += dur.total_seconds()
                sess_start = ts
            sess_end = ts
        dur = max(sess_end - sess_start, MIN_SESSION)
        dur = min(dur, MAX_SESSION)
        total_sec += dur.total_seconds()

    hours = round(total_sec / 3600.0, 2)
    mb = round(total_bytes / (1024.0 * 1024.0), 4)
    return hours, mb


def _user_hours_in_month(db: Session, user_id: int, year: int, month: int) -> float:
    """Backward-compat wrapper — faqat soat qaytaradi."""
    h, _ = _user_psych_activity_in_month(db, user_id, year, month)
    return h


def _user_mb_in_month(db: Session, user_id: int, year: int, month: int) -> float:
    """Backward-compat wrapper — faqat MB qaytaradi (psixologik chat'dan)."""
    _, mb = _user_psych_activity_in_month(db, user_id, year, month)
    return mb


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
    request: Request,
    db: Session = Depends(_get_db),
    user: database.User = Depends(_require_super_admin),
):
    grade = db.query(database.SalaryGrade).filter_by(id=grade_id).first()
    if not grade:
        raise HTTPException(status_code=404, detail="Не найдено")
    snapshot = {
        "id": grade.id,
        "position": grade.position,
        "degree": grade.degree,
        "base_salary": grade.base_salary,
    }
    label = f"{grade.position} / {grade.degree}"
    db.delete(grade)
    db.commit()
    # Audit log
    try:
        from utils.audit import log_audit
        log_audit(
            db, user,
            action="delete",
            entity_type="salary_grade",
            entity_id=str(grade_id),
            entity_label=label,
            details=snapshot,
            request=request,
        )
    except Exception:
        pass
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
        phone=prof.phone,
        date_of_birth=prof.date_of_birth,
        gender=prof.gender,
        city=prof.city,
        specialization=prof.specialization,
        years_of_experience=prof.years_of_experience,
        education=prof.education,
        bio=prof.bio,
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
    """Onboarding: lavozim, daraja va shaxsiy/kasbiy ma'lumotlar."""
    grade = db.query(database.SalaryGrade).filter_by(id=payload.salary_grade_id).first()
    if not grade:
        raise HTTPException(status_code=404, detail="Должность не найдена")
    prof = _get_or_create_profile(db, user.id)
    prof.salary_grade_id = grade.id
    prof.onboarding_completed = True

    # Shaxsiy
    if payload.phone is not None:
        prof.phone = payload.phone.strip() or None
    if payload.date_of_birth is not None:
        prof.date_of_birth = payload.date_of_birth.strip() or None
    if payload.gender is not None:
        prof.gender = payload.gender.strip() or None
    if payload.city is not None:
        prof.city = payload.city.strip() or None
    # Kasbiy
    if payload.specialization is not None:
        prof.specialization = payload.specialization.strip() or None
    if payload.years_of_experience is not None:
        prof.years_of_experience = payload.years_of_experience
    if payload.education is not None:
        prof.education = payload.education.strip() or None
    if payload.bio is not None:
        prof.bio = payload.bio.strip() or None

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
        hours, mb = _user_psych_activity_in_month(db, u.id, y, m)
        calc = calculate_salary(grade.base_salary, hours, mb)
        out.append(
            SalaryCurrent(
                user_id=u.id,
                user_name=u.name or u.email,
                user_email=u.email,
                position=grade.position,
                degree=grade.degree,
                base_salary=grade.base_salary,
                hours_worked=hours,
                mb_input=mb,
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
    hours, mb = _user_psych_activity_in_month(db, user_id, y, m)
    calc = calculate_salary(grade.base_salary, hours, mb)
    return SalaryCurrent(
        user_id=u.id,
        user_name=u.name or u.email,
        user_email=u.email,
        position=grade.position,
        degree=grade.degree,
        base_salary=grade.base_salary,
        hours_worked=hours,
        mb_input=mb,
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
    hours, mb = _user_psych_activity_in_month(db, user_id, year, month)
    calc = calculate_salary(grade.base_salary, hours, mb)
    snap = database.SalarySnapshot(
        user_id=user_id,
        year=year,
        month=month,
        position=grade.position,
        degree=grade.degree,
        base_salary=grade.base_salary,
        hours_worked=hours,
        mb_input=mb,
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


# === PDF report =============================================================

_RU_MONTHS = [
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
]

_FONT_CACHE: dict[str, str] = {}


def _register_pdf_fonts() -> tuple[str, str]:
    """Tizimdan kirillicha shriftni topib ro'yxatdan o'tkazadi.
    (regular, bold) shrift nomlarini qaytaradi.
    """
    if _FONT_CACHE:
        return _FONT_CACHE["regular"], _FONT_CACHE["bold"]
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg):
            try:
                pdfmetrics.registerFont(TTFont("SalaryF", reg))
                if os.path.exists(bold):
                    pdfmetrics.registerFont(TTFont("SalaryFB", bold))
                    _FONT_CACHE.update(regular="SalaryF", bold="SalaryFB")
                else:
                    _FONT_CACHE.update(regular="SalaryF", bold="SalaryF")
                return _FONT_CACHE["regular"], _FONT_CACHE["bold"]
            except Exception as exc:
                logger.warning("Font register failed %s: %s", reg, exc)
                continue
    _FONT_CACHE.update(regular="Helvetica", bold="Helvetica-Bold")
    return "Helvetica", "Helvetica-Bold"


def _build_snapshot_pdf(snap: database.SalarySnapshot, user: database.User) -> bytes:
    """SalarySnapshot uchun bir varaqlik hisob-kitob PDF generatsiyasi."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    fr, fb = _register_pdf_fonts()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def fmt_money(v: float) -> str:
        return f"{v:,.0f}".replace(",", " ") + " сум"

    # Header
    c.setFont(fb, 18)
    c.drawString(40, height - 50, "Расчёт заработной платы")
    c.setFont(fr, 10)
    c.setFillGray(0.4)
    period = f"{_RU_MONTHS[snap.month - 1]} {snap.year}"
    c.drawString(40, height - 68, f"Период: {period}")
    c.drawRightString(width - 40, height - 68,
                      f"Создан: {snap.created_at.strftime('%d.%m.%Y %H:%M')}")
    c.setFillGray(0)
    c.setLineWidth(0.5)
    c.line(40, height - 78, width - 40, height - 78)

    # Employee block
    y = height - 110
    c.setFont(fb, 12)
    c.drawString(40, y, "Сотрудник")
    c.setFont(fr, 11)
    y -= 18
    c.drawString(40, y, f"ФИО: {user.name or '—'}")
    y -= 16
    c.drawString(40, y, f"Email: {user.email}")
    y -= 16
    c.drawString(40, y, f"Должность: {snap.position}")
    y -= 16
    c.drawString(40, y, f"Степень / звание: {snap.degree}")

    # Calculations table
    y -= 30
    c.setFont(fb, 12)
    c.drawString(40, y, "Расчёт")
    y -= 6
    c.setLineWidth(0.3)
    c.line(40, y, width - 40, y)
    y -= 18

    rows = [
        ("Базовая ставка (месяц)", fmt_money(snap.base_salary)),
        ("Ставка в час (базовая / 22 / 8)", fmt_money(snap.hourly_rate) + "/ч"),
        ("Отработано часов", f"{snap.hours_worked:.2f} ч"),
        ("Введено данных", f"{getattr(snap, 'mb_input', 0.0):.4f} МБ"),
        ("Объём % (МБ / 10 × 100)", f"{snap.percentage:.2f}%"),
    ]
    c.setFont(fr, 11)
    for label, value in rows:
        c.drawString(50, y, label)
        c.drawRightString(width - 50, y, value)
        y -= 18

    # Formula box
    y -= 8
    c.setFillColorRGB(0.95, 0.97, 1.0)
    c.rect(40, y - 38, width - 80, 38, fill=1, stroke=0)
    c.setFillGray(0.2)
    c.setFont(fr, 9)
    c.drawString(50, y - 14,
                 "Формула:  Зарплата = ставка/час × часы × (МБ / 10)")
    c.drawString(50, y - 28,
                 f"= {snap.hourly_rate:.2f} × {snap.hours_worked:.2f} × "
                 f"{(getattr(snap, 'mb_input', 0.0) / 10.0):.4f}")
    c.setFillGray(0)
    y -= 60

    # Total
    c.setFont(fb, 14)
    c.drawString(40, y, "Итого к выплате")
    c.setFillColorRGB(0.05, 0.5, 0.3)
    c.drawRightString(width - 40, y, fmt_money(snap.monthly_salary))
    c.setFillGray(0)

    # Footer note
    c.setFont(fr, 8)
    c.setFillGray(0.5)
    c.drawString(40, 40,
                 "Учитываются только реальные психологические обращения "
                 "(chat_query_logs). Обычные переписки не считаются.")

    c.showPage()
    c.save()
    return buf.getvalue()


@router.post("/snapshot/pdf")
def create_snapshot_pdf(
    user_id: int,
    year: int,
    month: int,
    db: Session = Depends(_get_db),
    _: database.User = Depends(_require_super_admin),
):
    """Snapshot yaratadi (yoki mavjudini oladi) va PDF qaytaradi.

    Frontend `<a download>` orqali avto-yuklab oladi.
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

    snap = (
        db.query(database.SalarySnapshot)
        .filter_by(user_id=user_id, year=year, month=month)
        .first()
    )
    if not snap:
        hours, mb = _user_psych_activity_in_month(db, user_id, year, month)
        calc = calculate_salary(grade.base_salary, hours, mb)
        snap = database.SalarySnapshot(
            user_id=user_id,
            year=year,
            month=month,
            position=grade.position,
            degree=grade.degree,
            base_salary=grade.base_salary,
            hours_worked=hours,
            mb_input=mb,
            hourly_rate=calc["hourly_rate"],
            percentage=calc["percentage"],
            monthly_salary=calc["monthly_salary"],
        )
        db.add(snap)
        try:
            db.commit()
            db.refresh(snap)
        except Exception as exc:
            db.rollback()
            logger.warning("snapshot PDF create commit failed: %s", exc)
            # Mavjud bo'lsa olamiz
            snap = (
                db.query(database.SalarySnapshot)
                .filter_by(user_id=user_id, year=year, month=month)
                .first()
            )
            if not snap:
                raise HTTPException(status_code=500, detail="Не удалось создать snapshot")

    try:
        pdf_bytes = _build_snapshot_pdf(snap, u)
    except Exception as exc:
        logger.exception("PDF generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    safe_name = (u.name or u.email or f"user{user_id}").replace(" ", "_")
    filename = f"salary_{safe_name}_{year}_{month:02d}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# === Seed function ==========================================================

SEED_GRADES = [
    # (position, degree, base_salary)
    ("Заведующий кафедрой", "Доктор наук / Профессор", 14_146_482),
    ("Заведующий кафедрой", "Ph.D / Доцент", 13_271_444),
    ("Заведующий кафедрой", "Без степени", 11_703_662),
    ("Профессор", "Доктор наук + звание профессора", 13_490_202),
    ("Профессор", "Ph.D + звание доцента", 12_688_082),
    ("Профессор", "Доктор наук или Профессор", 12_104_722),
    ("Доцент", "Доктор наук / Профессор", 11_411_983),
    ("Доцент", "Ph.D / Доцент", 10_682_784),
    ("Доцент", "Без степени", 9_552_521),
    ("Старший преподаватель", "Степень и звание", 9_953_583),
    ("Старший преподаватель", "Степень или звание", 9_370_224),
    ("Старший преподаватель", "Без степени", 8_568_102),
    ("Ассистент", "Степень и звание", 8_786_864),
    ("Ассистент", "Степень или звание", 8_203_501),
    ("Ассистент", "Без степени", 7_620_142),
    ("Преподаватель-стажёр", "—", 6_745_101),
]

# Eski Uzbek nomlardan yangi Russian nomlarga moslik (production'da
# saqlangan eski yozuvlarni ko'chirish uchun bir martalik migration).
_UZ_TO_RU_MIGRATION = {
    # position eski → yangi
    "positions": {
        "Kafedra mudiri": "Заведующий кафедрой",
        "Professor": "Профессор",
        "Dotsent": "Доцент",
        "Katta o'qituvchi": "Старший преподаватель",
        "Assistent": "Ассистент",
        "O'qituvchi-stajyor": "Преподаватель-стажёр",
    },
    # degree eski → yangi
    "degrees": {
        "Fan doktori / Professor": "Доктор наук / Профессор",
        "Ph.D / Dotsent": "Ph.D / Доцент",
        "Darajasiz": "Без степени",
        "Fan doktori + Professor unvoni": "Доктор наук + звание профессора",
        "Ph.D + Dotsent unvoni": "Ph.D + звание доцента",
        "Fan doktori yoki Professor": "Доктор наук или Профессор",
        "Daraja va unvon": "Степень и звание",
        "Daraja yoki unvon": "Степень или звание",
    },
}


def _migrate_uz_to_ru(db: Session) -> int:
    """Eski Uzbek yozuvlarini Russian'ga ko'chiradi (idempotent).
    Return: yangilangan yozuvlar soni."""
    updated = 0
    grades = db.query(database.SalaryGrade).all()
    for g in grades:
        new_pos = _UZ_TO_RU_MIGRATION["positions"].get(g.position, g.position)
        new_deg = _UZ_TO_RU_MIGRATION["degrees"].get(g.degree, g.degree)
        if new_pos != g.position or new_deg != g.degree:
            g.position = new_pos
            g.degree = new_deg
            updated += 1
    if updated:
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("uz→ru migration commit failed: %s", exc)
            return 0
    return updated


def seed_salary_grades(db: Session) -> int:
    """Jadval bo'sh bo'lsa 16 yozuvni qo'shadi.
    Mavjud Uzbek yozuvlarini Russian'ga ko'chiradi (idempotent).
    Return: qo'shilgan + ko'chirilgan soni."""
    existing = db.query(database.SalaryGrade).count()
    if existing == 0:
        for position, degree, base_salary in SEED_GRADES:
            db.add(
                database.SalaryGrade(position=position, degree=degree, base_salary=base_salary)
            )
        db.commit()
        return len(SEED_GRADES)
    # Jadval bo'sh emas — eski Uzbek nomlarini Russian'ga aylantiramiz
    migrated = _migrate_uz_to_ru(db)
    if migrated > 0:
        logger.info("Migrated %d salary grade names from Uzbek to Russian", migrated)
    return migrated
