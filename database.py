import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean, UniqueConstraint, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_SQLITE_URL = f"sqlite:///{BACKEND_DIR / 'app.db'}"
DATABASE_URL = os.getenv("DATABASE_URL") or DEFAULT_SQLITE_URL


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _create_engine(database_url: str):
    engine_kwargs = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Production-safe defaults for PostgreSQL/MySQL under concurrent frame/audio traffic.
        engine_kwargs.update(
            {
                "pool_size": _env_int("DB_POOL_SIZE", 10),
                "max_overflow": _env_int("DB_MAX_OVERFLOW", 20),
                "pool_timeout": _env_int("DB_POOL_TIMEOUT", 30),
                "pool_recycle": _env_int("DB_POOL_RECYCLE", 1800),
                "pool_use_lifo": True,
            }
        )
    return create_engine(database_url, **engine_kwargs)


engine = _create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    is_enabled = Column(Boolean, default=True)
    description = Column(String)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String, default="Recruiter") # SuperAdmin, Recruiter, Psychologist
    is_active = Column(Boolean, default=True)
    login_count = Column(Integer, default=0)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    summary = Column(Text, nullable=True)
    status = Column(String, default="In Progress")
    access_code = Column(String, unique=True, index=True, nullable=True) # Will store secure 16-char token
    pin_hash = Column(String, nullable=True) # Hashed 6-digit PIN
    # User-friendly ID format: YYYYMMLNNNN (11 belgi)
    # Misol: 202604A0001 — 2026-yil 04-oy A-guruh 0001-nomzod
    # A-guruh 9999 ga to'lgach B'ga o'tadi, va h.k. A-Z gacha (oy ichida ~260K).
    # Eski yozuvlar uchun nullable; lazy backfill create_candidate'da bajariladi.
    # Eski 8-belgili formatdagi yozuvlar ham mosligicha qoladi (length 11
    # ularni ham sig'diradi).
    display_id = Column(String(11), unique=True, index=True, nullable=True)
    # User o'chirilsa — nomzodlar qoladi lekin owner_id NULL ga o'rnatiladi
    # (audit saqlaydi, lekin ma'lumot yo'qolmaydi)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    answers = Column(JSON, default=list)
    filters = Column(JSON, default=list)  # Per-candidate HR requirements (list of strings)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Soft delete — tasodifan o'chirilgan kandidatlar 30 kun saqlanadi
    # va admin paneldan tiklash imkoni bor. NULL = aktiv.
    deleted_at = Column(DateTime, nullable=True, index=True)
    deleted_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class VisualRecord(Base):
    __tablename__ = "visual_records"
    id = Column(Integer, primary_key=True, index=True)
    # Candidate o'chirilsa — uning barcha video kadrlari ham o'chiriladi (GDPR + tozalik)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"))
    emotion = Column(String)
    stress_level = Column(String)
    notes = Column(Text)
    image_url = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(String) # Can be improved to DateTime later

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    # User o'chirilsa — uning shaxsiy notifikatsiyalari o'chiriladi
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title = Column(String)
    message = Column(Text)
    type = Column(String, default="info")  # info, success, warning, error
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class GlobalSetting(Base):
    __tablename__ = "global_settings"
    key = Column(String, primary_key=True)
    value = Column(JSON)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_type = Column(String, default="text")  # "text" | "file"
    source_name = Column(String, nullable=True)
    category = Column(String, nullable=True, index=True)
    language = Column(String, default="uz", index=True)
    approved = Column(Boolean, default=False, index=True)
    # User o'chirilsa — hujjat saqlanib qoladi (audit), faqat yaratuvchi NULL
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    chunks_count = Column(Integer, default=0)
    qdrant_indexed = Column(Boolean, default=False)


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"
    id = Column(Integer, primary_key=True, index=True)
    # pending | running | completed | failed
    status = Column(String, default="pending", index=True)
    # User o'chirilsa audit saqlanadi (SET NULL)
    triggered_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    total_docs = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    succeeded = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    chunks_total = Column(Integer, default=0)
    current_doc_id = Column(Integer, nullable=True)
    failed_ids = Column(JSON, default=list)
    error = Column(Text, nullable=True)


class SalaryGrade(Base):
    """Lavozim x daraja kombinatsiyasi bo'yicha bazaviy maosh.

    SuperAdmin tahrirlaydi. Startup'da 16 ta yozuv avtomatik seed qilinadi
    (universitet o'qituvchilarining standart jadvalisi). Yangi kombinatsiyalar
    qo'shilishi mumkin.

    Maosh formulasi:
        hourly_rate = base_salary / 22 / 8
        monthly_salary = hourly_rate * hours_worked * (hours_worked / 176)
                       = base_salary / 22 / 8 * hours_worked * hours_worked / 176
    """
    __tablename__ = "salary_grades"

    id = Column(Integer, primary_key=True, index=True)
    position = Column(String(100), nullable=False)   # "Kafedra mudiri", "Professor", ...
    degree = Column(String(150), nullable=False)     # "Fan doktori / Professor", ...
    base_salary = Column(Integer, nullable=False)    # so'mda (masalan 14146482)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("position", "degree", name="uq_salary_grade_position_degree"),
    )


class UserSalaryProfile(Base):
    """Foydalanuvchi (psixolog)'ning to'liq profili.

    Birinchi login'da majburiy onboarding orqali to'ldiriladi. Lavozim va
    daraja SalaryGrade'dagi kombinatsiyaga ishora qiladi (bazaviy maosh
    o'sha yerda saqlanadi). Shaxsiy va kasbiy ma'lumotlar ham shu yerda —
    SuperAdmin keyinchalik hisobot uchun ko'ra oladi.
    """
    __tablename__ = "user_salary_profiles"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    salary_grade_id = Column(Integer, ForeignKey("salary_grades.id", ondelete="SET NULL"), nullable=True, index=True)
    onboarding_completed = Column(Boolean, default=False, index=True)

    # === Shaxsiy ma'lumotlar ===
    phone = Column(String(32), nullable=True)
    date_of_birth = Column(String(20), nullable=True)   # ISO YYYY-MM-DD
    gender = Column(String(10), nullable=True)          # "male" | "female"
    city = Column(String(100), nullable=True)

    # === Kasbiy ma'lumotlar ===
    specialization = Column(String(200), nullable=True)  # masalan "Семейный психолог"
    years_of_experience = Column(Integer, nullable=True)
    education = Column(Text, nullable=True)              # vuz + yil (multi-line)
    bio = Column(Text, nullable=True)                    # o'zi haqida qisqacha

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class SalarySnapshot(Base):
    """Har oy uchun maosh snapshot — audit va tarix uchun.

    SuperAdmin "Hisobla" tugmasini bosganda yaratiladi yoki cron orqali oy
    oxirida avtomatik. Bir user — bir oy — bir snapshot (unique constraint).
    """
    __tablename__ = "salary_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False)  # 1-12
    # Snapshot momentidagi qiymatlar (keyin grade o'zgarsa ham bu yerda saqlanadi)
    position = Column(String(100), nullable=False)
    degree = Column(String(150), nullable=False)
    base_salary = Column(Integer, nullable=False)
    hours_worked = Column(Float, nullable=False, default=0.0)
    hourly_rate = Column(Float, nullable=False, default=0.0)
    percentage = Column(Float, nullable=False, default=0.0)   # 0-100 (MB/10 * 100)
    mb_input = Column(Float, nullable=False, default=0.0)     # user'ning chat xabarlari hajmi (MB)
    monthly_salary = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("user_id", "year", "month", name="uq_salary_snapshot_user_month"),
    )


class ChatSession(Base):
    """Foydalanuvchi chat sessiyasi — ChatGPT'dagi suhbat thread analogi.

    Har user uchun bir nechta sessiya bo'lishi mumkin (sidebar'da ro'yxat).
    title — birinchi user xabaridan avtomatik chiqariladi yoki qo'lda
    o'zgartiriladi. updated_at — yangi xabar qo'shilganda yangilanadi
    (sidebar'da eng yangisi yuqorida bo'lishi uchun).
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="Новый чат")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, index=True)


class ChatSessionMessage(Base):
    """Sessiya ichidagi alohida xabar (user yoki assistant).

    Har xabar ChatSession bilan FK orqali bog'langan. Sessiya o'chirilsa
    barcha xabarlari ham CASCADE bilan o'chadi. sources/cited_indices —
    assistant javoblari uchun ishlatiladi (RAG metadata).
    """
    __tablename__ = "chat_session_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    confidence = Column(Integer, nullable=True)  # 0-100, faqat assistant
    sources = Column(JSON, nullable=True)        # [{ index, doc_id, title }, ...]
    cited_indices = Column(JSON, nullable=True)  # [1, 2, 3]
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class ChatQueryLog(Base):
    """Psixologik chat query'larining audit logi.

    Har RAG so'rov uchun yoziladi — analytics, sifat tahlili va xarajat
    nazorati uchun. Foydalanuvchi feedback'i (👍/👎) keyinchalik PATCH
    endpointi orqali yangilanadi.
    """
    __tablename__ = "chat_query_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    role = Column(String, index=True)                    # SuperAdmin / Psychologist / User
    query = Column(Text, nullable=False)
    answer = Column(Text)
    confidence = Column(Integer, nullable=True)          # 0-100
    chunks_used = Column(Integer, default=0)             # Necha chunk ishlatildi
    citations_count = Column(Integer, default=0)         # Necha citation [N] LLM da ishlatildi
    backend = Column(String)                             # "langchain" | "direct"
    feedback = Column(String, nullable=True, index=True) # "positive" | "negative" | NULL
    latency_ms = Column(Integer)                         # Total response time
    streamed = Column(Boolean, default=False)            # Stream orqali yuborilganmi
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class JobRecord(Base):
    """Celery task audit jadvali.

    Celery o'zining result backend (Redis) da vaqtinchalik natijalarni saqlaydi,
    lekin audit va debugging uchun ushbu jadvalga persistent yozuv yozamiz.
    Task signallari (`task_prerun`, `task_postrun`, `task_failure`) orqali
    yangilanadi. Server restartida `status=running` qolgan yozuvlar startup
    hookda `failed` ga o'zgartiriladi (stale cleanup).
    """
    __tablename__ = "job_records"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True, nullable=False)
    task_name = Column(String, index=True, nullable=False)
    # Candidate o'chirilsa — job yozuvi saqlanadi (audit), candidate_id NULL bo'ladi
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="SET NULL"), index=True, nullable=True)
    # queued | running | success | failed | retry
    status = Column(String, default="queued", index=True, nullable=False)
    payload = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class AuditLog(Base):
    """Audit log — har xil muhim amallar (delete, update, create, login)
    kim tomonidan qachon qilinganini saqlaydi.

    Davlat tashkilotlarida regulyator talablari uchun majburiy.
    Frontend `/audit` sahifa orqali SuperAdmin ko'rishi mumkin.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    # Kim (user o'chirilsa SET NULL — log saqlanadi)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    user_email = Column(String, nullable=True, index=True)   # log'da ko'rsatish uchun snapshot
    user_role = Column(String, nullable=True)                 # SuperAdmin/Psychologist/User
    # Nima qilingan
    action = Column(String, nullable=False, index=True)       # "delete", "update", "create", "login", "logout"
    entity_type = Column(String, nullable=True, index=True)   # "candidate", "user", "salary_grade", "chat_session"
    entity_id = Column(String, nullable=True, index=True)     # ID — text (turli xil bo'lishi mumkin)
    entity_label = Column(String, nullable=True)              # "Иван Петров", "Профессор" — human-readable
    # Kontekst (optional)
    details = Column(JSON, nullable=True)                     # qo'shimcha ma'lumot (before/after, IP, va h.k.)
    ip_address = Column(String, nullable=True)
    # Qachon
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def check_database_connection():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def get_database_metadata():
    return {
        "url": DATABASE_URL,
        "dialect": engine.dialect.name,
    }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
