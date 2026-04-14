import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, Boolean, create_engine, text
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

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    summary = Column(Text, nullable=True)
    status = Column(String, default="In Progress")
    access_code = Column(String, unique=True, index=True, nullable=True) # Will store secure 16-char token
    pin_hash = Column(String, nullable=True) # Hashed 6-digit PIN
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True) # The Recruiter who created this
    answers = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class VisualRecord(Base):
    __tablename__ = "visual_records"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
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

class GlobalSetting(Base):
    __tablename__ = "global_settings"
    key = Column(String, primary_key=True)
    value = Column(JSON)

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
