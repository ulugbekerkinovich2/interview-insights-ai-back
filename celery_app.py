"""Celery ilovasi — background job queue.

Ishlatilish
-----------
Worker ishga tushirish::

    cd backend
    ../venv/bin/celery -A celery_app worker -Q stt,rag,process \\
        --concurrency=${CELERY_WORKER_CONCURRENCY:-4} --loglevel=info

Flower monitoring UI::

    ../venv/bin/celery -A celery_app flower --port=5555

Tasklar avtomatik aniqlanadi (``tasks`` paketidan). Yangi task qo'shish uchun
``backend/tasks/`` ga modul yaratib, ``@celery_app.task`` dekoratorini ishlating.

Task signallari
---------------
``task_prerun``, ``task_postrun``, ``task_failure``, ``task_retry`` signallari
``utils/job_signals.py`` da ulanadi — ular ``database.JobRecord`` ga status
yozib boradi (audit log).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")

logger = logging.getLogger(__name__)

try:
    from celery import Celery
except ImportError as exc:  # pragma: no cover — celery o'rnatilmagan bo'lsa
    raise RuntimeError(
        "Celery o'rnatilmagan. Ishga tushiring: pip install -r backend/requirements.txt"
    ) from exc

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Har task uchun hard time-limit
STT_TIMEOUT = int(os.getenv("CELERY_TASK_STT_TIMEOUT", "120"))
RAG_TIMEOUT = int(os.getenv("CELERY_TASK_RAG_TIMEOUT", "60"))
PROCESS_TURN_TIMEOUT = int(os.getenv("CELERY_TASK_PROCESS_TURN_TIMEOUT", "300"))
MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", "2"))

celery_app = Celery(
    "ai_interview",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        "tasks.stt_tasks",
        "tasks.rag_tasks",
        "tasks.process_turn_tasks",
    ],
)

celery_app.conf.update(
    # Serializatsiya — pickle ishlatmaymiz (xavfsizlik)
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker crash bo'lsa job qayta ishga tushsin (ACK faqat task tugaganida)
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Result backend TTL: 1 soat
    result_expires=3600,
    # Queue routing — har turdagi task alohida queue da
    task_routes={
        "tasks.stt_tasks.*": {"queue": "stt"},
        "tasks.rag_tasks.*": {"queue": "rag"},
        "tasks.process_turn_tasks.*": {"queue": "process"},
    },
    # Har task turi uchun alohida time-limit
    task_annotations={
        "tasks.stt_tasks.transcribe_audio_task": {
            "time_limit": STT_TIMEOUT + 30,  # hard
            "soft_time_limit": STT_TIMEOUT,  # graceful
        },
        "tasks.rag_tasks.generate_ai_reply_task": {
            "time_limit": RAG_TIMEOUT + 30,
            "soft_time_limit": RAG_TIMEOUT,
        },
        "tasks.process_turn_tasks.process_turn_full_task": {
            "time_limit": PROCESS_TURN_TIMEOUT + 60,
            "soft_time_limit": PROCESS_TURN_TIMEOUT,
        },
    },
    # Umumiy retry politikasi (har task o'zi autoretry bilan ustun chiqaradi)
    task_default_retry_delay=10,
    task_max_retries=MAX_RETRIES,
)


def celery_enabled() -> bool:
    """Celery + Redis mavjudligini tekshiradi. FastAPI endpointlar buni
    ishlatib Celery yo'q bo'lsa eski threading.Thread ga fallback qilishi mumkin."""
    if os.getenv("CELERY_ENABLED", "true").lower() in ("false", "0", "no"):
        return False
    try:
        # Broker bilan ulanish sinovi (ping)
        with celery_app.connection_for_write() as conn:
            conn.ensure_connection(max_retries=1, timeout=2)
        return True
    except Exception as exc:
        logger.warning(f"Celery broker unavailable: {exc}")
        return False


# Task signallari — JobRecord audit loglash uchun
# Import ham import orqali task_prerun va boshqa signallarni bog'laydi.
try:
    from utils import job_signals  # noqa: F401
except Exception as exc:  # pragma: no cover
    logger.warning(f"job_signals bog'lanmadi: {exc}")
