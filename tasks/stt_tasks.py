"""Speech-to-text Celery tasks.

Audio faylni Whisper/Deepgram bilan transkripsiya qiladi. Natija::

* ``GlobalSetting[stt_result_{task_id}]`` ga yoziladi (frontend polling uchun —
  backward-compatible)
* WebSocket orqali ``STT_RESULT`` xabari broadcast qilinadi

Retry logikasi: tarmoq/API xatolari uchun 2 marta qayta urinish, exponential
backoff (10s, 20s).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.stt_tasks.transcribe_audio_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=int(os.getenv("CELERY_MAX_RETRIES", "2")),
    acks_late=True,
)
def transcribe_audio_task(
    self,
    audio_path: str,
    audio_url: str,
    candidate_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Audio faylni transkripsiya qiladi.

    Parametrlar
    -----------
    audio_path : str
        Audio fayl serverda (masalan /backend/media/audio/xxx.webm)
    audio_url : str
        Frontend ga qaytariladigan URL (/media/audio/xxx.webm)
    candidate_id : int | None
        Audit log uchun (JobRecord signalida ishlatiladi)

    Qaytarish
    ---------
    dict: ``{text, elapsed_ms, audio_url, status}``
    """
    import logic  # lazy import (Celery worker start paytida og'ir modullarni yuklamaslik)
    from database import SessionLocal, GlobalSetting

    task_id = self.request.id
    try:
        text, elapsed_ms = logic.transcribe_audio(audio_path)
    except logic.TranscriptionError as exc:
        # Bu tiklanmas xato (masalan, audio buzilgan) — retry qilmaymiz
        logger.warning(f"STT non-retryable error task={task_id}: {exc}")
        _write_stt_setting(task_id, audio_url, text="", error=str(exc), status="error")
        _broadcast_stt({"type": "STT_RESULT", "task_id": task_id, "error": str(exc),
                        "audio_url": audio_url, "status": "error"})
        return {"text": "", "elapsed_ms": 0, "audio_url": audio_url, "status": "error", "error": str(exc)}

    result = {"text": text, "elapsed_ms": elapsed_ms, "audio_url": audio_url, "status": "done"}
    _write_stt_setting(task_id, audio_url, text=text, elapsed_ms=elapsed_ms, status="done")
    _broadcast_stt({"type": "STT_RESULT", "task_id": task_id, "text": text,
                    "audio_url": audio_url, "elapsed_ms": elapsed_ms})
    return result


def _write_stt_setting(task_id: str, audio_url: str, *, text: str = "",
                        elapsed_ms: int = 0, error: Optional[str] = None,
                        status: str = "done") -> None:
    """Backward-compat: frontend hozirgi ``stt_result_{task_id}`` poll mexanizmini ishlatadi."""
    from database import SessionLocal, GlobalSetting
    value: Dict[str, Any] = {
        "text": text,
        "elapsed_ms": elapsed_ms,
        "audio_url": audio_url,
        "status": status,
    }
    if error:
        value["error"] = error
    try:
        with SessionLocal() as db:
            existing = db.query(GlobalSetting).filter_by(key=f"stt_result_{task_id}").first()
            if existing:
                existing.value = value
            else:
                db.add(GlobalSetting(key=f"stt_result_{task_id}", value=value))
            db.commit()
    except Exception as exc:
        logger.warning(f"STT setting write failed: {exc}")


def _broadcast_stt(message: Dict[str, Any]) -> None:
    """WebSocket broadcast — Celery worker sync kontekstidan ishlaydi."""
    try:
        import asyncio
        # Worker jarayonida asyncio loop yo'q — yangisini yaratamiz
        loop = asyncio.new_event_loop()
        try:
            from main import manager
            loop.run_until_complete(manager.broadcast(message))
        finally:
            loop.close()
    except Exception as exc:
        # Worker va main.py alohida jarayonlarda bo'lishi mumkin — bu holatda
        # broadcast main.py orqali emas, Redis pub/sub orqali amalga oshirilishi
        # kerak. Hozircha faqat log yozamiz (frontend polling bilan ishlaydi).
        logger.debug(f"STT WS broadcast skipped (worker isolation): {exc}")
