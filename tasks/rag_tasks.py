"""RAG / LLM Celery tasks — chat uchun AI javob generatsiyasi."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.rag_tasks.generate_ai_reply_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=int(os.getenv("CELERY_MAX_RETRIES", "2")),
    acks_late=True,
)
def generate_ai_reply_task(
    self,
    user_message_id: int,
    prompt: str,
    candidate_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Chat xabariga AI javob generatsiya qilib DB ga yozadi.

    Parametrlar
    -----------
    user_message_id : int
        Foydalanuvchi yuborgan ChatMessage ID si (javob shu xabarga bog'lanadi)
    prompt : str
        LLM ga yuboriladigan to'liq prompt
    candidate_id : int | None
        Audit log uchun
    """
    import datetime as dt
    import logic
    from database import SessionLocal, ChatMessage

    try:
        ai_text = logic.ask_mistral_raw(prompt)
    except Exception as exc:
        logger.warning(f"AI reply generation failed: {exc}")
        # Retry autoretry orqali amalga oshiriladi. Agar retry limiti bitgan bo'lsa,
        # fallback javob yoziladi.
        if self.request.retries >= self.max_retries:
            ai_text = "AI сервер временно недоступен. Попробуйте позже."
        else:
            raise  # autoretry

    try:
        with SessionLocal() as db:
            ai_msg = ChatMessage(
                role="assistant",
                content=ai_text,
                timestamp=dt.datetime.now().isoformat(),
            )
            db.add(ai_msg)
            db.commit()
            db.refresh(ai_msg)
            return {"message_id": ai_msg.id, "content": ai_text[:200]}
    except Exception as exc:
        logger.error(f"AI reply DB write failed: {exc}")
        raise
