"""To'liq intervyu turini qayta ishlash Celery taski.

Pipeline::

    1. Whisper STT (audio → matn)
    2. Transcript smoothing (LLM)
    3. Voice prosody (librosa)
    4. RAG AI tahlili (Mistral + face stats + filterlar)
    5. Candidate.answers JSON ni yangilash
    6. WebSocket broadcast + Telegram notification
"""
from __future__ import annotations

import datetime
import logging
import os
import time as _time
from typing import Any, Dict, List, Optional

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.process_turn_tasks.process_turn_full_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=int(os.getenv("CELERY_MAX_RETRIES", "2")),
    acks_late=True,
)
def process_turn_full_task(
    self,
    candidate_id: int,
    turn_uid: str,
    question: str,
    audio_path: str,
    audio_url: str,
    parsed_face_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """STT + prosody + RAG pipeline ni bajaradi.

    Natija ``Candidate.answers`` dagi turn_uid ga teng element ichiga yoziladi.
    Xato bo'lganda autoretry ishga tushadi (max_retries gacha).
    """
    import logic
    from database import SessionLocal, Candidate, VisualRecord, GlobalSetting
    from sqlalchemy.orm.attributes import flag_modified

    analysis_db = SessionLocal()
    try:
        # 1. Whisper STT
        transcript = ""
        stt_ms = 0
        t0 = _time.time()
        try:
            transcript, stt_ms = logic.transcribe_audio(audio_path)
        except Exception as e:
            logger.warning(f"Whisper STT failed: {e}")
            transcript = "(Речь не распознана)"
        stt_wall_ms = int((_time.time() - t0) * 1000)

        # 1b. Transcript smoothing
        transcript_raw = transcript
        smooth_ms = 0
        t0 = _time.time()
        try:
            transcript = logic.smooth_transcript(transcript_raw) or transcript_raw
        except Exception as e:
            logger.warning(f"Transcript smoothing failed: {e}")
            transcript = transcript_raw
        smooth_ms = int((_time.time() - t0) * 1000)

        # 2. Voice prosody
        voice_raw = ""
        t0 = _time.time()
        try:
            voice_raw = logic.run_voice_profiler(audio_path)
        except Exception as e:
            logger.warning(f"Voice profiler failed: {e}")
        prosody_ms = int((_time.time() - t0) * 1000)

        # 3. Face context + emotion aggregate
        face_context = ""
        if parsed_face_stats:
            face_context = (
                f"\nДАННЫЕ ВИДЕОАНАЛИЗА ЛИЦА: Взгляд сфокусирован "
                f"{parsed_face_stats.get('gaze_focused_pct', 0)}%, отведён "
                f"{parsed_face_stats.get('gaze_away_pct', 0)}%, глаза закрыты "
                f"{parsed_face_stats.get('eyes_closed_pct', 0)}%."
            )

        try:
            from collections import Counter
            window_sec = 120
            since = datetime.datetime.utcnow() - datetime.timedelta(seconds=window_sec)
            recent_visuals = (
                analysis_db.query(VisualRecord)
                .filter(
                    VisualRecord.candidate_id == candidate_id,
                    VisualRecord.timestamp >= since,
                )
                .all()
            )
            if recent_visuals:
                emotions: List[str] = [v.emotion for v in recent_visuals if v.emotion]
                stresses: List[str] = [v.stress_level for v in recent_visuals if v.stress_level]
                dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "—"
                dominant_stress = Counter(stresses).most_common(1)[0][0] if stresses else "—"
                face_context += (
                    f"\nПОВЕДЕНИЕ И ЭМОЦИИ (по {len(recent_visuals)} видеокадрам за этот ответ): "
                    f"доминирующая эмоция — {dominant_emotion}, уровень стресса — {dominant_stress}. "
                    f"Учитывайте эти сигналы при оценке уверенности, искренности и эмоционального состояния кандидата."
                )
        except Exception as exc:
            logger.warning(f"visual aggregate failed: {exc}")

        # HR filterlari
        try:
            global_setting = analysis_db.query(GlobalSetting).filter_by(key="global_filters").first()
            global_filters = (
                global_setting.value if global_setting and isinstance(global_setting.value, list) else []
            )
            cand_for_filters = analysis_db.query(Candidate).filter_by(id=candidate_id).first()
            candidate_filters = list(cand_for_filters.filters or []) if cand_for_filters else []
            merged = [*global_filters, *candidate_filters]
            if merged:
                face_context += "\nТРЕБОВАНИЯ HR (оценивайте соответствие):\n- " + "\n- ".join(merged[:16])
        except Exception as exc:
            logger.warning(f"filter merge failed: {exc}")

        # 4. RAG AI tahlili
        rag_ai = ""
        t0 = _time.time()
        try:
            rag_ai = logic.analyze_answer(question, transcript, context=face_context)
        except Exception:
            rag_ai = "AI анализ недоступен"
        ai_ms = int((_time.time() - t0) * 1000)

        # 5. Candidate.answers yangilash
        cand = analysis_db.query(Candidate).with_for_update().filter_by(id=candidate_id).first()
        if cand:
            ans = list(cand.answers or [])
            for i, item in enumerate(ans):
                if item.get("turn_uid") == turn_uid:
                    ans[i]["answer"] = transcript
                    ans[i]["candidate_raw"] = transcript_raw
                    ans[i]["ai"] = rag_ai
                    ans[i]["voice_raw"] = voice_raw
                    ans[i]["stt_ms"] = stt_ms
                    ans[i]["stt_wall_ms"] = stt_wall_ms
                    ans[i]["smooth_ms"] = smooth_ms
                    ans[i]["prosody_ms"] = prosody_ms
                    ans[i]["ai_ms"] = ai_ms
                    ans[i]["total_ms"] = stt_wall_ms + smooth_ms + prosody_ms + ai_ms

                    # Cost tracking
                    audio_duration_sec = 0
                    try:
                        import subprocess as _sp
                        probe = _sp.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "csv=p=0", audio_path],
                            capture_output=True, text=True, timeout=5,
                        )
                        audio_duration_sec = float(probe.stdout.strip() or 0)
                    except Exception:
                        audio_duration_sec = (stt_ms / 1000) if stt_ms > 0 else 15
                    audio_min = audio_duration_sec / 60
                    deepgram_cost = audio_min * 0.0043 if logic.DEEPGRAM_API_KEY else 0
                    mistral_cost = 0.0004
                    embed_cost = 0.0001
                    turn_cost = round(deepgram_cost + mistral_cost + embed_cost, 6)
                    ans[i]["cost_usd"] = turn_cost
                    ans[i]["audio_duration_sec"] = round(audio_duration_sec, 1)
                    ans[i]["stt_provider"] = "deepgram" if logic.DEEPGRAM_API_KEY else "whisper"
                    break
            cand.answers = ans
            flag_modified(cand, "answers")
            analysis_db.commit()

        # 6. Broadcast + Telegram
        try:
            _broadcast_turn({
                "type": "TURN_RESULT",
                "candidate_id": candidate_id,
                "question": question,
                "answer": transcript,
                "ai": rag_ai,
                "voice_raw": voice_raw,
                "audio_url": audio_url,
                "turn_uid": turn_uid,
                "stt_ms": stt_ms,
                "stt_wall_ms": stt_wall_ms,
                "prosody_ms": prosody_ms,
                "ai_ms": ai_ms,
                "total_ms": stt_wall_ms + prosody_ms + ai_ms,
            })
        except Exception as exc:
            logger.debug(f"WS broadcast skipped (worker isolation): {exc}")

        try:
            from main import send_telegram_notification
            send_telegram_notification(
                f"📝 <b>Ответ проанализирован</b>\n❓ {question}\n"
                f"💬 {transcript[:100]}...\n🧠 {rag_ai[:150]}"
            )
        except Exception:
            pass

        return {
            "turn_uid": turn_uid,
            "transcript": transcript,
            "ai": rag_ai,
            "stt_ms": stt_ms,
            "ai_ms": ai_ms,
        }
    except Exception as e:
        logger.error(f"Background analysis failed: {e}")
        try:
            analysis_db.rollback()
        except Exception:
            pass
        raise
    finally:
        analysis_db.close()


def _broadcast_turn(message: Dict[str, Any]) -> None:
    """Worker jarayonida WebSocket broadcast — izolatsiya sababli cheklangan."""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            from main import manager
            loop.run_until_complete(manager.broadcast(message))
        finally:
            loop.close()
    except Exception as exc:
        raise exc
