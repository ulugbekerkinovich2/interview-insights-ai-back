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


def _safe_prosody(audio_path: str) -> str:
    """Prosody analiz — vaqt o'lchaydi, future.result() ga elapsed_ms qo'shadi."""
    import logic as _logic
    t0 = _time.time()
    try:
        result = _logic.run_voice_profiler(audio_path)
    except Exception as e:
        logger.warning(f"Voice profiler error: {e}")
        result = ""
    return result


def _safe_smooth(transcript_raw: str, enabled: bool) -> str:
    """Transcript smoothing wrapper — fail-safe (xato bo'lsa raw qaytaradi)."""
    if not enabled or not transcript_raw or transcript_raw == "(Речь не распознана)":
        return transcript_raw
    import logic as _logic
    try:
        return _logic.smooth_transcript(transcript_raw) or transcript_raw
    except Exception as e:
        logger.warning(f"Transcript smoothing failed: {e}")
        return transcript_raw


def _safe_rag(question: str, transcript: str, face_context: str) -> str:
    """RAG AI tahlil wrapper — fail-safe."""
    import logic as _logic
    try:
        return _logic.analyze_answer(question, transcript, context=face_context)
    except Exception as e:
        logger.warning(f"RAG analyze_answer failed: {e}")
        return "AI анализ недоступен"


def _build_face_context(
    parsed_face_stats: Optional[Dict[str, Any]],
    candidate_id: int,
    analysis_db,
) -> str:
    """Face stats + visual_records + HR filters'ni RAG context'iga aylantiradi.

    RAG'dan oldin tezda qurib bo'linadi (faqat DB query'lari) — RAG bilan parallel
    ishga tushirish uchun mavjud bo'lishi kerak.
    """
    from database import Candidate, VisualRecord, GlobalSetting

    face_context = ""
    if parsed_face_stats:
        parts = [
            f"Взгляд сфокусирован {parsed_face_stats.get('gaze_focused_pct', 0)}%",
            f"отведён {parsed_face_stats.get('gaze_away_pct', 0)}%",
            f"рот открыт (говорит) {parsed_face_stats.get('mouth_open_pct', 0)}%",
            f"глаза закрыты {parsed_face_stats.get('eyes_closed_pct', 0)}%",
        ]
        face_context = "\nДАННЫЕ ВИДЕОАНАЛИЗА ЛИЦА: " + ", ".join(parts) + "."

        dominant_emotion = parsed_face_stats.get("dominant_emotion")
        avg_stress = parsed_face_stats.get("avg_stress_score")
        if dominant_emotion or avg_stress is not None:
            emo_part = []
            if dominant_emotion:
                emo_part.append(f"доминирующая эмоция (по мимике лица): {dominant_emotion}")
            if avg_stress is not None:
                stress_label = "высокий" if avg_stress >= 60 else ("средний" if avg_stress >= 30 else "низкий")
                emo_part.append(f"уровень стресса по мимике: {stress_label} ({avg_stress}/100)")
            face_context += (
                "\nМИМИКА И ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ (реальное время, MediaPipe blendshapes): "
                + ", ".join(emo_part) + "."
            )

    # Server-side visual aggregate (oxirgi 120 sek)
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

    return face_context


def process_turn_pipeline(
    candidate_id: int,
    turn_uid: str,
    question: str,
    audio_path: str,
    audio_url: str,
    parsed_face_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """STT + prosody + RAG pipeline ni bajaradi (Celery'siz, plain function).

    Bu funksiyani threading fallback (main.py) ham, Celery task wrapper ham
    chaqiradi. Celery'ga bog'liqlik YO'Q — Celery o'rnatilmagan bo'lsa ham
    ishlaydi.

    Natija ``Candidate.answers`` dagi turn_uid ga teng element ichiga yoziladi.
    """
    import logic
    from database import SessionLocal, Candidate
    from sqlalchemy.orm.attributes import flag_modified

    analysis_db = SessionLocal()

    # Progressive update — har bosqich tugaganda DB'ga yozish + WS broadcast.
    # DB yangilash MUHIM: WebSocket Celery cross-process'da yo'qoladi, lekin
    # polling (frontend har 5 sek) DB'dan o'qiydi. Polling = ishonchli tarmoq,
    # WebSocket = optional speedup.
    def _save_turn_partial(updates: Dict[str, Any]) -> None:
        """Candidate.answers ichidagi turn_uid ga teng element'ni atomic yangilaydi."""
        try:
            cand = (
                analysis_db.query(Candidate)
                .with_for_update()
                .filter_by(id=candidate_id)
                .first()
            )
            if not cand:
                return
            ans = list(cand.answers or [])
            for i, item in enumerate(ans):
                if item.get("turn_uid") == turn_uid:
                    ans[i].update(updates)
                    break
            cand.answers = ans
            flag_modified(cand, "answers")
            analysis_db.commit()
        except Exception as exc:
            logger.warning(f"DB partial save failed (stage={updates.get('_stage')}): {exc}")
            try:
                analysis_db.rollback()
            except Exception:
                pass

    def _broadcast_partial(partial_update: Dict[str, Any]) -> None:
        """Avval DB ga yozadi (polling uchun), so'ng WebSocket broadcast (tezroq UI)."""
        # 1. DB ga yozish — polling shu yerdan oladi
        _save_turn_partial(partial_update)
        # 2. WebSocket broadcast (Celery'da fail bo'lishi mumkin — ahamiyat yo'q)
        try:
            _broadcast_turn({
                "type": "TURN_RESULT",
                "candidate_id": candidate_id,
                "question": question,
                "audio_url": audio_url,
                "turn_uid": turn_uid,
                **partial_update,
            })
        except Exception as exc:
            logger.debug(f"WS partial broadcast skipped: {exc}")

    try:
        # 1. STT (Deepgram → fallback Whisper) — birinchi va eng tez ko'rsatiladigan natija
        transcript_raw = ""
        stt_ms = 0
        t0 = _time.time()
        try:
            transcript_raw, stt_ms = logic.transcribe_audio(audio_path)
        except Exception as e:
            logger.warning(f"STT exception: {e}")
            transcript_raw = "(Речь не распознана)"
        # Bo'sh transkript — audio jim (kandidat indamadi yoki mikrofon o'chiq)
        if not transcript_raw:
            transcript_raw = "(Тишина)"
        stt_wall_ms = int((_time.time() - t0) * 1000)

        # ⚡ DARHOL: STT natijasini frontend'ga yuboramiz (foydalanuvchi
        # transkriptni ~3 sekundda ko'radi, AI tahlilini esa keyinroq)
        _broadcast_partial({
            "answer": transcript_raw,
            "stt_ms": stt_ms,
            "stt_wall_ms": stt_wall_ms,
            "_stage": "stt_done",
        })

        # 2. Face context — RAG'dan oldin tezda quramiz (faqat DB query, ~50-100ms)
        face_context = _build_face_context(parsed_face_stats, candidate_id, analysis_db)

        # 3. PARALLEL POOL — Smooth, Prosody, RAG ni bir vaqtda ishga tushiramiz
        # Avvalgi versiyada RAG smooth+prosody'dan KEYIN ishlardi (sekvensial chain).
        # Endi RAG raw transcript bilan darhol boshlanadi — smooth keyinroq display
        # uchun yangilanadi (RAG quality'siga ta'sir qilmaydi: Mistral kichik
        # noaniqlik'larni o'zi tushunadi).
        from concurrent.futures import ThreadPoolExecutor as _TP
        _parallel_pool = _TP(max_workers=3, thread_name_prefix="turn-parallel")
        parallel_started_at = _time.time()
        smooth_enabled = os.getenv("TRANSCRIPT_SMOOTH", "true").lower() not in ("false", "0", "no")

        prosody_future = _parallel_pool.submit(_safe_prosody, audio_path)
        smooth_future = _parallel_pool.submit(_safe_smooth, transcript_raw, smooth_enabled)
        rag_future = _parallel_pool.submit(_safe_rag, question, transcript_raw, face_context)

        # Prosody odatda eng tez tugaydi (0.5-2s) — uni alohida kutib darhol broadcast
        try:
            voice_raw = prosody_future.result(timeout=120)
            prosody_ms = int((_time.time() - parallel_started_at) * 1000)
        except Exception as e:
            logger.warning(f"Voice profiler failed: {e}")
            voice_raw = ""
            prosody_ms = 0

        if voice_raw:
            _broadcast_partial({
                "voice_raw": voice_raw,
                "prosody_ms": prosody_ms,
                "_stage": "prosody_done",
            })

        # Smooth — keyingi (1-3s)
        try:
            transcript = smooth_future.result(timeout=60) or transcript_raw
            smooth_ms = int((_time.time() - parallel_started_at) * 1000) - prosody_ms
            if smooth_ms < 0:
                smooth_ms = 0
        except Exception as e:
            logger.warning(f"Transcript smoothing failed: {e}")
            transcript = transcript_raw
            smooth_ms = 0

        # Smoothed transcript'ni yangi answer sifatida broadcast (faqat sezilarli farq bo'lsa)
        if smooth_enabled and transcript != transcript_raw and smooth_ms > 100:
            _broadcast_partial({
                "answer": transcript,
                "candidate_raw": transcript_raw,
                "smooth_ms": smooth_ms,
                "_stage": "smooth_done",
            })

        # RAG — eng oxiri (3-10s). Smooth+prosody bilan parallel ishlagani uchun
        # ai_ms = parallel boshlangandan to RAG natijasi kelgangacha (wall clock).
        try:
            rag_ai = rag_future.result(timeout=180)
        except Exception as e:
            logger.warning(f"RAG result fetch failed: {e}")
            rag_ai = "AI анализ недоступен"
        ai_ms = int((_time.time() - parallel_started_at) * 1000)
        _parallel_pool.shutdown(wait=False)

        # Total = STT (sequential) + max(smooth, prosody, RAG) (parallel) — wall clock
        parallel_wall_ms = int((_time.time() - parallel_started_at) * 1000)
        total_ms = stt_wall_ms + parallel_wall_ms

        # 5. Candidate.answers yangilash — yakuniy holatni saqlaymiz
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
                    ans[i]["total_ms"] = total_ms
                    ans[i]["_stage"] = "done"  # terminal — frontend polling toxtatadi

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

        logger.info(
            f"Pipeline done turn={turn_uid}: STT={stt_wall_ms}ms, "
            f"parallel(smooth/prosody/RAG)={parallel_wall_ms}ms, "
            f"total={total_ms}ms"
        )

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
                "smooth_ms": smooth_ms,
                "prosody_ms": prosody_ms,
                "ai_ms": ai_ms,
                "total_ms": total_ms,
                "_stage": "done",
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


# Celery task wrapper — Celery o'rnatilgan bo'lsa, decorator celery_app.task
# (yo'q bo'lsa stub orqali plain functionga aylanadi). Wrapper plain pipeline
# funksiyasiga o'tkazadi.
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
    """Celery task wrapper — pipeline'ni chaqiradi, retry/log Celery boshqaradi."""
    return process_turn_pipeline(
        candidate_id=candidate_id,
        turn_uid=turn_uid,
        question=question,
        audio_path=audio_path,
        audio_url=audio_url,
        parsed_face_stats=parsed_face_stats,
    )
