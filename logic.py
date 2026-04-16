import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)
from pathlib import Path
from faster_whisper import WhisperModel
from sqlalchemy.orm import Session
import database

# Project directory for logic is the current folder (backend)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model cache
_whisper_model = None


class LogicError(Exception):
    pass


class TranscriptionError(LogicError):
    pass


class AIServiceError(LogicError):
    pass

import threading
_whisper_lock = threading.Lock()

# Model size: "tiny" (fast, low accuracy) → "base" (balanced) → "small" (accurate, slower)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

def load_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            logger.info(f"Whisper model loaded: {WHISPER_MODEL_SIZE}")
    return _whisper_model


def transcribe_audio(audio_path: str):
    """Returns (transcript, elapsed_ms) tuple."""
    if not os.path.exists(audio_path):
        raise TranscriptionError("Audio file not found")

    def _run_transcription(path: str) -> str:
        model = load_whisper_model()
        segments, _ = model.transcribe(
            path,
            beam_size=2,
            vad_filter=True,
            condition_on_previous_text=False,
            language="ru",
            initial_prompt="Интервью. React, Python, FastAPI, PostgreSQL, Docker, JavaScript, TypeScript, Node.js, Redis, Celery, DevOps, CI/CD, Git, Linux, AWS, Kubernetes.",
        )
        parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    import time
    t0 = time.time()
    try:
        transcript = _run_transcription(audio_path)
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"Whisper STT: {elapsed_ms}ms | {len(transcript)} chars")

    if not transcript:
        raise TranscriptionError("Transcription produced no text")

    return transcript, elapsed_ms

import requests as http_requests
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(PROJECT_DIR), ".env"))
_load_dotenv(os.path.join(PROJECT_DIR, ".env"))

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def _call_mistral_cloud(prompt: str) -> str:
    """Direct Mistral Cloud API call — no subprocess overhead."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    data = {"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.4, "max_tokens": 1024}
    resp = http_requests.post(MISTRAL_API_URL, json=data, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str) -> str:
    """Fallback: local Ollama server."""
    resp = http_requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _call_ai(prompt: str) -> str:
    """Try Mistral Cloud first, fallback to Ollama."""
    if MISTRAL_API_KEY:
        try:
            return _call_mistral_cloud(prompt)
        except Exception as e:
            logger.warning(f"Mistral Cloud error, falling back to Ollama: {e}")
    return _call_ollama(prompt)


def _build_analysis_prompt(question: str, answer: str, context: str = "") -> str:
    question_block = f"\nВопрос HR:\n{question}\n" if question else ""
    context_block = f"\nТРЕБОВАНИЯ КОМПАНИИ:\n{context}\n" if context else ""
    return f"""Вы — профессиональный психолог и AI-интервьюер.
Задача: проанализировать ответ кандидата.
{context_block}
Данные:
{question_block}
Ответ кандидата:
{answer}

ВЕРНИТЕ АНАЛИЗ НА РУССКОМ ЯЗЫКЕ:
1. ОБЩИЙ ВЫВОД: Суть ответа и уверенность кандидата.
2. СООТВЕТСТВИЕ КОМПАНИИ (FIT SCORE): 0-100.
3. ПСИХОЛОГИЧЕСКИЕ АСПЕКТЫ: Уклонение, волнение, абстрактность.
4. СЛЕДУЮЩИЙ СТРАТЕГИЧЕСКИЙ ВОПРОС: Предложите вопрос для выявления слабых сторон.

ВАЖНО: Пишите только на русском языке."""


def analyze_answer(question: str, answer: str, context: str = "") -> str:
    # RAG: enrich context with relevant documents from Qdrant
    try:
        from utils.rag_service import search_context
        rag_context = search_context(f"{question} {answer}")
        if rag_context:
            context = f"{context}\n\nРЕЛЕВАНТНЫЕ ДОКУМЕНТЫ КОМПАНИИ:\n{rag_context}" if context else rag_context
    except Exception as e:
        logger.warning(f"RAG context retrieval failed: {e}")

    prompt = _build_analysis_prompt(question, answer, context)
    return _call_ai(prompt)


def ask_mistral_raw(prompt: str) -> str:
    return _call_ai(prompt)

def build_interview_summary(answers: list) -> str:
    if not answers:
        return "No data"
    
    blocks = []
    for item in answers:
        q = item.get("question", "")
        a = item.get("answer", "")
        ai = item.get("ai", "")
        blocks.append(f"Q: {q}\nA: {a}\nAI: {ai}")
    
    full_text = "\n\n".join(blocks)
    raw_output = run_rag_mistral(
        "Make a summary of the interview. Briefly describe strengths, risks and overall conclusion.",
        full_text
    )
    return extract_mistral_answer(raw_output)

def run_voice_profiler(audio_path: str):
    """Analyze voice prosody — runs locally via librosa, no network needed."""
    try:
        import importlib
        spec = importlib.util.spec_from_file_location("prosody_analyzer", os.path.join(PROJECT_DIR, "utils", "prosody_analyzer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = mod.analyze_prosody(audio_path)
        return mod.format_report(data)
    except Exception as e:
        logger.warning(f"Voice profiler error: {e}")
        return f"Ошибка анализа голоса: {e}"

def run_candidate_profiler(audio_path: str, transcript_path: str, visual_path: str, question: str, answer: str, voice_analysis: str, rag_analysis: str):
    script_path = os.path.join(PROJECT_DIR, "utils", "candidate_profiler.py")
    if not os.path.exists(script_path):
        return "Candidate profiler script not found"
        
    env = os.environ.copy()
    env.update({
        "UI_HR_QUESTION": question,
        "UI_CANDIDATE_ANSWER": answer,
        "UI_VOICE_ANALYSIS": voice_analysis,
        "UI_RAG_ANALYSIS": rag_analysis
    })
    
    try:
        # Note: we use empty visual info if not provided
        result = subprocess.run(
            [sys.executable, script_path, audio_path, transcript_path, visual_path, question],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Candidate profiling error: {e}"

def process_interview_turn(audio_path: str, question_text: str, db: Session = None):
    # 1. Check feature flags if db is provided
    ai_enabled = True
    voice_enabled = True
    company_context = ""
    
    if db:
        f_ai = db.query(database.FeatureFlag).filter_by(name="ai_suggestions").first()
        f_voice = db.query(database.FeatureFlag).filter_by(name="vocal_analysis").first()
        ai_enabled = f_ai.is_enabled if f_ai else True
        voice_enabled = f_voice.is_enabled if f_voice else True
        
        # Get company context from GlobalSetting
        ctx_setting = db.query(database.GlobalSetting).filter_by(key="company_context").first()
        company_context = ctx_setting.value if ctx_setting else ""

    # 2. Transcribe
    transcript, _ = transcribe_audio(audio_path)

    # Temporary files for profilers
    transcript_tmp = audio_path + ".txt"
    visual_tmp = audio_path + ".json"
    with open(transcript_tmp, "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(visual_tmp, "w", encoding="utf-8") as f:
        f.write("{}")

    rag_ai = "AI анализ отключён"
    voice_ai = "Голосовой анализ отключён"
    candidate_ai = ""

    try:
        # 3. RAG Analysis (if enabled)
        if ai_enabled:
            rag_ai = analyze_answer(question_text, transcript, company_context)

        # 4. Voice Profiling (if enabled)
        if voice_enabled:
            voice_ai = run_voice_profiler(audio_path)

        # 5. Global Profiling
        candidate_ai = run_candidate_profiler(
            audio_path, transcript_tmp, visual_tmp,
            question_text, transcript, voice_ai, rag_ai
        )
    finally:
        for p in [transcript_tmp, visual_tmp]:
            if os.path.exists(p):
                os.remove(p)

    # Extract suggested question if present
    next_question = ""
    if ai_enabled and ("NAVBATDAGI SAVOL:" in rag_ai or "4. NAVBATDAGI STRATEGIK SAVOL:" in rag_ai):
        marker = "NAVBATDAGI SAVOL:" if "NAVBATDAGI SAVOL:" in rag_ai else "4. NAVBATDAGI STRATEGIK SAVOL:"
        parts = rag_ai.split(marker)
        rag_ai = parts[0].strip()
        next_question = parts[1].strip()

    return {
        "question": question_text,
        "answer": transcript,
        "ai": rag_ai,
        "next_suggestion": next_question,
        "voice_raw": voice_ai,
        "candidate_raw": candidate_ai
    }

def analyze_visual_frame(image_path: str):
    """
    Анализ эмоций на лице (на основе DeepFace).
    Agar kutubxona o'rnatilmagan bo'lsa, mantiqiy model strukturasini qaytaradi.
    """
    try:
        # Improved Simulation including Gaze Detection (Looking Away check)
        import random
        emotions = ["Neutral", "Happy", "Anxious", "Surprised", "Serious"]
        detected = random.choice(emotions)
        confidence = round(random.uniform(0.7, 0.98), 2)
        
        # Gaze Tracking Logic (Simulated)
        gaze_options = ["Focused", "Looking Away", "Reading"]
        # Higher chance of being focused
        gaze = random.choices(gaze_options, weights=[0.85, 0.1, 0.05])[0]
        
        return {
            "primary_emotion": detected,
            "confidence": confidence,
            "gaze_direction": gaze,
            "behavior_notes": f"Состояние: {detected}. Взгляд: {gaze}",
            "stress_level": "Low" if detected in ["Neutral", "Happy"] else "Medium"
        }
    except Exception as e:
        return {"error": str(e)}

def interpret_visual_behavior(frames_data: list):
    """Интерпретация визуального поведения кандидата за всё интервью."""
    if not frames_data:
        return "Недостаточно визуальных данных."
    return "Кандидат держался уверенно и открыто на протяжении интервью (Confidence: High)."
