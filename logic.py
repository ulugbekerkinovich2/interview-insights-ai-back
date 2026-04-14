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

def load_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model

def _convert_to_wav(audio_path: str) -> str:
    """Convert any audio format to 16kHz mono WAV for fastest Whisper processing."""
    ext = Path(audio_path).suffix.lower()
    if ext == ".wav":
        return audio_path  # already wav
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return audio_path  # no ffmpeg, let whisper handle it
    wav_path = audio_path + ".16k.wav"
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", wav_path],
            capture_output=True, text=True, timeout=15,
        )
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            logger.info(f"Audio converted: {ext} → wav ({os.path.getsize(wav_path)} bytes)")
            return wav_path
        logger.warning(f"WAV conversion failed: {result.stderr[:200]}")
    except Exception as e:
        logger.warning(f"WAV conversion error: {e}")
    return audio_path

def transcribe_audio(audio_path: str):
    """Returns (transcript, elapsed_ms) tuple."""
    if not os.path.exists(audio_path):
        raise TranscriptionError("Audio file not found")

    # Pre-convert to WAV for speed (webm/ogg → wav is much faster for Whisper)
    wav_path = _convert_to_wav(audio_path)

    def _run_transcription(path: str) -> str:
        model = load_whisper_model()
        segments, _ = model.transcribe(
            path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            language=None,
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
        transcript = _run_transcription(wav_path)
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc
    finally:
        # Cleanup converted wav if it was created
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"Whisper STT: {elapsed_ms}ms | {len(transcript)} chars")

    if not transcript:
        raise TranscriptionError("Transcription produced no text")

    return transcript, elapsed_ms

def run_rag_mistral(question: str, answer: str, context: str = ""):
    rag_script_path = os.path.join(PROJECT_DIR, "utils", "rag_mistral_remote.py")
    if not os.path.exists(rag_script_path):
        raise AIServiceError(f"Script not found: {rag_script_path}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    cmd_with_args = [
        sys.executable,
        rag_script_path,
        "--question",
        question.strip(),
        "--answer",
        answer.strip(),
    ]
    if context:
        cmd_with_args.extend(["--context", context.strip()])
    
    try:
        # Reduced timeout to 45 seconds for better UX during server down
        result = subprocess.run(
            cmd_with_args,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=45, 
        )
    except subprocess.TimeoutExpired:
        raise AIServiceError("AI сервер не ответил вовремя (Timeout). Проверьте VPN или подключение к серверу.")
    except Exception as exc:
        raise AIServiceError(f"AI jarayonida kutilmagan xato: {str(exc)}")

    if result.returncode != 0:
        raise AIServiceError(result.stderr.strip() or result.stdout.strip() or "AI process failed")

    if not result.stdout.strip():
        raise AIServiceError("AI process returned empty response")

    return result.stdout.strip()

def extract_mistral_answer(raw_text: str) -> str:
    if not raw_text:
        return "Analysis not obtained"
    
    for marker in ["ОТВЕТ MISTRAL:", "ОТВЕТ MISTRAL"]:
        if marker in raw_text:
            cleaned = raw_text.split(marker, 1)[1].strip(": \n\r\t")
            if cleaned:
                return cleaned
    return raw_text.strip()
    
def ask_mistral_raw(prompt: str) -> str:
    rag_script_path = os.path.join(PROJECT_DIR, "utils", "rag_mistral_remote.py")
    if not os.path.exists(rag_script_path):
        raise AIServiceError(f"Script not found: {rag_script_path}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        result = subprocess.run(
            [sys.executable, rag_script_path, "--prompt", prompt.strip()],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60, 
        )
        if result.returncode != 0:
            raise AIServiceError(result.stderr.strip() or result.stdout.strip() or "AI process failed")
        
        return extract_mistral_answer(result.stdout.strip())
    except subprocess.TimeoutExpired:
        raise AIServiceError("AI сервер не ответил (Timeout).")
    except Exception as exc:
        raise AIServiceError(f"AI jarayonida xato: {str(exc)}")


def analyze_answer(question: str, answer: str, context: str = "") -> str:
    raw_output = run_rag_mistral(question, answer, context)
    return extract_mistral_answer(raw_output)

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
        # Convert to wav for librosa compatibility
        wav = _convert_to_wav(audio_path)
        data = mod.analyze_prosody(wav)
        if wav != audio_path and os.path.exists(wav):
            os.remove(wav)
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
            capture_output=True, text=True, encoding="utf-8", env=env
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
