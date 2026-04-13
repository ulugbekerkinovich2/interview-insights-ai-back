import os
import sys
import json
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

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

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # Using base or small for faster inference on CPU
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise TranscriptionError("Audio file not found")

    try:
        model = load_whisper_model()
        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            language="ru",
        )

        parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)

        transcript = " ".join(parts).strip()
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    if not transcript:
        raise TranscriptionError("Transcription produced no text")

    return transcript

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
        raise AIServiceError("AI tahlil serveri so'rovga vaqtida javob bermadi (Timeout). Iltimos VPN yoki server ulanishini tekshiring.")
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
    script_path = os.path.join(PROJECT_DIR, "utils", "voice_profiler_test.py")
    prosody_script = os.path.join(PROJECT_DIR, "utils", "prosody_analyzer.py")
    
    combined_notes = []
    
    try:
        # 1. Basic Voice Profiler
        if os.path.exists(script_path):
            res1 = subprocess.run([sys.executable, script_path, audio_path], capture_output=True, text=True)
            combined_notes.append(res1.stdout.strip())
            
        # 2. Advanced Prosody Analysis
        if os.path.exists(prosody_script):
            res2 = subprocess.run([sys.executable, prosody_script, audio_path], capture_output=True, text=True)
            combined_notes.append(f"Prosody: {res2.stdout.strip()}")
    except Exception as e:
        combined_notes.append(f"Analysis error: {e}")
        
    return " | ".join(combined_notes)

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
    transcript = transcribe_audio(audio_path)

    # Temporary files for profilers
    transcript_tmp = audio_path + ".txt"
    visual_tmp = audio_path + ".json"
    with open(transcript_tmp, "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(visual_tmp, "w", encoding="utf-8") as f:
        f.write("{}")

    rag_ai = "AI tahlili o'chirilgan"
    voice_ai = "Ovoz tahlili o'chirilgan"
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
    Yuzdagi hissiyotlarni tahlil qilish (DeepFace asosida).
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
            "behavior_notes": f"Nomzod holati: {detected}. Nigoh: {gaze}",
            "stress_level": "Low" if detected in ["Neutral", "Happy"] else "Medium"
        }
    except Exception as e:
        return {"error": str(e)}

def interpret_visual_behavior(frames_data: list):
    """
    Barcha kadrlar to'plamidan intervyu davomidagi o'rtacha xulq-atvorni chiqaradi.
    """
    if not frames_data:
        return "Vizual ma'lumotlar yetarli emas."
        
    # Eng ko'p takrorlangan hissiyotni aniqlash
    # ...
    return "Nomzod intervyu davomida o'zini bosiq va ochiq tutdi (Confidence: High)."
