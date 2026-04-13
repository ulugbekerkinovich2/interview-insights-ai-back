import os
import re
import sys
import json
import subprocess
from project_paths import resolve_project_dir

PROJECT_DIR = resolve_project_dir(__file__)
DEFAULT_VISUAL_OUTPUT_FILE = os.path.join(PROJECT_DIR, "visual_profile.json")
DEFAULT_QUESTION_TEXT = ""


def get_paths_from_args():
    """
    Можно запускать так:
    python candidate_profiler.py <audio_path> <transcript_path> <visual_json_path> [question_text]

    Если аргументы не переданы, используются тестовые пути по умолчанию.
    """
    if len(sys.argv) >= 4:
        question_text = os.getenv("UI_HR_QUESTION", "").strip() or DEFAULT_QUESTION_TEXT
        if len(sys.argv) >= 5:
            question_text = sys.argv[4]

        return {
            "audio_path": sys.argv[1],
            "transcript_path": sys.argv[2],
            "visual_path": sys.argv[3],
            "question_text": question_text
        }

    return {
        "audio_path": os.path.join(PROJECT_DIR, "test_audio.wav"),
        "transcript_path": os.path.join(PROJECT_DIR, "whisper_output.txt"),
        "visual_path": DEFAULT_VISUAL_OUTPUT_FILE,
        "question_text": os.getenv("UI_HR_QUESTION", "").strip() or DEFAULT_QUESTION_TEXT
    }


def print_block(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_script_and_capture(script_name, args=None, extra_env=None):
    if args is None:
        args = []

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            [sys.executable, script_name] + args,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        return {
            "script": script_name,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }

    except Exception as e:
        return {
            "script": script_name,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def build_reused_result(script_name, stdout_text):
    return {
        "script": script_name,
        "returncode": 0,
        "stdout": str(stdout_text or "").strip(),
        "stderr": "",
        "reused": True,
    }


def read_visual_profile(visual_path):
    if not os.path.exists(visual_path):
        return {
            "behavior_flags": ["Нет визуального файла"],
            "behavior_profile": "Визуальный анализ не запускался",
            "head_motion_status": "Нет данных",
            "gaze_motion_text": "Нет данных"
        }

    try:
        with open(visual_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "behavior_flags": ["Не удалось прочитать визуальный файл"],
            "behavior_profile": "Ошибка чтения визуального профиля",
            "head_motion_status": "Нет данных",
            "gaze_motion_text": "Нет данных"
        }


def read_transcript_text(transcript_path):
    if not os.path.exists(transcript_path):
        return ""

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def extract_whisper_text(stdout_text):
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    cleaned_lines = []

    for line in lines:
        lower_line = line.lower()
        if lower_line.startswith("язык:"):
            continue
        if lower_line.startswith("lang:"):
            continue
        if lower_line.startswith("транскрипт сохранён"):
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    return " ".join(cleaned_lines)


def extract_voice_summary(stdout_text):
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]

    if not lines:
        return "Нет данных по голосу"

    important_lines = []
    for line in lines:
        lower_line = line.lower()

        if "распознанный текст" in lower_line:
            continue

        important_lines.append(line)

    if not important_lines:
        return "Нет данных по голосу"

    return " ".join(important_lines[:12])


def extract_rag_summary(stdout_text):
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]

    if not lines:
        return "Нет данных по смысловому анализу"

    useful_lines = []
    for line in lines:
        if "ОТВЕТ AI" in line:
            continue
        if "ОТВЕТ MISTRAL" in line:
            continue
        if line.startswith("="):
            continue
        useful_lines.append(line)

    if not useful_lines:
        return "Нет данных по смысловому анализу"

    return " ".join(useful_lines[:16])


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def score_text_behavior(rag_text):
    text = normalize_text(rag_text)

    score = 0
    signals = []

    if "уход от прямого ответа" in text or "отсутствие прямого ответа" in text:
        score += 3
        signals.append("Есть признаки ухода от ответа")

    if "непонимание вопроса" in text:
        score += 1
        signals.append("Есть признаки затруднения с пониманием вопроса")

    if "неуместная вежливость" in text:
        score += 1
        signals.append("Есть признаки защитной манеры ответа")

    if "не доказывает ложь" in text or "не всегда" in text:
        signals.append("Прямых доказательств лжи нет")

    if "мало конкретики" in text or "слабая конкретика" in text:
        score += 1
        signals.append("Есть признаки слабой конкретики")

    if "уверенно" in text or "структурно" in text or "по делу" in text:
        score -= 1
        signals.append("Смысловой анализ отмечает относительно уверенную подачу")

    if not signals:
        signals.append("Смысловой анализ не выявил явных отклонений")

    return score, signals


def score_voice_behavior(voice_text):
    text = normalize_text(voice_text)

    score = 0
    signals = []

    weak_markers = [
        "неувер",
        "монотон",
        "напряж",
        "сбив",
        "паузы",
        "нерв",
        "сомнен",
        "тихий",
        "рван",
        "останов"
    ]

    strong_markers = [
        "уверен",
        "спокойн",
        "структур",
        "четк",
        "ясн",
        "связно"
    ]

    for marker in weak_markers:
        if marker in text:
            score += 1

    for marker in strong_markers:
        if marker in text:
            score -= 1

    if "неувер" in text or "нерв" in text or "напряж" in text:
        signals.append("В голосе возможны признаки напряжения")

    if "монотон" in text:
        signals.append("Голос может быть эмоционально сдержанным")

    if "тихий" in text:
        signals.append("Голос тихий — возможна осторожная подача")

    if "паузы" in text or "рван" in text or "останов" in text:
        signals.append("В речи заметны паузы или рваный темп")

    if "структур" in text or "четк" in text or "ясн" in text or "связно" in text:
        signals.append("Речь выглядит достаточно организованной")

    if not signals:
        signals.append("Явных голосовых отклонений не найдено")

    return score, signals


def score_whisper_text(answer_text):
    text = normalize_text(answer_text)

    score = 0
    signals = []

    if len(text) < 25:
        score += 2
        signals.append("Ответ слишком короткий")

    if len(text.split()) < 6:
        score += 1
        signals.append("Ответ малосодержательный")

    filler_words = [
        "ну",
        "как бы",
        "типа",
        "это самое",
        "в общем",
        "короче"
    ]

    filler_count = 0
    for filler in filler_words:
        filler_count += text.count(filler)

    if filler_count >= 3:
        score += 1
        signals.append("Есть слова-паразиты или речевые заполнители")

    if not signals:
        signals.append("Текст ответа по форме выглядит нормальным")

    return score, signals


def score_visual_behavior(visual_data):
    flags = visual_data.get("behavior_flags", [])
    behavior_profile = visual_data.get("behavior_profile", "")
    head_motion_status = visual_data.get("head_motion_status", "")
    gaze_motion_text = visual_data.get("gaze_motion_text", "")

    score = 0
    signals = []

    joined = " ".join(flags).lower() + " " + behavior_profile.lower()

    if "отводит взгляд" in joined:
        score += 2
        signals.append("Визуально есть избегание взгляда")

    if "частое моргание" in joined:
        score += 1
        signals.append("Визуально заметно частое моргание")

    if "повышенная моторика" in joined:
        score += 2
        signals.append("Визуально заметна повышенная моторика")

    if "резкие движения" in joined or "резкие скачки" in joined:
        score += 2
        signals.append("Визуально есть резкие движения")

    if "напряжения" in behavior_profile.lower():
        score += 1
        signals.append("Визуальный профиль указывает на напряжение")

    if "резкое" in head_motion_status.lower():
        score += 1
        signals.append("Голова движется резко")

    if "резко" in gaze_motion_text.lower():
        score += 1
        signals.append("Взгляд меняется резко")

    if not signals:
        signals.append("Визуально поведение выглядит спокойным")

    return score, signals


def build_final_assessment(total_score):
    if total_score <= 0:
        return "Профиль выглядит спокойным и достаточно уверенным"
    if total_score <= 2:
        return "Есть лёгкие признаки напряжения, но без сильных отклонений"
    if total_score <= 5:
        return "Есть заметные признаки напряжения или ухода от прямого ответа"
    return "Есть выраженные признаки напряжения, нестабильности или уклонения"


def build_recommendation(total_score):
    if total_score <= 0:
        return "Можно переходить к более глубоким профессиональным вопросам"
    if total_score <= 2:
        return "Стоит задать 1–2 уточняющих вопроса и проверить устойчивость ответа"
    if total_score <= 5:
        return "Нужно задать уточняющие вопросы и проверить конкретику на примерах"
    return "Нужно перепроверить ответ через конкретные кейсы и дополнительные вопросы"


def run_rag_mistral_analysis(question_text, transcript_path):
    script_name = "rag_mistral_remote.py"
    args = [transcript_path]

    extra_env = {
        "UI_HR_QUESTION": question_text.strip()
    }

    return run_script_and_capture(script_name, args=args, extra_env=extra_env)


def main():
    print_block("ЕДИНЫЙ ПРОФАЙЛЕР КАНДИДАТА")

    paths = get_paths_from_args()
    audio_path = paths["audio_path"]
    transcript_path = paths["transcript_path"]
    visual_path = paths["visual_path"]
    question_text = paths["question_text"]

    print("Запуск модулей...\n")
    print(f"Audio path      : {audio_path}")
    print(f"Transcript path : {transcript_path}")
    print(f"Visual path     : {visual_path}")
    print(f"Question text   : {question_text if question_text else '[не передан]'}")

    whisper_text = read_transcript_text(transcript_path)
    if whisper_text:
        whisper_result = build_reused_result("transcript_file", whisper_text)
    else:
        whisper_result = {
            "script": "transcript_file",
            "returncode": 1,
            "stdout": "",
            "stderr": "Transcript file is missing or empty.",
            "reused": True,
        }

    voice_env = os.getenv("UI_VOICE_ANALYSIS", "").strip()
    if voice_env:
        voice_result = build_reused_result("voice_profiler_test.py", voice_env)
    else:
        voice_result = run_script_and_capture("voice_profiler_test.py", [audio_path])

    visual_data = read_visual_profile(visual_path)

    if not whisper_text:
        whisper_text = extract_whisper_text(whisper_result["stdout"])

    rag_env = os.getenv("UI_RAG_ANALYSIS", "").strip()
    if rag_env:
        rag_result = build_reused_result("rag_mistral_remote.py", rag_env)
    else:
        rag_result = run_rag_mistral_analysis(question_text, transcript_path)

    print_block("СТАТУС МОДУЛЕЙ")
    print(f"Whisper: {'REUSED' if whisper_result.get('reused') else ('OK' if whisper_result['returncode'] == 0 else 'ОШИБКА')}")
    print(f"Voice profiler: {'REUSED' if voice_result.get('reused') else ('OK' if voice_result['returncode'] == 0 else 'ОШИБКА')}")
    print(f"RAG / Mistral анализ: {'REUSED' if rag_result.get('reused') else ('OK' if rag_result['returncode'] == 0 else 'ОШИБКА')}")
    print(f"Visual profiler: {'OK' if visual_data else 'ОШИБКА'}")

    voice_summary = extract_voice_summary(voice_result["stdout"])
    rag_summary = extract_rag_summary(rag_result["stdout"])

    print_block("РАСПОЗНАННЫЙ ТЕКСТ")
    print(whisper_text if whisper_text else "Нет текста")

    print_block("КРАТКИЙ АНАЛИЗ ГОЛОСА")
    print(voice_summary)

    print_block("КРАТКИЙ СМЫСЛОВОЙ АНАЛИЗ")
    print(rag_summary)

    print_block("КРАТКИЙ ВИЗУАЛЬНЫЙ АНАЛИЗ")
    print(f"Профиль поведения: {visual_data.get('behavior_profile', 'Нет данных')}")
    print(f"Head motion: {visual_data.get('head_motion_status', 'Нет данных')}")
    print(f"Gaze motion: {visual_data.get('gaze_motion_text', 'Нет данных')}")
    print("Флаги:")
    for flag in visual_data.get("behavior_flags", ["Нет данных"]):
        print(f"- {flag}")

    whisper_score, whisper_signals = score_whisper_text(whisper_text)
    voice_score, voice_signals = score_voice_behavior(voice_summary)
    rag_score, rag_signals = score_text_behavior(rag_summary)
    visual_score, visual_signals = score_visual_behavior(visual_data)

    total_score = whisper_score + voice_score + rag_score + visual_score

    final_assessment = build_final_assessment(total_score)
    recommendation = build_recommendation(total_score)

    print_block("СИГНАЛЫ ПО МОДУЛЯМ")

    print("По тексту ответа:")
    for signal in whisper_signals:
        print(f"- {signal}")

    print("\nПо голосу:")
    for signal in voice_signals:
        print(f"- {signal}")

    print("\nПо смысловому анализу:")
    for signal in rag_signals:
        print(f"- {signal}")

    print("\nПо визуальному поведению:")
    for signal in visual_signals:
        print(f"- {signal}")

    print_block("ИТОГОВЫЙ AI-ПРОФИЛЬ КАНДИДАТА")
    print(f"Интегральный балл: {total_score}")
    print(f"Вывод: {final_assessment}")
    print(f"Рекомендация интервьюеру: {recommendation}")

    print_block("ЧТО ДАЛЬШЕ")
    print("Следующий шаг:")
    print("1. Сохранять итоговый профиль по каждому turn")
    print("2. Делать финальный отчёт по всей сессии")


if __name__ == "__main__":
    main()
