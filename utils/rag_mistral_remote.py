import os
import sys
import argparse
import requests

from dotenv import load_dotenv

from project_paths import resolve_project_dir


PROJECT_DIR = resolve_project_dir(__file__)
PROJECT_ROOT = os.path.dirname(PROJECT_DIR)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


DEFAULT_TRANSCRIPT_FILE = os.path.join(PROJECT_DIR, "whisper_output.txt")


def read_transcript_text(transcript_path):
    if not os.path.exists(transcript_path):
        return ""

    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(candidate_answer, question_text="", company_context=""):
    question_block = f"\nHR tomonidan berilgan savol:\n{question_text}\n" if question_text else ""
    context_block = f"\nKOMPANIYA TALABLARI VA KONTEKST:\n{company_context}\n" if company_context else ""
    
    prompt = f"""
Siz professional psixolog va AI-intervyuer profraylersiz. (Professional psychologist and AI profiler).

Vazifangiz: Nomzodning javobini quyidagi kompaniya talablari va kontekst asosida tahlil qilish:
{context_block}

Ma'lumotlar:
{question_block}
Nomzodning javobi:
{candidate_answer}

TAHLILNI QUYIDAGI FORMATDA (O'ZBEK TILIDA) QAYTARIN:

1. UMUMIY XULOSA: Javobning mohiyati va nomzodning o'ziga bo'lgan ishonchi.
2. KOMPANIYAGA MOSLIK (FIT SCORE): Yuqoridagi kompaniya talablariga nomzod qanchalik mos keladi (0/100).
3. PSIXOLOGIK JIHATLAR: Javobdan qochish, mavhumlik yoki hayajon belgilari (Ovoz tahlili va kognitiv tushkunlik).
4. NAVBATDAGI STRATEGIK SAVOL: Nomzodning zaif nuqtalarini aniqlash uchun KEYINGI SAVOLNI TAVSIYA QILING.

ESLATMA:
- JAVOBNI FAQAT O'ZBEK TILIDA YOZING.
- MUSTAQIL SAVOL BO'LIMINI "NAVBATDAGI SAVOL:" belgisidan so'ng yozing.
""".strip()

    return prompt


def ask_mistral(prompt):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data, timeout=120)
    response.raise_for_status()

    payload = response.json()
    return payload.get("response", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Run Mistral analysis for interview transcript or direct Q/A text.")
    parser.add_argument("transcript_path", nargs="?", default=None)
    parser.add_argument("--question", default="")
    parser.add_argument("--answer", default="")
    parser.add_argument("--context", default="")
    args = parser.parse_args()

    question_text = args.question.strip() or os.getenv("UI_HR_QUESTION", "").strip()
    candidate_answer = args.answer.strip() or os.getenv("UI_CANDIDATE_ANSWER", "").strip()
    company_context = args.context.strip() or os.getenv("UI_COMPANY_CONTEXT", "").strip()

    transcript_path = args.transcript_path or DEFAULT_TRANSCRIPT_FILE
    if not candidate_answer:
        candidate_answer = read_transcript_text(transcript_path)

    if not candidate_answer:
        print("Не найден transcript-файл или в нём нет текста.")
        return

    print("=" * 80)
    print("РЕАЛЬНЫЙ ТЕКСТ КАНДИДАТА:")
    print("=" * 80)
    print(candidate_answer)

    prompt = build_prompt(candidate_answer, question_text=question_text, company_context=company_context)

    try:
        answer = ask_mistral(prompt)
    except Exception as e:
        print("\nОшибка при обращении к Mistral на МК:")
        print(e)
        return

    print("\n" + "=" * 80)
    print("ОТВЕТ MISTRAL:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
