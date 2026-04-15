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

# Mistral Cloud API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Fallback: Ollama local
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

DEFAULT_TRANSCRIPT_FILE = os.path.join(PROJECT_DIR, "whisper_output.txt")


def read_transcript_text(transcript_path):
    if not os.path.exists(transcript_path):
        return ""
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(candidate_answer, question_text="", company_context=""):
    question_block = f"\nВопрос HR:\n{question_text}\n" if question_text else ""
    context_block = f"\nТРЕБОВАНИЯ КОМПАНИИ И КОНТЕКСТ:\n{company_context}\n" if company_context else ""

    prompt = f"""
Вы — профессиональный психолог и AI-интервьюер-профайлер.

Ваша задача: проанализировать ответ кандидата.
{context_block}

Данные:
{question_block}
Ответ кандидата:
{candidate_answer}

ВЕРНИТЕ АНАЛИЗ В СЛЕДУЮЩЕМ ФОРМАТЕ (НА РУССКОМ ЯЗЫКЕ):

1. ОБЩИЙ ВЫВОД: Суть ответа и уровень уверенности кандидата в себе.
2. СООТВЕТСТВИЕ КОМПАНИИ (FIT SCORE): Насколько кандидат подходит (0-100).
3. ПСИХОЛОГИЧЕСКИЕ АСПЕКТЫ: Уклонение от ответа, абстрактность или признаки волнения.
4. СЛЕДУЮЩИЙ СТРАТЕГИЧЕСКИЙ ВОПРОС: Предложите следующий вопрос для выявления слабых сторон кандидата.

ВАЖНО:
- ПИШИТЕ ОТВЕТ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
- РАЗДЕЛ ВОПРОСА ПИШИТЕ ПОСЛЕ МЕТКИ "СЛЕДУЮЩИЙ ВОПРОС:".
""".strip()

    return prompt


def ask_mistral_cloud(prompt):
    """Call Mistral Cloud API (api.mistral.ai)"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    response = requests.post(MISTRAL_API_URL, json=data, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def ask_ollama(prompt):
    """Fallback: Call local Ollama server"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(url, json=data, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def ask_ai(prompt):
    """Try Mistral Cloud first, fallback to Ollama"""
    if MISTRAL_API_KEY:
        try:
            return ask_mistral_cloud(prompt)
        except Exception as e:
            print(f"Mistral Cloud error: {e}", file=sys.stderr)
    # Fallback to Ollama
    return ask_ollama(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path", nargs="?", default=None)
    parser.add_argument("--question", default="")
    parser.add_argument("--answer", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--prompt", default="")
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

    if args.prompt:
        prompt = args.prompt.strip()
    else:
        prompt = build_prompt(candidate_answer, question_text=question_text, company_context=company_context)

    try:
        answer = ask_ai(prompt)
    except Exception as e:
        print(f"\nОшибка AI: {e}")
        return

    print("\n" + "=" * 80)
    print("ОТВЕТ MISTRAL:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()
