"""Celery tasks paketi.

Modul strukturasi:

* ``stt_tasks`` — audio transkripsiya (faster-whisper / Deepgram)
* ``rag_tasks`` — Mistral/Ollama asosidagi AI javob generatsiyasi
* ``process_turn_tasks`` — to'liq intervyu turi pipeline (STT + prosody + RAG)

Celery avtomatik bu paketdan tasklarni aniqlaydi (``celery_app.include``).
"""
# Defensive sys.path fix — Celery worker fork'idan keyin ham `import logic`,
# `from database import ...` ishlasin. celery_app.py'da bir marta qilinadi,
# lekin ba'zi spawn-rejimlarida (e.g. solo pool, debug) qaytadan bo'shatiladi.
# Bu yerda har bir task module import qilinishi bilan birga sys.path tekshiriladi.
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
