"""Celery tasks paketi.

Modul strukturasi:

* ``stt_tasks`` — audio transkripsiya (faster-whisper / Deepgram)
* ``rag_tasks`` — Mistral/Ollama asosidagi AI javob generatsiyasi
* ``process_turn_tasks`` — to'liq intervyu turi pipeline (STT + prosody + RAG)

Celery avtomatik bu paketdan tasklarni aniqlaydi (``celery_app.include``).
"""
