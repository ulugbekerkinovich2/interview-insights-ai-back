"""utils paketi — yordamchi modullar (rate limit, cost tracker, RAG, va h.k.).

Defensive sys.path fix: Celery worker fork'lari va boshqa sub-process'larda
`from database import ...`, `import logic` ishlashi uchun. celery_app.py
import paytida sys.path qo'shadi, lekin ba'zi paytda fork'dan keyin yo'qoladi
yoki worker `--include` orqali to'g'ridan-to'g'ri tasks/utils ni import qiladi.
Bu yerda har gal utils import qilinganda sys.path tekshiriladi.
"""
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
