import os
import secrets
import string
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Header, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
import asyncio
import json
import shutil
import tempfile
import datetime
import requests
import uuid
import re
import base64
import hashlib
import bcrypt
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import bleach
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# #32 — Sentry error tracking (ixtiyoriy). SDK yo'q bo'lsa silent skip.
# Yoqish uchun: pip install sentry-sdk + .env'da SENTRY_DSN=...
_SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            send_default_pii=False,  # PII tashlamaymiz (GDPR)
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        logger.info("Sentry initialized")
    except ImportError:
        logger.warning("SENTRY_DSN set but sentry-sdk not installed: pip install sentry-sdk")
    except Exception as exc:
        logger.warning(f"Sentry init failed: {exc}")

import database
import schemas
import logic
from api.knowledge import router as knowledge_router
from api.salary import router as salary_router, seed_salary_grades as _seed_salary_grades
from api.audit import router as audit_router
from utils.executor import (
    stt_executor, llm_executor, frame_executor, run_bounded, QueueFull, pool_stats,
    shutdown_all as _shutdown_executors,
)

# Celery queue — agar Redis mavjud bo'lsa ishlatamiz, aks holda threading fallback.
# ``CELERY_ENABLED=false`` bilan ham to'liq o'chirib qo'yish mumkin.
try:
    from celery_app import celery_app, celery_enabled  # noqa: F401
    from tasks.stt_tasks import transcribe_audio_task
    from tasks.rag_tasks import generate_ai_reply_task
    from tasks.process_turn_tasks import process_turn_full_task
    _CELERY_IMPORTED = True
except Exception as _celery_import_exc:  # pragma: no cover
    _CELERY_IMPORTED = False
    logging.getLogger(__name__).warning(
        f"Celery import failed — threading fallback ishlatiladi: {_celery_import_exc}"
    )

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MEDIA_DIR = BACKEND_DIR / "media"
MEDIA_AUDIO_DIR = MEDIA_DIR / "audio"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

from concurrent.futures import ThreadPoolExecutor as _TPExec

# Bounded Telegram pool — har turn'da yangi thread spawn qilmaslik uchun.
# 1000 ta tez yuborish bo'lsa ham faqat 4 thread ishlatiladi.
_telegram_pool = _TPExec(max_workers=4, thread_name_prefix="tg-pool")


def send_telegram_notification(message: str):
    """Fire-and-forget Telegram xabari — bounded thread pool orqali (#21).
    Cheksiz thread spawn xavfini yo'q qiladi."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    def _send():
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram error: {e}")

    try:
        _telegram_pool.submit(_send)
    except Exception:
        # Pool yopilgan bo'lsa (shutdown) — silent skip
        pass

PASSWORD_HASH_PREFIX = "bcrypt_sha256$"

def get_password_hash(password):
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    secret = base64.b64encode(digest)
    return PASSWORD_HASH_PREFIX + bcrypt.hashpw(secret, bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password, hashed_password):
    if not hashed_password:
        return False

    if hashed_password.startswith(PASSWORD_HASH_PREFIX):
        raw_hash = hashed_password[len(PASSWORD_HASH_PREFIX):].encode("utf-8")
        digest = hashlib.sha256(plain_password.encode("utf-8")).digest()
        secret = base64.b64encode(digest)
        return bcrypt.checkpw(secret, raw_hash)

    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # bcrypt 5 raises for >72-byte plain secrets; old hashes cannot match safely.
        return False

from database import Candidate, ChatMessage, GlobalSetting, SessionLocal


def _broadcast_sync(message: dict):
    """Safely broadcast via WebSocket from sync context (background threads)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(manager.broadcast(message))
    except Exception:
        pass
    finally:
        loop.close()


# WebSocket Manager for Real-time Admin Updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for connection in dead_connections:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

manager = ConnectionManager()
ADMIN_MEDIA_ROLES = {"SuperAdmin", "Recruiter", "Psychologist"}


# --- WebSocket rate limiter ---
# HTTP endpoint'lar slowapi'dan foydalanadi, lekin WebSocket endpoint'lar
# (notifications, live-analysis, webrtc) himoya qilinmagan edi → DoS xavfi.
# Per-user va per-IP soddalashtirilgan in-memory counter.
import collections as _collections

_WS_MAX_PER_KEY = int(os.getenv("WS_MAX_CONNECTIONS_PER_USER", "5"))
_WS_WINDOW_SEC = int(os.getenv("WS_WINDOW_SEC", "60"))
_ws_connections: dict = _collections.defaultdict(list)  # key -> [timestamp, ...]
_ws_conn_lock = __import__("threading").Lock()


def _ws_rate_limit_check(key: str) -> bool:
    """True qaytarsa ulanishga ruxsat. False qaytarsa rad etish."""
    import time as _t
    now = _t.time()
    with _ws_conn_lock:
        timestamps = _ws_connections[key]
        # Eski timestamplarni tozalash
        timestamps[:] = [t for t in timestamps if now - t < _WS_WINDOW_SEC]
        if len(timestamps) >= _WS_MAX_PER_KEY:
            return False
        timestamps.append(now)
        return True


def _ws_dict_gc() -> int:
    """#20 — _ws_connections dict ning bo'sh kalitlarini tozalaydi
    (cheksiz o'sishni oldini oladi). Periodic background task chaqiradi."""
    import time as _t
    now = _t.time()
    removed = 0
    with _ws_conn_lock:
        empty_keys = []
        for k, ts_list in _ws_connections.items():
            ts_list[:] = [t for t in ts_list if now - t < _WS_WINDOW_SEC]
            if not ts_list:
                empty_keys.append(k)
        for k in empty_keys:
            _ws_connections.pop(k, None)
            removed += 1
    return removed


def _ws_extract_token(websocket: WebSocket, query_token: Optional[str]) -> Optional[str]:
    """#16 — Token'ni 2 ta joydan oladi: query param yoki HttpOnly cookie.
    Cookie ustun bo'ladi (XSS-himoyali)."""
    cookie_header = websocket.headers.get("cookie") or ""
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("admin_token="):
            return part.split("=", 1)[1]
    return query_token


def _ws_rate_key_from_token(token: Optional[str], fallback: str) -> str:
    """JWT bor bo'lsa user-based, aks holda IP-based key."""
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except JWTError:
            pass
    return f"ip:{fallback}"

# --- Upload validation (size + MIME + extension) ---
# Hajm chegaralari .env orqali sozlanadi
MAX_AUDIO_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_UPLOAD_MB", "50")) * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_UPLOAD_MB", "10")) * 1024 * 1024
# Global per-request tana hajmi cheklovi (multipart uchun ham). Audio eng katta tur.
MAX_REQUEST_BODY_BYTES = max(MAX_AUDIO_UPLOAD_BYTES, MAX_IMAGE_UPLOAD_BYTES) + 2 * 1024 * 1024  # +2 MB headers/form padding

ALLOWED_AUDIO_MIMES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/aac", "audio/flac",
    # Brauzer MediaRecorder ko'pincha audio-only oqimni video/webm sifatida yuboradi
    "video/webm",
    # Ba'zi brauzerlar blob uchun generic MIME yuboradi — ext tekshiruvi himoya qiladi
    "application/octet-stream",
}
ALLOWED_AUDIO_EXTS = {".webm", ".ogg", ".oga", ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac"}

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_MIME_TO_EXT = {
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/mp4": ".m4a", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
    "audio/aac": ".aac", "audio/flac": ".flac",
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


def _validate_upload_mime(
    file: UploadFile,
    allowed_mimes: set,
    allowed_exts: set,
    label: str,
    default_ext: str,
) -> str:
    """MIME va kengaytmani tekshiradi. Xavfsiz kengaytmani (nuqta bilan) qaytaradi.

    Har ikkalasi ham (ext va content-type) mavjud bo'lsa, ikkisi ham whitelistda bo'lishi shart.
    Faqat bittasi bo'lsa, shu bittasi whitelistda bo'lishi kifoya.
    """
    ctype = (file.content_type or "").lower().split(";")[0].strip()
    raw_ext = os.path.splitext(file.filename or "")[1].lower()

    if raw_ext and raw_ext not in allowed_exts:
        raise HTTPException(status_code=415, detail=f"Ruxsat etilmagan {label} kengaytmasi: {raw_ext}")
    # application/octet-stream ni faqat ext ma'lum bo'lsa ruxsat etamiz
    if ctype and ctype not in allowed_mimes:
        raise HTTPException(status_code=415, detail=f"Ruxsat etilmagan {label} turi: {ctype}")
    if ctype == "application/octet-stream" and not raw_ext:
        raise HTTPException(status_code=415, detail=f"{label} fayl turi aniqlanmadi")

    if raw_ext:
        return raw_ext
    inferred = _MIME_TO_EXT.get(ctype)
    return inferred or default_ext


def _stream_upload_to_path(
    file: UploadFile,
    dest_path: Path,
    max_bytes: int,
    label: str,
) -> int:
    """UploadFile ni dest_path ga yozadi, max_bytes dan oshsa 413 qaytaradi.
    Qisman yozilgan faylni o'chiradi. Yozilgan baytlar sonini qaytaradi.
    """
    CHUNK = 1024 * 1024  # 1 MB
    written = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = file.file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{label} hajmi {max_bytes // (1024*1024)} MB dan oshib ketdi",
                    )
                out.write(chunk)
    except HTTPException:
        try:
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Faylni saqlashda xato: {exc}")
    if written == 0:
        try:
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"{label} fayli bo'sh")
    return written


def _read_upload_bytes(file: UploadFile, max_bytes: int, label: str) -> bytes:
    """UploadFile ni xotiraga o'qiydi, hajm cheklovini majburlaydi."""
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} hajmi {max_bytes // (1024*1024)} MB dan oshib ketdi",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} fayli bo'sh")
    return data


# --- Path traversal himoya ---
_SAFE_FILENAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _safe_media_path(base_dir: Path, filename: str) -> Path:
    """Path traversal hujumidan himoya qiladi.

    Fayl nomi xavfsizligini tekshiradi: ``..``, yo'l ajratgichlari, null bayt va
    unicode hiylalarni rad etadi. ``Path.resolve()`` orqali kanonik yo'l
    ``base_dir`` ichida ekanligini majburlaydi (symlink/relative hujumlarga qarshi).

    Xatolik turlari
    ---------------
    * ``400`` — fayl nomi noto'g'ri formatda yoki taqiqlangan belgilar bor
    * ``403`` — yo'l ``base_dir`` tashqarisiga chiqadi
    * ``404`` — fayl mavjud emas yoki katalog
    """
    if not filename or len(filename) > 255:
        raise HTTPException(status_code=400, detail="Noto'g'ri fayl nomi")
    forbidden_chars = ("..", "/", "\\", "\x00")
    if any(ch in filename for ch in forbidden_chars):
        raise HTTPException(status_code=400, detail="Noto'g'ri fayl nomi")
    if not _SAFE_FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=400, detail="Noto'g'ri fayl nomi formati")

    try:
        base_resolved = base_dir.resolve()
        candidate = (base_dir / filename).resolve()
        # relative_to() xato chiqadi agar candidate base tashqarisida bo'lsa
        candidate.relative_to(base_resolved)
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Ruxsat etilmagan yo'l")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Fayl topilmadi")
    return candidate


DEFAULT_FEATURE_FLAGS = [
    {"name": "linkedin_search", "description": "Nomzod profilida LinkedIn qidiruv tugmasini ko'rsatish", "is_enabled": True},
    {"name": "pdf_export", "description": "Nomzod profilidan PDF hisobot eksportini yoqish", "is_enabled": True},
    {"name": "voice_tts", "description": "Nomzod sahifasida savolni ovozli o'qib berish (TTS)", "is_enabled": True},
    {"name": "interview_timer", "description": "Nomzod sessiyasida vaqt taymerini ko'rsatish", "is_enabled": True},
    {"name": "stress_overlay", "description": "Interview LIVE oynasida stress overlay effektini yoqish", "is_enabled": True},
    {"name": "gaze_tracking", "description": "AI Visual blokida gaze (nigoh) diagnostikasini ko'rsatish", "is_enabled": True},
    {"name": "ai_suggestions", "description": "Intervyu console'da AI keyingi qadam tavsiyasini ko'rsatish", "is_enabled": True},
    {"name": "vocal_analysis", "description": "Nomzod ovozi bo'yicha prosody va holat tahlilini yoqish", "is_enabled": True},
    {"name": "hr_filters_panel", "description": "Intervyu console'da HR talablarini (filterlar) ko'rsatish va tahrirlash", "is_enabled": True},
    {"name": "chat_citations", "description": "Psixolog AI chat'da iqtiboslar va manbalar ([1], [2]...) ko'rsatish", "is_enabled": False},
    {"name": "ai_cost_panel", "description": "Sozlamalar sahifasida AI расходы и лимиты panelini ko'rsatish", "is_enabled": False},
]

# --- WebRTC Signaling ---
class WebRTCRoom:
    def __init__(self):
        self.admin: Optional[WebSocket] = None
        self.candidate: Optional[WebSocket] = None
        # Admin takeover'ni oldini olish — kim hozirda admin ekanligini saqlaymiz
        self.admin_email: Optional[str] = None
        # Buffer signaling messages if peer is not connected yet.
        self.pending_for_admin: List[dict] = []
        self.pending_for_candidate: List[dict] = []

    def other(self, ws: WebSocket) -> Optional[WebSocket]:
        if ws == self.admin:
            return self.candidate
        if ws == self.candidate:
            return self.admin
        return None

    def enqueue_for(self, target: str, message: dict, max_len: int = 100):
        q = self.pending_for_admin if target == "admin" else self.pending_for_candidate
        q.append(message)
        # Keep bounded to avoid memory growth on unstable clients.
        if len(q) > max_len:
            del q[: len(q) - max_len]

    def flush_for(self, target: str) -> List[dict]:
        q = self.pending_for_admin if target == "admin" else self.pending_for_candidate
        out = list(q)
        q.clear()
        return out

import threading
webrtc_rooms: Dict[int, WebRTCRoom] = {}
_rooms_lock = threading.Lock()
# Room oxirgi faolligi vaqti — stale roomlarni avtomatik tozalash uchun.
_webrtc_room_last_active: Dict[int, float] = {}
# Stale room TTL — 30 daqiqa faolsiz bo'lgan roomlar o'chiriladi (memory leak oldini olish)
WEBRTC_ROOM_TTL_SEC = int(os.getenv("WEBRTC_ROOM_TTL_SEC", "1800"))
# Server ishga tushirilgan vaqt — frontend restart aniqlash uchun /health/detail da
SERVER_STARTED_AT = datetime.datetime.utcnow().isoformat() + "Z"


def _get_room(candidate_id: int) -> WebRTCRoom:
    import time as _time
    with _rooms_lock:
        room = webrtc_rooms.get(candidate_id)
        if room is None:
            room = WebRTCRoom()
            webrtc_rooms[candidate_id] = room
        _webrtc_room_last_active[candidate_id] = _time.time()
        return room


def _cleanup_stale_webrtc_rooms() -> int:
    """Uzoq vaqt faolsiz roomlarni tozalaydi (memory leak oldini oladi).
    Periodic task chaqiriladi."""
    import time as _time
    now = _time.time()
    removed = 0
    with _rooms_lock:
        for cid in list(webrtc_rooms.keys()):
            last = _webrtc_room_last_active.get(cid, 0)
            if now - last > WEBRTC_ROOM_TTL_SEC:
                room = webrtc_rooms.get(cid)
                # Faol WS connectionlari bormi — tekshirmaymiz, chunki
                # ular yopilganda room allaqachon tozalanadi. Bu faqat
                # yarim yopilgan (zombie) roomlar uchun.
                if room and room.admin is None and room.candidate is None:
                    webrtc_rooms.pop(cid, None)
                    _webrtc_room_last_active.pop(cid, None)
                    removed += 1
    return removed

def create_candidate_token(candidate_id: int, expires_minutes: int = 60 * 24) -> str:
    return create_access_token(
        {"sub": f"candidate:{candidate_id}", "role": "Candidate", "candidate_id": candidate_id},
        expires_delta=datetime.timedelta(minutes=expires_minutes),
    )

# --- Security Config ---
# SECRET_KEY — JWT imzo uchun. Prod muhitlarida ``.env`` dan olinishi SHART.
# Prod-ga yaqin nomlardagi muhitlarda (production/prod/staging/live) SECRET_KEY
# bo'lmasa yoki placeholder qiymatda bo'lsa server startup da to'xtaydi.
# Dev/test muhitlarida har protsess boshlanishida yangi random kalit yaratiladi
# (restart da barcha JWT tokenlar avtomatik bekor bo'ladi — bu hardcoded
# secretdan xavfsizroq, chunki kod commit qilinsa ham real xavf yo'q).
_PROD_ENV_NAMES = {"production", "prod", "staging", "live"}
_env_name = (os.getenv("ENVIRONMENT") or "").lower()
_is_prod_env = _env_name in _PROD_ENV_NAMES

SECRET_KEY = os.getenv("SECRET_KEY")
_SECRET_KEY_SOURCE = "env"

if SECRET_KEY and SECRET_KEY.startswith("CHANGE_ME"):
    # `.env.example` dagi placeholder qiymat — prodda qabul qilmaymiz
    if _is_prod_env:
        raise RuntimeError(
            "FATAL: SECRET_KEY placeholder qiymatda ('CHANGE_ME...'). "
            "Haqiqiy kalit yarating: "
            "python3 -c 'import secrets; print(secrets.token_urlsafe(48))'"
        )
    logger.warning(
        "SECRET_KEY placeholder qiymatda ('CHANGE_ME...') — dev rejimida ishlaydi, "
        "lekin prodda xavfli. Haqiqiy kalit o'rnating."
    )

if not SECRET_KEY:
    if _is_prod_env:
        raise RuntimeError(
            f"FATAL: SECRET_KEY muhit o'zgaruvchisi belgilanmagan (ENVIRONMENT={_env_name!r}). "
            "Ishga tushirish uchun: export SECRET_KEY=$(python3 -c "
            "'import secrets; print(secrets.token_urlsafe(48))')"
        )
    # Dev/test: har protsess uchun yangi random kalit
    SECRET_KEY = secrets.token_urlsafe(48)
    _SECRET_KEY_SOURCE = "random_per_process"
    logger.warning(
        "SECRET_KEY .env da yo'q — dev uchun random kalit generatsiya qilindi. "
        "Server restart da barcha JWT tokenlar bekor bo'ladi. "
        "Prod uchun: export SECRET_KEY=... (.env faylga yozing)"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

def _user_or_ip_key(request: Request) -> str:
    """Rate limit kaliti: autentifikatsiyalangan foydalanuvchilar uchun JWT ``sub``
    (har user alohida chegara), aks holda IP asosida. NAT ortidagi ko'plab
    foydalanuvchilarni jarimaga tortmaslik uchun mos.
    """
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = jwt.decode(auth[7:], SECRET_KEY, algorithms=[ALGORITHM])
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except JWTError:
            pass
    # Candidate tokenlar query param yoki cookie da kelishi mumkin — fallback
    token_param = request.query_params.get("token")
    if token_param:
        try:
            payload = jwt.decode(token_param, SECRET_KEY, algorithms=[ALGORITHM])
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except JWTError:
            pass
    return get_remote_address(request)


# Rate limit chegaralari ``.env`` dan o'qiladi (har endpoint uchun alohida)
RL_TRANSCRIBE = os.getenv("RATE_LIMIT_TRANSCRIBE", "30/minute")
RL_PROCESS_TURN = os.getenv("RATE_LIMIT_PROCESS_TURN", "20/minute")
RL_UPLOAD_AUDIO = os.getenv("RATE_LIMIT_UPLOAD_AUDIO", "60/minute")
RL_ANALYZE_FRAME = os.getenv("RATE_LIMIT_ANALYZE_FRAME", "120/minute")
RL_FACE_AI = os.getenv("RATE_LIMIT_FACE_AI", "30/minute")
RL_CHAT = os.getenv("RATE_LIMIT_CHAT", "30/minute")


limiter = Limiter(key_func=_user_or_ip_key)
app = FastAPI(title="AI Interview Backend API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    # #19 — har token uchun jti (JWT ID) — revocation list'da kalit
    to_encode.update({"exp": expire, "jti": secrets.token_urlsafe(16)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# #19 — JWT revocation list. Kalit: jti, qiymat: revoke timestamp.
# In-memory (single-process). Multi-process uchun Redis bilan almashtirsh kerak.
_revoked_jtis: Dict[str, float] = {}
_revoked_lock = __import__("threading").Lock()


def _is_token_revoked(jti: str) -> bool:
    if not jti:
        return False
    with _revoked_lock:
        return jti in _revoked_jtis


def _revoke_token(jti: str) -> None:
    if not jti:
        return
    import time as _t
    with _revoked_lock:
        _revoked_jtis[jti] = _t.time()
        # Eski (24 soatdan ko'p) entry'larni tozalash — token TTL o'tgan bo'lsa
        cutoff = _t.time() - 24 * 3600
        for k in list(_revoked_jtis.keys()):
            if _revoked_jtis[k] < cutoff:
                _revoked_jtis.pop(k, None)

# Add Restricted CORS middleware
# Only allow known origins (add your production domain here)
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "https://interview.misterdev.uz",
    "https://www.interview.misterdev.uz",
    "https://interview-api.misterdev.uz",
    "https://ufqpai.uz",
    "https://www.ufqpai.uz",
    "https://api.ufqpai.uz",
]

# #18 — CORS: explicit origins + regex pattern (subdomenlar va localhost portlari)
# `allow_credentials=True` cookie auth uchun shart, lekin u `*` bilan ishlamaydi.
_ALLOWED_ORIGIN_REGEX = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://([a-z0-9-]+\.)*(misterdev\.uz|ufqpai\.uz)$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)


# ====================================================================
# #54 — Security headers middleware (HSTS, CSP, X-Frame-Options, va h.k.)
# Davlat tashkilotlari uchun majburiy (clickjacking, MIME-sniff, XSS himoya).
# ====================================================================
_IS_PRODUCTION = (os.getenv("ENVIRONMENT", "development").lower() == "production")

@app.middleware("http")
async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    # Clickjacking — frame ichida sayt yuklanmasin
    response.headers["X-Frame-Options"] = "DENY"
    # MIME type sniffing — javob `text/html` deb noto'g'ri talqin qilinmasin
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Referrer minimal — to'liq URL boshqa saytga uzatilmaydi
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Cross-origin permissions — kamera/mikrofon faqat shu sayt uchun
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=()"
    # XSS protection (eski brauzerlar)
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # HSTS — faqat HTTPS prodakshen muhitida (lokal dev'da brauzerga HSTS pin
    # qo'yib qo'ymaslik uchun)
    if _IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # CSP — strict, lekin `unsafe-inline` Vite dev uchun kerak. Prodakshen
    # build'da inline yo'q, lekin recharts/markdown uchun blob: ham kerak.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob: https:; "
        "media-src 'self' blob:; "
        "connect-src 'self' https: wss: ws:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Early guard: Content-Length bilan kelgan haddan tashqari katta POST/PUT/PATCH ni
    Starlette tanasini bufferlashdan oldin rad etadi. Multipart yuklashlarda
    disk/xotira ekzozini oldini oladi."""
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"So'rov tanasi {MAX_REQUEST_BODY_BYTES // (1024*1024)} MB dan oshib ketdi"
                        },
                    )
            except ValueError:
                # Noto'g'ri Content-Length — quyidagi oddiy ishlov beruvchilarga qoldiramiz
                pass
    return await call_next(request)


# Role-based RAG knowledge-base API (/knowledge/*).
app.include_router(knowledge_router)
app.include_router(salary_router)
app.include_router(audit_router)

# app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media") # Unsecured mount removed


@app.on_event("startup")
async def startup():
    # init_db() = Base.metadata.create_all() — barcha jadvallarni yaratish
    # (idempotent: mavjudlarini tegmaydi, faqat YO'Q bo'lganlarini qo'shadi).
    # Avval faqat SQLite uchun chaqirilardi va Postgres'ga yangi modellar
    # yetib bormasdi (alembic ishlatish kerak edi). Endi Postgres'ga ham
    # chaqiramiz — yangi jadvallar (chat_sessions, salary_*, ...) avtomatik
    # paydo bo'ladi.
    try:
        database.init_db()
    except Exception as exc:
        logger.warning(f"init_db skipped: {exc}")

    # Capture the running event loop so sync handlers can schedule
    # per-user WebSocket pushes via ``NotificationHub``.
    try:
        from utils.notifications import hub as _notif_hub
        _notif_hub.set_loop(asyncio.get_running_loop())
    except Exception as exc:
        logger.warning(f"Notification hub loop binding failed: {exc}")
    # Ensure media directories exist (prevents 500 when saving uploads on fresh deploys)
    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        MEDIA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        (MEDIA_DIR / "frames").mkdir(parents=True, exist_ok=True)
    except Exception:
        # Avoid failing startup due to filesystem edge cases; endpoints will surface errors if writes fail.
        pass

    # Ensure feature toggles exist in every environment.
    try:
        with SessionLocal() as db:
            _ensure_default_feature_flags(db)
    except Exception as exc:
        # Do not crash startup; settings page will surface this if DB/migrations are missing.
        print(f"feature-flag bootstrap skipped: {exc}")

    # Auto-migration: yangi ustunlar (alembic ishlamagan deploy'larda) qo'lda
    # qo'shilishi. Hozirda: candidates.display_id (c008 migration).
    # Postgres uchun ALTER TABLE IF NOT EXISTS sintaksisi ishlatilsa idempotent.
    try:
        from sqlalchemy import text as _sa_text
        with database.engine.begin() as conn:
            # display_id ustunini qo'shish — agar mavjud emas bo'lsa.
            # Yangi format YYYYMMLNNNN = 11 belgi.
            conn.execute(_sa_text(
                "ALTER TABLE candidates ADD COLUMN IF NOT EXISTS display_id VARCHAR(11)"
            ))
            # Eski o'rnatilishlarda VARCHAR(8) qolgan bo'lsa kengaytamiz —
            # PostgreSQL ALTER COLUMN TYPE qisqartirmaydi, faqat kengaytmaydi.
            try:
                conn.execute(_sa_text(
                    "ALTER TABLE candidates ALTER COLUMN display_id TYPE VARCHAR(11)"
                ))
            except Exception:
                # Boshqa DB (SQLite) — type widening kerakmas / ishlamaydi
                pass
            # Unique index — alohida statement (IF NOT EXISTS index uchun ishlaydi)
            conn.execute(_sa_text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_candidates_display_id "
                "ON candidates(display_id)"
            ))

            # user_salary_profiles — shaxsiy va kasbiy ma'lumotlar uchun
            # yangi ustunlar (eski deploylar uchun lazy migration).
            for ddl in [
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS phone VARCHAR(32)",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS date_of_birth VARCHAR(20)",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS gender VARCHAR(10)",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS city VARCHAR(100)",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS specialization VARCHAR(200)",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS years_of_experience INTEGER",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS education TEXT",
                "ALTER TABLE user_salary_profiles ADD COLUMN IF NOT EXISTS bio TEXT",
                "ALTER TABLE salary_snapshots ADD COLUMN IF NOT EXISTS mb_input DOUBLE PRECISION NOT NULL DEFAULT 0.0",
                # Scale indexes — 100k+ record'da qidiruv tez bo'lsin
                "CREATE INDEX IF NOT EXISTS ix_candidates_name ON candidates (name)",
                "CREATE INDEX IF NOT EXISTS ix_candidates_status ON candidates (status)",
                "CREATE INDEX IF NOT EXISTS ix_candidates_created_at ON candidates (created_at DESC)",
                # ILIKE pattern uchun trigram (PostgreSQL pg_trgm extension) —
                # extension yo'q bo'lsa CREATE INDEX xatoga uchraydi, lekin try/except ushlaydi
                "CREATE INDEX IF NOT EXISTS ix_audit_log_action ON audit_log (action)",
                "CREATE INDEX IF NOT EXISTS ix_audit_log_entity_type ON audit_log (entity_type)",
            ]:
                try:
                    conn.execute(_sa_text(ddl))
                except Exception:
                    # SQLite yoki jadval hali yaratilmagan — create_all qo'shadi
                    pass
        logger.info("Auto-migration: candidates.display_id + user_salary_profiles columns ensured")
    except Exception as exc:
        # SQLite yoki boshqa DB'larda ALTER TABLE ADD COLUMN IF NOT EXISTS
        # ishlamasligi mumkin — shu holda alembic ishlatish tavsiya etiladi.
        logger.warning(f"Auto-migration display_id skipped: {exc}")

    # Stale Celery jobs cleanup — server restart qilinganda "running" yoki "queued"
    # holatida qolgan JobRecord yozuvlarini "failed" ga o'zgartirish. Bu yozuvlarni
    # kutayotgan klientlar yangi urinishni boshlashi mumkin.
    try:
        with SessionLocal() as db:
            stale = (
                db.query(database.JobRecord)
                .filter(database.JobRecord.status.in_(["running", "queued"]))
                .all()
            )
            if stale:
                for rec in stale:
                    rec.status = "failed"
                    rec.error = (rec.error or "") + " [server restart — stale job]"
                    rec.finished_at = datetime.datetime.utcnow()
                db.commit()
                logger.info(f"Stale Celery jobs cleaned: {len(stale)}")
    except Exception as exc:
        logger.warning(f"stale-jobs cleanup skipped: {exc}")

    # Salary grades seed — startup'da jadval bo'sh bo'lsa 16 ta yozuv qo'shamiz
    try:
        with SessionLocal() as db:
            _seed_salary_grades(db)
    except Exception as exc:
        logger.warning(f"salary grades seed skipped: {exc}")

    # Whisper idle TTL eviction — har 60 sekund tekshiradi, foydalanilmagan
    # modelni xotiradan bo'shatadi. Server tugasa task ham tugaydi.
    async def _whisper_idle_watcher():
        while True:
            try:
                await asyncio.sleep(60)
                logic._evict_if_idle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"whisper idle watcher: {exc}")

    # WebRTC stale rooms tozalash — har 5 daqiqa
    async def _webrtc_rooms_watcher():
        while True:
            try:
                await asyncio.sleep(300)
                n = _cleanup_stale_webrtc_rooms()
                if n:
                    logger.info(f"WebRTC stale rooms cleaned: {n}")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"webrtc rooms watcher: {exc}")

    # #20 — WS connection dict cleanup (cheksiz o'sishni oldini oladi)
    async def _ws_dict_cleanup_watcher():
        while True:
            try:
                await asyncio.sleep(3600)  # har 1 soat
                n = _ws_dict_gc()
                if n:
                    logger.info(f"WS connection dict cleaned: {n} keys removed")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"ws dict cleanup watcher: {exc}")

    # Frame disk cleanup — har 5 daqiqa, eski 2 soatdan ko'p faollarni o'chiradi.
    # Eski probabilistic 1% chance o'rniga deterministic — disk fill xavfini yo'q qiladi.
    async def _frames_cleanup_watcher():
        frames_dir = MEDIA_DIR / "frames"
        while True:
            try:
                await asyncio.sleep(300)
                if not frames_dir.exists():
                    continue
                _prune_old_frames(frames_dir, max_age_sec=2 * 3600)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"frames cleanup watcher: {exc}")

    try:
        app.state.whisper_watcher = asyncio.create_task(_whisper_idle_watcher())
        app.state.webrtc_watcher = asyncio.create_task(_webrtc_rooms_watcher())
        app.state.frames_watcher = asyncio.create_task(_frames_cleanup_watcher())
        app.state.ws_dict_watcher = asyncio.create_task(_ws_dict_cleanup_watcher())
    except Exception as exc:
        logger.warning(f"watcher start failed: {exc}")


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown — bounded executorlarni yopadi, Whisper modelni
    bo'shatadi, background tasklarni to'xtatadi."""
    for attr in ("whisper_watcher", "webrtc_watcher", "frames_watcher", "ws_dict_watcher"):
        try:
            task = getattr(app.state, attr, None)
            if task:
                task.cancel()
        except Exception:
            pass
    try:
        logic.release_whisper_model()
    except Exception:
        pass
    try:
        _shutdown_executors()
    except Exception:
        pass


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# User context for RBAC using JWT.
# Tokenni 2 ta joydan qabul qiladi:
#   1) `Authorization: Bearer ...` header (mavjud usul, frontend hozirgi)
#   2) `admin_token` HttpOnly cookie (yangi, XSS-himoyali — kelajakda asosiy)
def get_current_user(
    token: str = Depends(oauth2_scheme),
    admin_token_cookie: Optional[str] = Cookie(default=None, alias="admin_token"),
    db: Session = Depends(get_db),
):
    # Cookie ustun bo'lsin — XSS himoyasi uchun (Bearer header endi fallback)
    effective_token = admin_token_cookie or token
    if not effective_token:
        return None
    try:
        payload = jwt.decode(effective_token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        jti: str = payload.get("jti", "")
        if email is None:
            return None
        # #19 — revocation list tekshiruvi (logout qilingan tokenlar)
        if jti and _is_token_revoked(jti):
            return None
    except JWTError:
        return None

    user = db.query(database.User).filter_by(email=email).first()
    return user

def require_role(roles: List[str]):
    def role_checker(user: database.User = Depends(get_current_user)):
        if not user:
            raise HTTPException(status_code=401, detail="Сессия истекла или вы не авторизованы")
        if user.role not in roles and user.role != "SuperAdmin":
            raise HTTPException(status_code=403, detail="У вас нет прав для выполнения этого действия")
        return user
    return role_checker

# Shortcut: any authenticated admin user
require_admin = require_role(["SuperAdmin", "Recruiter", "Psychologist"])


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def _decode_candidate_token(candidate_token: str) -> Optional[int]:
    try:
        payload = jwt.decode(candidate_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

    if payload.get("role") != "Candidate":
        return None

    candidate_id = payload.get("candidate_id")
    if isinstance(candidate_id, int):
        return candidate_id
    if isinstance(candidate_id, str) and candidate_id.isdigit():
        return int(candidate_id)
    return None


def _ensure_default_feature_flags(db: Session) -> int:
    existing = {f.name: f for f in db.query(database.FeatureFlag).all()}
    created = 0
    for feature in DEFAULT_FEATURE_FLAGS:
        current = existing.get(feature["name"])
        if current is None:
            db.add(
                database.FeatureFlag(
                    name=feature["name"],
                    description=feature["description"],
                    is_enabled=feature["is_enabled"],
                )
            )
            created += 1
        elif not current.description:
            current.description = feature["description"]
    if created:
        db.commit()
    return created


def get_candidate_or_404(db: Session, candidate_id: int) -> Candidate:
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


def generate_access_code() -> str:
    return "".join(secrets.choice(string.digits) for _ in range(6))


def generate_unique_access_code(db: Session, max_attempts: int = 10) -> str:
    for _ in range(max_attempts):
        code = generate_access_code()
        exists = db.query(Candidate.id).filter(Candidate.access_code == code).first()
        if not exists:
            return code
    raise RuntimeError("Unable to generate a unique access code")


# --- Health Check ---
@app.get("/media/frames/{filename}")
def get_protected_frame(
    filename: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    # Keep frame serving DB-free. This endpoint is hit very frequently by live previews.
    jwt_token = token or _extract_bearer_token(authorization)
    if not jwt_token:
        raise HTTPException(status_code=401, detail="Rasmga kirish uchun tizimga kiring")

    try:
        payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Noto'g'ri yoki eskirgan token")

    role = payload.get("role")
    if role not in ADMIN_MEDIA_ROLES:
        raise HTTPException(status_code=403, detail="Ushbu rasmga kirish uchun ruxsat yo'q")

    file_path = _safe_media_path(MEDIA_DIR / "frames", filename)
    from fastapi.responses import FileResponse
    return FileResponse(file_path)

@app.get("/media/audio/{filename}")
def get_protected_audio(
    filename: str,
    token: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    jwt_token = token or _extract_bearer_token(authorization)
    if not jwt_token:
        raise HTTPException(status_code=401, detail="Audio faylga kirish uchun tizimga kiring")

    try:
        payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Noto'g'ri yoki eskirgan token")

    # Allow both admin roles and candidates to access audio
    role = payload.get("role")
    if role not in ADMIN_MEDIA_ROLES and role != "Candidate":
        raise HTTPException(status_code=403, detail="Audio faylga kirish uchun ruxsat yo'q")

    file_path = _safe_media_path(MEDIA_AUDIO_DIR, filename)
    from fastapi.responses import FileResponse
    media_type = "audio/webm" if filename.endswith(".webm") else "audio/ogg" if filename.endswith(".ogg") else "audio/wav"
    file_size = file_path.stat().st_size
    return FileResponse(file_path, media_type=media_type, headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)})


@app.get("/health", response_model=schemas.HealthSchema)
def health_check():
    try:
        database.check_database_connection()
    except Exception as exc:
        payload = {
            "status": "degraded",
            "service": "AI Interview Backend",
            "database": {
                "available": False,
                "dialect": database.get_database_metadata()["dialect"],
                "detail": str(exc),
            },
        }
        return JSONResponse(status_code=503, content=payload)

    return {
        "status": "ok",
        "service": "AI Interview Backend",
        "database": {
            "available": True,
            "dialect": database.get_database_metadata()["dialect"],
            "detail": None,
        },
    }


@app.get("/health/detail")
def health_detail(_: database.User = Depends(require_admin)):
    """Kengaytirilgan diagnostika: thread pool yuklamasi, Whisper holati,
    AI xarajat holati. Faqat admin rolliga ruxsat."""
    try:
        db_ok = True
        database.check_database_connection()
    except Exception as exc:
        db_ok = False

    from utils import cost_tracker
    return {
        "status": "ok" if db_ok else "degraded",
        "server_started_at": SERVER_STARTED_AT,
        "database": {
            "available": db_ok,
            "dialect": database.get_database_metadata()["dialect"],
        },
        "whisper": logic.whisper_status(),
        "thread_pools": pool_stats(),
        "ai_cost": cost_tracker.stats(),
        "webrtc": {
            "active_rooms": len(webrtc_rooms),
            "ttl_sec": WEBRTC_ROOM_TTL_SEC,
        },
    }


@app.get("/health/cost")
def health_cost(_: database.User = Depends(require_admin)):
    """AI xarajat holati — Mistral va Deepgram uchun sutkali/oylik summa.
    Chegarani oshsa AI chaqiruvlari bloklanadi."""
    from utils import cost_tracker
    return cost_tracker.stats()


# Ephemeral TURN credentials — STUN/TURN konfig'ini har user uchun
# qisqa muddatli credential bilan qaytaradi. Hardcoded credentials xavfini
# yo'q qiladi (eski Frontend ishlatib turgan ochiq creds bekor qilinadi).
import hmac as _hmac
import hashlib as _hashlib
import base64 as _b64

TURN_SHARED_SECRET = os.getenv("TURN_SHARED_SECRET", "")
TURN_URLS = [
    u.strip()
    for u in os.getenv(
        "TURN_URLS",
        "stun:stun.l.google.com:19302,turn:global.relay.metered.ca:80,turn:global.relay.metered.ca:443?transport=tcp",
    ).split(",")
    if u.strip()
]
TURN_TTL_SEC = int(os.getenv("TURN_TTL_SEC", "1800"))  # 30 daqiqa


@app.get("/webrtc/turn-config")
def get_turn_config(current_user: database.User = Depends(get_current_user)):
    """Ephemeral TURN credentials. Har 30 daqiqada bekor bo'ladi.
    Frontend STUN+TURN URL'larni shu endpointdan oladi (hardcoded YO'Q).
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Auth required")

    import time as _time
    expiry = int(_time.time()) + TURN_TTL_SEC
    username = f"{expiry}:{current_user.id}"

    if TURN_SHARED_SECRET:
        # RFC 5766-TURN-REST: HMAC-SHA1(username, secret)
        digest = _hmac.new(
            TURN_SHARED_SECRET.encode(), username.encode(), _hashlib.sha1
        ).digest()
        credential = _b64.b64encode(digest).decode()
    else:
        # Fallback: dev'da yoki TURN secret yo'q bo'lsa public STUN ishlatamiz
        credential = ""

    ice_servers = []
    for url in TURN_URLS:
        if url.startswith("stun:"):
            ice_servers.append({"urls": url})
        else:
            ice_servers.append(
                {
                    "urls": url,
                    "username": username,
                    "credential": credential,
                }
            )
    return {"iceServers": ice_servers, "ttl": TURN_TTL_SEC}


@app.get("/auth/me", response_model=schemas.UserSchema)
def get_me(current_user: database.User = Depends(get_current_user)):
    """JWT token'ni server tarafda tekshiradi. Frontend `<AdminGuard>` shu
    endpoint orqali real auth tasdiqlash imkoniyatini olishi mumkin (eski
    `localStorage.admin_authenticated` o'rniga)."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user


@app.post("/admin/cost/reset")
def admin_cost_reset(_: database.User = Depends(require_admin)):
    """SuperAdmin uchun xarajat chegaralarini qayta o'rnatish (manual override)."""
    from utils import cost_tracker
    cost_tracker.reset()
    return {"status": "reset"}


# --- Candidates Endpoints ---

@app.get("/candidates/stats/")
def get_candidate_stats(db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    candidates = db.query(database.Candidate).all()
    total = len(candidates)
    
    # Simple logic to extract fit scores from summaries or answers
    # For now, let's assume we can parse it from summaries like "Fit Score: 85%"
    scores = []
    import re
    for c in candidates:
        if c.summary:
            match = re.search(r"Fit Score:\s*(\d+)%", c.summary)
            if match:
                scores.append(int(match.group(1)))
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    status_counts = {}
    for c in candidates:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1
        
    return {
        "total_candidates": total,
        "average_fit_score": round(avg_score, 1),
        "status_distribution": status_counts,
        "recent_activity": total # simplified
    }

_timing_cache: Dict[int, tuple] = {}  # candidate_id -> (timestamp, result, answers_len)
_timing_cache_lock = __import__("threading").Lock()
_TIMING_CACHE_TTL = 5.0  # sek


@app.get("/candidates/{candidate_id}/timing")
def get_candidate_timing(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    """Get processing time statistics for each turn of a candidate.
    #30 — natija 5 sekund cache qilinadi (frontend har 5-10 sek polling
    qilganda CPU yuqotishni kamaytiradi). Cache invalidate: agar
    `len(candidate.answers)` o'zgargan bo'lsa avtomatik freshly hisoblanadi.
    """
    import time as _t
    candidate = get_candidate_or_404(db, candidate_id)
    answers = candidate.answers or []
    answers_len = len(answers)

    # Cache lookup
    with _timing_cache_lock:
        cached = _timing_cache.get(candidate_id)
    if cached:
        cached_ts, cached_result, cached_len = cached
        if cached_len == answers_len and (_t.time() - cached_ts) < _TIMING_CACHE_TTL:
            return cached_result
    turns = []
    total_stt = 0
    total_prosody = 0
    total_ai = 0
    total_cost = 0
    total_audio_sec = 0
    for i, a in enumerate(answers):
        stt = a.get("stt_wall_ms", a.get("stt_ms", 0))
        prosody = a.get("prosody_ms", 0)
        ai = a.get("ai_ms", 0)
        total = a.get("total_ms", stt + prosody + ai)
        cost = a.get("cost_usd", 0)
        audio_sec = a.get("audio_duration_sec", 0)
        total_stt += stt
        total_prosody += prosody
        total_ai += ai
        total_cost += cost
        total_audio_sec += audio_sec
        turns.append({
            "turn": i + 1,
            "question": (a.get("question") or "")[:80],
            "stt_ms": stt,
            "prosody_ms": prosody,
            "ai_ms": ai,
            "total_ms": total,
            "cost_usd": cost,
            "stt_provider": a.get("stt_provider", "whisper"),
        })
    result = {
        "candidate_id": candidate_id,
        "candidate_name": candidate.name,
        "turns": turns,
        "summary": {
            "total_turns": len(turns),
            "total_stt_ms": total_stt,
            "total_prosody_ms": total_prosody,
            "total_ai_ms": total_ai,
            "total_ms": total_stt + total_prosody + total_ai,
            "avg_stt_ms": round(total_stt / len(turns)) if turns else 0,
            "avg_prosody_ms": round(total_prosody / len(turns)) if turns else 0,
            "avg_ai_ms": round(total_ai / len(turns)) if turns else 0,
            "total_cost_usd": round(total_cost, 4),
            "total_audio_sec": round(total_audio_sec, 1),
        },
    }
    # #30 — cache yozish
    with _timing_cache_lock:
        _timing_cache[candidate_id] = (_t.time(), result, answers_len)
        # Memory leak oldini olish — cache 1000+ ga yetsa eskini chiqaramiz
        if len(_timing_cache) > 1000:
            oldest = min(_timing_cache.items(), key=lambda kv: kv[1][0])[0]
            _timing_cache.pop(oldest, None)
    return result


@app.get("/candidates/", response_model=List[schemas.CandidateSchema])
def read_candidates(
    limit: int = 200,
    offset: int = 0,
    page: int = 0,
    size: int = 0,
    q: Optional[str] = None,
    status: Optional[str] = None,
    include_answers: bool = False,
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    """Nomzodlar ro'yxati. Server-side pagination + qidiruv + filter.

    Pagination:
      - Yangi format: `?page=1&size=25` (1-indexed)
      - Eski format: `?limit=200&offset=0` (backward compat)

    Qidiruv (`q=`): name, display_id, summary (case-insensitive ILIKE)
    Filter (`status=`): "New" / "In Progress" / "Completed" / "Hired" / "Rejected"

    Response header'da `X-Total-Count: <total>` qaytariladi — frontend
    pagination UI uchun.

    `include_answers=false` (default) — `answers` JSON qaytarilmaydi
    (10x tezroq, har nomzod 5-50 KB tejash).
    """
    # Page/size argumentlari ustuvor — agar berilgan bo'lsa
    if page > 0 and size > 0:
        if size > 500:
            size = 500
        offset = (page - 1) * size
        limit = size
    else:
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500
        if offset < 0:
            offset = 0

    qs = db.query(Candidate)

    # Filter — status
    if status:
        qs = qs.filter(Candidate.status == status)

    # Qidiruv — name / display_id / summary
    if q:
        like = f"%{q.strip()}%"
        from sqlalchemy import or_
        qs = qs.filter(
            or_(
                Candidate.name.ilike(like),
                Candidate.display_id.ilike(like),
                Candidate.summary.ilike(like),
            )
        )

    qs = qs.order_by(Candidate.created_at.desc())
    candidates = qs.offset(offset).limit(limit).all()

    if not include_answers:
        # `answers` ni bo'sh ro'yxatga almashtirib, response payload'ni 90% kichraytiramiz.
        for c in candidates:
            try:
                db.expunge(c)
                c.answers = []
            except Exception:
                pass

    return candidates


@app.get("/candidates/count")
def count_candidates(
    q: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    """Pagination UI uchun jami count.
    Filter parametrlari /candidates/ bilan bir xil.
    """
    qs = db.query(Candidate)
    if status:
        qs = qs.filter(Candidate.status == status)
    if q:
        like = f"%{q.strip()}%"
        from sqlalchemy import or_
        qs = qs.filter(
            or_(
                Candidate.name.ilike(like),
                Candidate.display_id.ilike(like),
                Candidate.summary.ilike(like),
            )
        )
    return {"total": qs.count()}

@app.get("/candidates/{candidate_id}/visual", response_model=List[schemas.VisualRecordSchema])
def read_visual_records(
    candidate_id: int,
    limit: int = 200,
    order: str = "asc",
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    get_candidate_or_404(db, candidate_id)
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    order = order.lower().strip()
    if order not in {"asc", "desc"}:
        order = "asc"

    order_by = database.VisualRecord.timestamp.asc() if order == "asc" else database.VisualRecord.timestamp.desc()
    records = (
        db.query(database.VisualRecord)
        .filter(database.VisualRecord.candidate_id == candidate_id)
        .order_by(order_by)
        .limit(limit)
        .all()
    )
    return [
        {
            "emotion": r.emotion,
            "stress_level": r.stress_level,
            "image_url": r.image_url,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


@app.get("/candidates/{candidate_id}/visual/latest-frame")
def read_latest_visual_frame(
    candidate_id: int,
    db: Session = Depends(get_db),
    user: database.User = Depends(require_role(["SuperAdmin", "Recruiter", "Psychologist"])),
):
    get_candidate_or_404(db, candidate_id)
    record = (
        db.query(database.VisualRecord)
        .filter(database.VisualRecord.candidate_id == candidate_id)
        .order_by(database.VisualRecord.timestamp.desc())
        .first()
    )
    if not record or not record.image_url:
        raise HTTPException(status_code=404, detail="No visual frame available")

    # image_url is stored as "/media/frames/<filename>"
    filename = Path(record.image_url).name
    # Defense-in-depth: DB dagi image_url buzilgan bo'lsa ham path traversal
    # qo'shimcha validatsiya orqali to'sib qo'yiladi.
    file_path = _safe_media_path(MEDIA_DIR / "frames", filename)
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return {
        "candidate_id": candidate_id,
        "image": f"data:image/jpeg;base64,{encoded}",
        "analysis": {
            "primary_emotion": record.emotion,
            "stress_level": record.stress_level,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        },
    }

@app.get("/candidates/{candidate_id}", response_model=schemas.CandidateSchema)
def read_candidate(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    return get_candidate_or_404(db, candidate_id)

def _generate_display_id(db: Session, ts: Optional[datetime.datetime] = None) -> str:
    """Kandidat uchun foydalanuvchi-do'st ID format: YYYYMMLNNNN.

    Tarkibi (jami 11 belgi):
      - YYYY — 4 raqamli yil (2026)
      - MM   — 2 raqamli oy (04)
      - L    — 1 ta katta harf A-Z (A, keyin to'lgach B, va h.k.)
      - NNNN — 4 raqamli ketma-ketlik 0001..9999 shu (yil, oy, harf) bo'limida

    Misol: 202604A0001 — 2026-yil 04-oy, "A" guruhi, 1-nomzod.
    A-guruhi 9999 ga yetganda B'ga o'tiladi, B → C va h.k.

    Race condition'dan himoya: SQLAlchemy unique constraint + retry. Bir
    vaqtda 2 ta nomzod yaratilsa, IntegrityError tushadi va retry'da yangi
    raqam olinadi.
    """
    if ts is None:
        ts = datetime.datetime.utcnow()
    yyyy = ts.strftime("%Y")  # 2026
    mm = ts.strftime("%m")    # 04
    ym_prefix = f"{yyyy}{mm}"  # YYYYMM (6 belgi)

    # Shu oyda mavjud eng katta display_id ni topib, undan keyingisini hisoblaymiz.
    last = (
        db.query(database.Candidate.display_id)
        .filter(database.Candidate.display_id.like(f"{ym_prefix}%"))
        .order_by(database.Candidate.display_id.desc())
        .first()
    )
    letter = "A"
    next_seq = 1
    if last and last[0] and len(last[0]) >= 11:
        try:
            last_letter = last[0][6]
            last_seq = int(last[0][7:11])
            if last_seq < 9999:
                letter = last_letter
                next_seq = last_seq + 1
            elif last_letter < "Z":
                # Harfdan keyingisiga o'tamiz, raqam 0001'dan boshlanadi
                letter = chr(ord(last_letter) + 1)
                next_seq = 1
            else:
                # Z9999 — bu oyda quvvat tugadi (260000 ta nomzod)
                # Bu juda kam ehtimol; har ehtimolga qarshi shu format'da
                # qoldiramiz va unique constraint xatosini chaqirishga
                # ruxsat beramiz (retry endpointi handle qiladi).
                letter = "Z"
                next_seq = 9999
        except (ValueError, IndexError):
            letter = "A"
            next_seq = 1
    return f"{ym_prefix}{letter}{next_seq:04d}"


@app.post("/candidates/", response_model=schemas.CandidateCreateResponse)
def create_candidate(candidate: schemas.CandidateCreate, db: Session = Depends(get_db), current_user: database.User = Depends(require_role(["SuperAdmin", "Recruiter"]))):
    # Sanitize name
    safe_name = bleach.clean(candidate.name, tags=[], strip=True) if candidate.name else ""

    # Generate secure 16-char access token and a 6-digit PIN
    access_token = secrets.token_urlsafe(16)
    pin = "".join(secrets.choice(string.digits) for _ in range(6))

    # Foydalanuvchi-do'st ID generatsiya — 2 marta retry race condition uchun
    display_id = None
    for _ in range(3):
        try:
            display_id = _generate_display_id(db)
            db_candidate = Candidate(
                name=safe_name,
                summary=candidate.summary,
                status=candidate.status,
                access_code=access_token,
                pin_hash=get_password_hash(pin),
                owner_id=current_user.id if current_user else None,
                answers=candidate.answers,
                display_id=display_id,
            )
            db.add(db_candidate)
            db.commit()
            db.refresh(db_candidate)
            break
        except Exception as exc:
            db.rollback()
            logger.warning(f"create_candidate retry (display_id={display_id}): {exc}")
    else:
        # Barcha retrylar muvaffaqiyatsiz — display_id'siz yaratamiz
        db_candidate = Candidate(
            name=safe_name,
            summary=candidate.summary,
            status=candidate.status,
            access_code=access_token,
            pin_hash=get_password_hash(pin),
            owner_id=current_user.id if current_user else None,
            answers=candidate.answers,
        )
        db.add(db_candidate)
        db.commit()
        db.refresh(db_candidate)

    # Important: We return the plain PIN only ONCE during creation
    res = schemas.CandidateCreateResponse.from_orm(db_candidate)
    res.pin = pin
    return res

@app.post("/candidates/login")
@limiter.limit("5/minute")
def candidate_login(request: Request, access_code: str = Form(...), pin: str = Form(...), db: Session = Depends(get_db)):
    # 1. Find by long access token
    candidate = db.query(Candidate).filter(Candidate.access_code == access_code).first()
    if not candidate:
        raise HTTPException(status_code=401, detail="Noto'g'ri havola")
    
    # 2. Verify hashed PIN
    if not candidate.pin_hash or not verify_password(pin, candidate.pin_hash):
        raise HTTPException(status_code=401, detail="PIN kod noto'g'ri")
    
    # Notify HR
    send_telegram_notification(f"🚀 <b>Кандидат вошёл в сессию!</b>\n\n👤 {candidate.name}\n🆔 ID: {candidate.id}")
    create_notification(db, f"Кандидат подключился", f"{candidate.name} (ID: {candidate.id}) вошёл в сессию", "info")

    # Notify admin about candidate joining
    _broadcast_sync({"type": "NOTIFICATION", "message": f"📢 Кандидат подключился: {candidate.name}", "timestamp": datetime.datetime.now().strftime("%H:%M:%S")})

    candidate_token = create_candidate_token(candidate.id)
    return {"status": "success", "candidate_id": candidate.id, "name": candidate.name, "candidate_token": candidate_token}


@app.post("/candidates/{candidate_id}/pairing-token")
def create_candidate_pairing_token(
    candidate_id: int,
    db: Session = Depends(get_db),
    _: database.User = Depends(require_role(["SuperAdmin", "Recruiter", "Psychologist"])),
):
    candidate = get_candidate_or_404(db, candidate_id)
    if not candidate.access_code:
        raise HTTPException(status_code=400, detail="Nomzod uchun havola kodi topilmadi")

    expires_minutes = 30
    candidate_token = create_candidate_token(candidate.id, expires_minutes=expires_minutes)
    return {
        "status": "success",
        "candidate_id": candidate.id,
        "access_code": candidate.access_code,
        "candidate_token": candidate_token,
        "expires_minutes": expires_minutes,
    }


@app.post("/candidates/login/by-token")
@limiter.limit("20/minute")
def candidate_login_by_token(
    request: Request,
    candidate_token: str = Form(...),
    access_code: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    candidate_id = _decode_candidate_token(candidate_token)
    if candidate_id is None:
        raise HTTPException(status_code=401, detail="QR token noto'g'ri yoki eskirgan")

    query = db.query(Candidate).filter(Candidate.id == candidate_id)
    if access_code:
        query = query.filter(Candidate.access_code == access_code)
        
    candidate = query.first()
    if not candidate:
        raise HTTPException(status_code=401, detail="Ссылка недействительна или кандидат не найден")

    refreshed_token = create_candidate_token(candidate.id)
    send_telegram_notification(
        f"📱 <b>Кандидат вошёл по QR</b>\n\n👤 {candidate.name}\n🆔 ID: {candidate.id}"
    )

    _broadcast_sync({"type": "NOTIFICATION", "message": f"📢 Кандидат подключился по QR: {candidate.name}", "timestamp": datetime.datetime.now().strftime("%H:%M:%S")})

    return {
        "status": "success",
        "candidate_id": candidate.id,
        "name": candidate.name,
        "candidate_token": refreshed_token,
    }

@app.put("/candidates/{candidate_id}", response_model=schemas.CandidateSchema)
def update_candidate(candidate_id: int, candidate: schemas.CandidateCreate, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    db_candidate = get_candidate_or_404(db, candidate_id)
    
    db_candidate.name = candidate.name
    db_candidate.summary = candidate.summary
    db_candidate.status = candidate.status
    db_candidate.answers = candidate.answers
    db_candidate.filters = candidate.filters
    flag_modified(db_candidate, "answers")
    flag_modified(db_candidate, "filters")

    db.commit()
    db.refresh(db_candidate)
    return db_candidate

def _delete_candidate_media(candidate_id: int, db: Session) -> dict:
    """GDPR uchun: kandidat bilan bog'liq barcha media fayllarni diskdan o'chiradi.
    DB yozuvlari CASCADE bilan o'chiriladi (visual_records), lekin
    /backend/media/frames/ va /backend/media/audio/ dagi fayllar qo'lda
    o'chirilishi kerak. Natija: o'chirilgan fayllar sonini qaytaradi."""
    stats = {"frames_deleted": 0, "audio_deleted": 0, "errors": 0}

    # 1. VisualRecord.image_url dan frame fayllarini topish
    records = db.query(database.VisualRecord).filter_by(candidate_id=candidate_id).all()
    for rec in records:
        if rec.image_url:
            fname = Path(rec.image_url).name
            try:
                fp = _safe_media_path(MEDIA_DIR / "frames", fname)
                fp.unlink(missing_ok=True)
                stats["frames_deleted"] += 1
            except HTTPException:
                # Noto'g'ri nom — e'tibor bermaymiz
                pass
            except Exception:
                stats["errors"] += 1

    # 2. Candidate.answers dagi audio_url larni topish
    cand = db.query(database.Candidate).filter_by(id=candidate_id).first()
    if cand and cand.answers:
        for ans in (cand.answers or []):
            audio_url = ans.get("audio_url") or ""
            question_audio_url = ans.get("question_audio_url") or ""
            for url in (audio_url, question_audio_url):
                if not url:
                    continue
                fname = Path(url).name
                try:
                    fp = _safe_media_path(MEDIA_AUDIO_DIR, fname)
                    fp.unlink(missing_ok=True)
                    stats["audio_deleted"] += 1
                except HTTPException:
                    pass
                except Exception:
                    stats["errors"] += 1
    return stats


@app.delete("/candidates/{candidate_id}")
def delete_candidate(
    candidate_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: database.User = Depends(require_role(["SuperAdmin", "Recruiter"])),
):
    """Nomzodni to'liq o'chirish — GDPR "o'chirilish huquqi" (right to erasure).
    DB yozuvlari (visual_records CASCADE orqali) + media fayllar (frames, audio)
    birga o'chiriladi. Audit log'ga yoziladi."""
    candidate = get_candidate_or_404(db, candidate_id)
    # Snapshot — log uchun (delete'dan keyin candidate yo'q bo'ladi)
    snapshot = {
        "id": candidate.id,
        "name": candidate.name,
        "display_id": candidate.display_id,
        "status": candidate.status,
    }
    media_stats = _delete_candidate_media(candidate_id, db)
    db.query(database.VisualRecord).filter(database.VisualRecord.candidate_id == candidate_id).delete()
    db.delete(candidate)
    db.commit()

    # Audit log (xato sukut bilan o'tadi — asosiy oqimni to'xtatmaydi)
    try:
        from utils.audit import log_audit
        log_audit(
            db, user,
            action="delete",
            entity_type="candidate",
            entity_id=str(candidate_id),
            entity_label=snapshot["name"],
            details={"deleted": snapshot, "media": media_stats},
            request=request,
        )
    except Exception:
        pass

    return {"status": "deleted", "media": media_stats}


@app.get("/candidates/{candidate_id}/gdpr-export")
def gdpr_export_candidate(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    """GDPR "portativlik huquqi" (right to portability) — kandidat ma'lumotlarini
    JSON formatida eksport qiladi. Admin yoki kandidat o'zi so'rashi mumkin.
    Media URL lar ham qaytariladi, lekin faylning o'zi emas (alohida endpoint orqali)."""
    candidate = get_candidate_or_404(db, candidate_id)
    visuals = db.query(database.VisualRecord).filter_by(candidate_id=candidate_id).all()

    return {
        "export_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "candidate": {
            "id": candidate.id,
            "name": candidate.name,
            "summary": candidate.summary,
            "status": candidate.status,
            "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
            "answers": candidate.answers,
            "filters": candidate.filters,
        },
        "visual_records": [
            {
                "id": v.id,
                "emotion": v.emotion,
                "stress_level": v.stress_level,
                "notes": v.notes,
                "image_url": v.image_url,
                "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            }
            for v in visuals
        ],
        "data_controller": {
            "note": "Ushbu ma'lumotlar GDPR art. 20 asosida portativ formatda taqdim etiladi",
        },
    }

# --- Chat Endpoints ---

@app.get("/chat/", response_model=List[schemas.ChatMessageSchema])
def get_chat_history(db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    return db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()

def _celery_available() -> bool:
    """Runtime tekshiruv — Celery worker ulanmagan bo'lsa sync fallback."""
    if not _CELERY_IMPORTED:
        return False
    try:
        return celery_enabled()
    except Exception:
        return False


def _submit_chat_reply(user_message_id: int, prompt: str) -> Optional[str]:
    """Celery ga AI chat javob taskini yuboradi yoki threading fallback ishlatadi.
    Celery task ID ni qaytaradi (fallback holatda None)."""
    if _celery_available():
        try:
            result = generate_ai_reply_task.delay(
                user_message_id=user_message_id, prompt=prompt
            )
            return result.id
        except Exception as exc:
            logger.warning(f"Celery submit failed, falling back to thread: {exc}")

    # Fallback — threading (eski xulq, faqat Celery mavjud bo'lmaganda)
    import threading as _threading
    import datetime as _dt

    def _run():
        try:
            ai_text = logic.ask_mistral_raw(prompt)
        except Exception:
            ai_text = "AI сервер временно недоступен. Попробуйте позже."
        try:
            with SessionLocal() as ai_db:
                ai_db.add(ChatMessage(
                    role="assistant",
                    content=ai_text,
                    timestamp=_dt.datetime.now().isoformat(),
                ))
                ai_db.commit()
        except Exception as e:
            logger.error(f"AI chat reply DB write failed: {e}")

    _threading.Thread(target=_run, daemon=True).start()
    return None


def _submit_transcribe(audio_path: str, audio_url: str,
                       candidate_id: Optional[int] = None) -> str:
    """STT taskini Celery ga yuboradi va task_id qaytaradi. Fallback threading bilan."""
    if _celery_available():
        try:
            result = transcribe_audio_task.delay(
                audio_path=audio_path, audio_url=audio_url, candidate_id=candidate_id
            )
            return result.id
        except Exception as exc:
            logger.warning(f"Celery STT submit failed, fallback: {exc}")

    # Fallback threading
    import threading as _threading
    task_id = str(uuid.uuid4())

    def _run():
        try:
            text, elapsed_ms = logic.transcribe_audio(audio_path)
            with SessionLocal() as tdb:
                tdb.add(database.GlobalSetting(
                    key=f"stt_result_{task_id}",
                    value={"text": text, "elapsed_ms": elapsed_ms,
                           "audio_url": audio_url, "status": "done"},
                ))
                tdb.commit()
            _broadcast_sync({"type": "STT_RESULT", "task_id": task_id,
                             "text": text, "audio_url": audio_url,
                             "elapsed_ms": elapsed_ms})
        except Exception as e:
            logger.warning(f"Background transcribe failed: {e}")
            with SessionLocal() as tdb:
                tdb.add(database.GlobalSetting(
                    key=f"stt_result_{task_id}",
                    value={"text": "", "error": str(e),
                           "audio_url": audio_url, "status": "error"},
                ))
                tdb.commit()

    _threading.Thread(target=_run, daemon=True).start()
    return task_id


def _submit_process_turn(candidate_id: int, turn_uid: str, question: str,
                          audio_path: str, audio_url: str,
                          parsed_face_stats: Optional[dict]) -> Optional[str]:
    """Process-turn pipeline ni Celery ga yuboradi. Fallback threading bilan."""
    if _celery_available():
        try:
            result = process_turn_full_task.delay(
                candidate_id=candidate_id,
                turn_uid=turn_uid,
                question=question,
                audio_path=audio_path,
                audio_url=audio_url,
                parsed_face_stats=parsed_face_stats,
            )
            return result.id
        except Exception as exc:
            logger.warning(f"Celery process-turn submit failed, fallback: {exc}")

    # Fallback — threading orqali plain pipeline funksiyasi chaqiriladi.
    # MUHIM: bu yo'l Celery o'rnatilmagan bo'lsa ham ishlaydi (process_turn_pipeline
    # — Celery'ga bog'liq emas).
    import threading as _threading
    try:
        from tasks.process_turn_tasks import process_turn_pipeline as _pipeline
    except Exception as exc:
        logger.error(f"process_turn_pipeline import failed: {exc}")
        return None

    def _run():
        try:
            _pipeline(
                candidate_id=candidate_id,
                turn_uid=turn_uid,
                question=question,
                audio_path=audio_path,
                audio_url=audio_url,
                parsed_face_stats=parsed_face_stats,
            )
        except Exception as e:
            logger.error(f"Fallback process-turn failed: {e}")

    _threading.Thread(target=_run, daemon=True).start()
    return None


def _build_psychologist_prompt(db: Session, user_message: str) -> str:
    """Psixolog AI javobi uchun prompt quradi. Celery task bilan tashqarida
    chaqirilib, tayyor prompt satrini taskka uzatamiz (DB access worker da
    takrorlanmaydi)."""
    recent = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(10).all()
    context = "\n".join([f"{m.role}: {m.content}" for m in reversed(recent)])
    insights_setting = db.query(GlobalSetting).filter(GlobalSetting.key == "psychologist_insights").first()
    filters_setting = db.query(GlobalSetting).filter(GlobalSetting.key == "global_filters").first()
    saved_insights = insights_setting.value if insights_setting and isinstance(insights_setting.value, list) else []
    active_filters = filters_setting.value if filters_setting and isinstance(filters_setting.value, list) else []
    filters_block = (
        "Активные требования:\n- " + "\n- ".join(active_filters[:8])
        if active_filters
        else "Активные требования пока не заданы."
    )
    insights_block = (
        "Сохранённые инсайты:\n- " + "\n- ".join(saved_insights[:8])
        if saved_insights
        else "Сохранённых инсайтов пока нет."
    )
    return (
        "Ты внутренний AI-психолог платформы интервью.\n"
        "Отвечай профессионально, кратко и практично.\n"
        "Делай акцент на методологии оценки кандидата, поведенческих сигналах, рисках и улучшении процесса интервью.\n\n"
        f"{filters_block}\n\n"
        f"{insights_block}\n\n"
        f"Недавний контекст диалога:\n{context}\n\n"
        f"Запрос психолога:\n{user_message}"
    )


@app.post("/chat/")
@limiter.limit(RL_CHAT)
def add_chat_message(
    request: Request,
    msg: schemas.ChatMessageCreate,
    generate_reply: bool = False,
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    import datetime as dt

    # Save user message
    db_msg = ChatMessage(
        role=msg.role,
        content=msg.content,
        timestamp=dt.datetime.now().isoformat()
    )
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)

    user_msg = {"id": db_msg.id, "role": db_msg.role, "content": db_msg.content, "timestamp": db_msg.timestamp}

    # Generate assistant reply only when the caller explicitly asks for it.
    if msg.role == "user" and generate_reply:
        prompt = _build_psychologist_prompt(db, msg.content)
        _submit_chat_reply(user_message_id=db_msg.id, prompt=prompt)

    return user_msg


@app.get("/logic/provider-status/")
def ai_provider_status(_: database.User = Depends(require_admin)):
    return logic.get_ai_runtime_status()


@app.delete("/chat/")
def clear_chat_history(db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    db.query(ChatMessage).delete()
    db.commit()
    return {"status": "cleared"}

@app.delete("/chat/{message_id}")
def delete_chat_message(message_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Сообщение не найдено")
    db.delete(msg)
    db.commit()
    return {"status": "deleted"}

# --- Settings Endpoints ---

@app.get("/settings/{key}", response_model=schemas.GlobalSettingSchema)
def get_setting(key: str, db: Session = Depends(get_db)):
    setting = db.query(GlobalSetting).filter(GlobalSetting.key == key).first()
    if setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting

@app.post("/settings/", response_model=schemas.GlobalSettingSchema)
def set_setting(setting: schemas.GlobalSettingBase, db: Session = Depends(get_db), _: database.User = Depends(require_role(["SuperAdmin", "Recruiter"]))):
    db_setting = db.query(GlobalSetting).filter(GlobalSetting.key == setting.key).first()
    if db_setting:
        db_setting.value = setting.value
    else:
        db_setting = GlobalSetting(key=setting.key, value=setting.value)
        db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

@app.delete("/settings/{key}")
def delete_setting(key: str, db: Session = Depends(get_db), _: database.User = Depends(require_role(["SuperAdmin"]))):
    setting = db.query(GlobalSetting).filter(GlobalSetting.key == key).first()
    if not setting:
        raise HTTPException(status_code=404, detail="Настройка не найдена")
    db.delete(setting)
    db.commit()
    return {"status": "deleted"}

# --- Logic Endpoints (AI) ---

@app.post("/logic/upload-audio/")
@limiter.limit(RL_UPLOAD_AUDIO)
def upload_audio_api(request: Request, file: UploadFile = File(...), _: database.User = Depends(require_admin)):
    """Upload audio file and return URL immediately. No STT processing."""
    ext = _validate_upload_mime(
        file, ALLOWED_AUDIO_MIMES, ALLOWED_AUDIO_EXTS, label="audio", default_ext=".wav"
    )
    audio_filename = f"{secrets.token_hex(16)}{ext}"
    save_path = MEDIA_AUDIO_DIR / audio_filename
    _stream_upload_to_path(file, save_path, MAX_AUDIO_UPLOAD_BYTES, label="audio")
    return {"audio_url": f"/media/audio/{audio_filename}"}


_STT_EXEC_TIMEOUT = int(os.getenv("CELERY_TASK_STT_TIMEOUT", "120"))
_LLM_EXEC_TIMEOUT = int(os.getenv("CELERY_TASK_RAG_TIMEOUT", "60"))


@app.post("/logic/transcribe/")
@limiter.limit(RL_TRANSCRIBE)
async def transcribe_audio_api(request: Request, file: UploadFile = File(...), save: bool = False, _: database.User = Depends(require_admin)):
    """Audio fayl transkripsiya. `save=true` bo'lsa Celery task (immediate return),
    `save=false` bo'lsa sync — lekin **bounded STT pool** (max 4 parallel) orqali.
    Pool to'lsa 503, exec timeoutda 504."""
    ext = _validate_upload_mime(
        file, ALLOWED_AUDIO_MIMES, ALLOWED_AUDIO_EXTS, label="audio", default_ext=".wav"
    )

    audio_url = None
    if save:
        audio_filename = f"{secrets.token_hex(16)}{ext}"
        save_path = MEDIA_AUDIO_DIR / audio_filename
        _stream_upload_to_path(file, save_path, MAX_AUDIO_UPLOAD_BYTES, label="audio")
        tmp_path = str(save_path)
        audio_url = f"/media/audio/{audio_filename}"
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.close()
        tmp_dest = Path(tmp.name)
        try:
            _stream_upload_to_path(file, tmp_dest, MAX_AUDIO_UPLOAD_BYTES, label="audio")
        except HTTPException:
            raise
        tmp_path = str(tmp_dest)

    # save=true — Celery task queue (immediate return)
    if save and audio_url:
        task_id = _submit_transcribe(audio_path=tmp_path, audio_url=audio_url)
        return {"text": "", "elapsed_ms": 0, "audio_url": audio_url, "task_id": task_id, "status": "processing"}

    # save=false — sync, lekin bounded STT pool orqali
    try:
        text, elapsed_ms = await run_bounded(
            stt_executor,
            logic.transcribe_audio, tmp_path,
            exec_timeout_sec=_STT_EXEC_TIMEOUT,
        )
        result = {"text": text, "elapsed_ms": elapsed_ms}
        if audio_url:
            result["audio_url"] = audio_url
        return result
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"STT {_STT_EXEC_TIMEOUT}s ichida tugamadi")
    except logic.TranscriptionError as exc:
        if audio_url:
            raise HTTPException(status_code=400, detail={"message": str(exc), "audio_url": audio_url})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Transcription error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"STT error: {exc}") from exc
    finally:
        if not save and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/logic/stt-result/{task_id}")
def get_stt_result(task_id: str, db: Session = Depends(get_db)):
    """STT natijasini polling orqali olish.

    IDEMPOTENT — bir necha marta poll qilish mumkin (frontend polling +
    WebSocket broadcast race oldini olish). Avval birinchi o'qishda DB'dan
    o'chirib yuborardi — bu Celery cross-process WS broadcast yo'qotganda
    polling fallback'i ishlamasligiga olib kelardi.

    Eski stt_result_* yozuvlari saqlanadi (har biri ~200 byte). Periodik
    cleanup boshqa joyda amalga oshirilishi mumkin (hozirda yo'q —
    foydalanuvchi sessiya davomida bu arziydi).
    """
    setting = db.query(GlobalSetting).filter(GlobalSetting.key == f"stt_result_{task_id}").first()
    if not setting:
        return {"status": "processing"}
    return setting.value

@app.post("/logic/analyze/")
async def analyze_answer_api(question: str, answer: str, _: database.User = Depends(require_admin)):
    """Mistral RAG analiz — bounded LLM pool orqali."""
    try:
        analysis = await run_bounded(
            llm_executor,
            logic.analyze_answer, question, answer,
            exec_timeout_sec=_LLM_EXEC_TIMEOUT,
        )
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"LLM {_LLM_EXEC_TIMEOUT}s ichida javob bermadi")
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"analysis": analysis}


@app.post("/logic/ask/")
async def ask_mistral_api(prompt: str, _: database.User = Depends(require_admin)):
    """To'g'ridan Mistral chaqiruvi — bounded LLM pool orqali."""
    try:
        response = await run_bounded(
            llm_executor,
            logic.ask_mistral_raw, prompt,
            exec_timeout_sec=_LLM_EXEC_TIMEOUT,
        )
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"LLM {_LLM_EXEC_TIMEOUT}s ichida javob bermadi")
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"response": response}


@app.post("/logic/summary/")
async def generate_summary_api(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    """Intervyu xulosasi — bounded LLM pool orqali."""
    candidate = get_candidate_or_404(db, candidate_id)

    try:
        summary = await run_bounded(
            llm_executor,
            logic.build_interview_summary, candidate.answers,
            exec_timeout_sec=_LLM_EXEC_TIMEOUT,
        )
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Summary {_LLM_EXEC_TIMEOUT}s ichida tugamadi")
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    candidate.summary = summary
    db.commit()
    return {"summary": summary}


@app.patch("/candidates/{candidate_id}/answers/{turn_uid}")
def patch_answer_field(
    candidate_id: int,
    turn_uid: str,
    question_audio_url: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    """Update specific fields of an answer by turn_uid (e.g. question_audio_url after HR STT)."""
    candidate = get_candidate_or_404(db, candidate_id)
    answers = list(candidate.answers or [])
    updated = False
    for ans in answers:
        if ans.get("turn_uid") == turn_uid:
            if question_audio_url is not None:
                ans["question_audio_url"] = question_audio_url
            if question is not None:
                ans["question"] = question
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Answer not found")
    candidate.answers = answers
    flag_modified(candidate, "answers")
    db.commit()
    return {"status": "updated"}


@app.post("/logic/process-turn/")
@limiter.limit(RL_PROCESS_TURN)
def process_turn_api(
    request: Request,
    candidate_id: int = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
    question_audio_url: Optional[str] = Form(None),
    face_stats: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    _: database.User = Depends(require_admin),
):
    try:
        candidate = get_candidate_or_404(db, candidate_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"process-turn candidate lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Create permanent audio storage path (MIME + hajm tekshiruvi bilan)
    ext = _validate_upload_mime(
        file, ALLOWED_AUDIO_MIMES, ALLOWED_AUDIO_EXTS, label="audio", default_ext=".wav"
    )

    audio_filename = f"{secrets.token_hex(16)}{ext}"
    save_path = MEDIA_AUDIO_DIR / audio_filename
    _stream_upload_to_path(file, save_path, MAX_AUDIO_UPLOAD_BYTES, label="audio")

    # INSTANT RESPONSE — save audio, return immediately, process everything in background
    audio_path = str(save_path)
    turn_uid = str(uuid.uuid4())
    # Parse + validate face stats from JSON.
    # Schema: {gaze_focused_pct, gaze_away_pct, mouth_open_pct, eyes_closed_pct,
    #          face_not_found_pct} — har biri 0-100 oralig'ida float.
    parsed_face_stats = None
    if face_stats:
        try:
            raw = json.loads(face_stats)
            if isinstance(raw, dict):
                allowed_numeric_keys = {
                    "gaze_focused_pct", "gaze_away_pct", "mouth_open_pct",
                    "eyes_closed_pct", "face_not_found_pct", "duration_sec", "total",
                    "avg_stress_score",  # blendshape-asoslangan stress 0-100
                }
                allowed_string_keys = {
                    "dominant_emotion",  # blendshape-asoslangan emotion (Спокойный, Радость va h.k.)
                }
                clean: Dict[str, Any] = {}
                for k, v in raw.items():
                    if k in allowed_string_keys:
                        # Faqat oddiy ascii/cyrillic so'z, max 32 char
                        if isinstance(v, str) and len(v) <= 32:
                            clean[k] = v
                        continue
                    if k not in allowed_numeric_keys:
                        continue
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    if f != f or f in (float("inf"), float("-inf")):
                        continue
                    if k.endswith("_pct") or k == "avg_stress_score":
                        f = max(0.0, min(100.0, f))
                    clean[k] = round(f, 1)
                parsed_face_stats = clean if clean else None
        except Exception:
            pass

    basic_result = {
        "turn_uid": turn_uid,
        "question": question,
        "answer": "⏳ Обработка...",
        "ai": "",
        "next_suggestion": "",
        "voice_raw": "",
        "candidate_raw": "",
        "audio_url": f"/media/audio/{audio_filename}",
        "question_audio_url": question_audio_url or "",
        "face_stats": parsed_face_stats,
        "stt_ms": 0,
    }
    # Row-level lock — concurrent process_turn so'rovlari `answers` JSON ni
    # bir-birining ustidan yozib qo'ymasligi uchun. PostgreSQL'da SELECT FOR UPDATE,
    # SQLite ignore qiladi (lekin SQLite'da yagona writer bor).
    try:
        cand_locked = (
            db.query(database.Candidate)
            .filter_by(id=candidate.id)
            .with_for_update()
            .first()
        )
        if not cand_locked:
            raise HTTPException(status_code=404, detail="Кандидат не найден (concurrent delete?)")
        answers = list(cand_locked.answers or [])
        answers.append(basic_result.copy())
        cand_locked.answers = answers
        flag_modified(cand_locked, "answers")
        db.commit()
        candidate = cand_locked  # rest of function uses `candidate`
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить: {e}")

    # ALL PROCESSING IN BACKGROUND — Whisper + prosody + AI (Celery task queue)
    try:
        task_id = _submit_process_turn(
            candidate_id=candidate_id,
            turn_uid=turn_uid,
            question=question,
            audio_path=audio_path,
            audio_url=f"/media/audio/{audio_filename}",
            parsed_face_stats=parsed_face_stats,
        )
        if task_id:
            basic_result["job_id"] = task_id
    except Exception as e:
        logger.error(f"Failed to submit process-turn job: {e}")

    return basic_result


# Helper to validate password strength (#57)
# Davlat tizimi uchun: 10+ belgi, kichik+katta+raqam+maxsus
def validate_password_strength(password: str):
    if len(password) < 10:
        raise HTTPException(status_code=400, detail="Пароль должен содержать минимум 10 символов")
    if not re.search("[a-z]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы одну строчную букву")
    if not re.search("[A-Z]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы одну заглавную букву")
    if not re.search("[0-9]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы одну цифру")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\",.<>/?\\|`~]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы один спецсимвол (!@#$...)")
    # Eng keng tarqalgan zaif parollarni rad etamiz
    common_weak = {"password", "12345678", "qwerty123", "admin1234", "letmein01"}
    if password.lower() in common_weak:
        raise HTTPException(status_code=400, detail="Этот пароль слишком распространён")
    return True


# ===== Account lockout (#55) — brute force himoya =====
# In-memory tracker (single-instance). Multi-instance uchun Redis kerak.
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}  # email → [timestamp, ...]
_LOCKOUT_THRESHOLD = 5         # 5 marta xato
_LOCKOUT_WINDOW_SEC = 900      # 15 daqiqa oynada
_LOCKOUT_DURATION_SEC = 900    # 15 daqiqa bloklash

def check_account_lockout(email: str):
    """Email bo'yicha urinishlar sonini tekshirish. Cheklov oshilgan
    bo'lsa 429 qaytaradi. Foydalanish: login endpoint boshida chaqirish."""
    import time as _time
    now = _time.time()
    history = _LOGIN_ATTEMPTS.get(email, [])
    # Eski yozuvlarni tozalash
    history = [t for t in history if now - t < _LOCKOUT_WINDOW_SEC]
    _LOGIN_ATTEMPTS[email] = history
    if len(history) >= _LOCKOUT_THRESHOLD:
        oldest = history[0]
        retry_after = int(_LOCKOUT_DURATION_SEC - (now - oldest))
        if retry_after > 0:
            raise HTTPException(
                status_code=429,
                detail=f"Слишком много попыток входа. Попробуйте через {retry_after // 60 + 1} мин.",
                headers={"Retry-After": str(retry_after)},
            )

def record_login_failure(email: str):
    """Login muvaffaqiyatsiz bo'lganda chaqiriladi."""
    import time as _time
    _LOGIN_ATTEMPTS.setdefault(email, []).append(_time.time())

def clear_login_attempts(email: str):
    """Muvaffaqiyatli loginda urinishlar tarixi tozalanadi."""
    _LOGIN_ATTEMPTS.pop(email, None)

def _prune_old_frames(frames_dir: Path, max_age_sec: int) -> None:
    """Best-effort cleanup — delete JPEGs older than max_age_sec."""
    import time as _time
    cutoff = _time.time() - max_age_sec
    for p in frames_dir.glob("*.jpg"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            continue


@app.post("/logic/analyze-frame/")
@limiter.limit(RL_ANALYZE_FRAME)
async def analyze_frame_api(request: Request, candidate_id: int, file: UploadFile = File(...)):
    """#27 — endi async + bounded `frame_executor` (max 4 paralel Haar cascade).
    20+ paralel kandidat bo'lsa thread pool exhaustion bo'lmaydi."""
    from utils.face_analyzer import analyze_frame

    _validate_upload_mime(
        file, ALLOWED_IMAGE_MIMES, ALLOWED_IMAGE_EXTS, label="rasm", default_ext=".jpg"
    )
    image_bytes = _read_upload_bytes(file, MAX_IMAGE_UPLOAD_BYTES, label="rasm")
    try:
        result = await run_bounded(
            frame_executor, analyze_frame, image_bytes, exec_timeout_sec=10
        )
    except QueueFull:
        raise HTTPException(status_code=503, detail="Frame analysis pool busy")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Frame analysis timeout")

    # Persist frame + visual record so admins can see the live feed and so
    # the answer analyzer can pull emotion/stress signals from this window.
    if result.get("face_detected"):
        try:
            frames_dir = MEDIA_DIR / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{secrets.token_hex(12)}.jpg"
            file_path = frames_dir / filename
            file_path.write_bytes(image_bytes)
            image_url = f"/media/frames/{filename}"

            with SessionLocal() as frame_db:
                record = database.VisualRecord(
                    candidate_id=candidate_id,
                    emotion=result.get("primary_emotion"),
                    stress_level=result.get("stress_level"),
                    image_url=image_url,
                    notes=f"Взгляд: {result.get('gaze_direction', '—')}",
                )
                frame_db.add(record)
                frame_db.commit()

            # Eski 1% probabilistic cleanup endi periodic background task'ga
            # ko'chirildi — har 5 daqiqada deterministic ishlaydi (frame disk
            # to'ldirish xavfini yo'q qiladi). _frames_cleanup_watcher ga qarang.
        except Exception as e:
            logger.warning(f"Visual record save failed: {e}")

    return result

@app.post("/logic/face-ai-analysis/")
@limiter.limit(RL_FACE_AI)
def face_ai_analysis(
    request: Request,
    gaze_focused_pct: float = Form(0),
    gaze_away_pct: float = Form(0),
    mouth_open_pct: float = Form(0),
    eyes_closed_pct: float = Form(0),
    face_not_found_pct: float = Form(0),
    duration_sec: int = Form(0),
    _: database.User = Depends(require_admin),
):
    """AI analysis of face behavior during interview segment."""
    prompt = f"""Вы — профессиональный психолог. Проанализируйте поведение кандидата на основе данных видеонаблюдения за {duration_sec} секунд.

ДАННЫЕ:
- Взгляд сфокусирован: {gaze_focused_pct:.0f}%
- Взгляд отведён: {gaze_away_pct:.0f}%
- Рот открыт (говорит): {mouth_open_pct:.0f}%
- Глаза закрыты: {eyes_closed_pct:.0f}%
- Лицо не найдено: {face_not_found_pct:.0f}%

ВЕРНИТЕ КРАТКИЙ АНАЛИЗ НА РУССКОМ (2-3 предложения):
1. Уровень вовлечённости и внимания
2. Признаки стресса или дискомфорта
3. Общая оценка невербального поведения"""

    try:
        result = logic._call_ai(prompt)
        return {"analysis": result}
    except Exception:
        return {"analysis": "Анализ недоступен"}


@app.websocket("/ws/live-analysis/")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
    # #16 — token cookie'dan ham olinadi (HttpOnly auth)
    token = _ws_extract_token(websocket, token)
    # Rate limit — har user/IP uchun max 5 connection / 60 sek
    rl_key = _ws_rate_key_from_token(token, websocket.client.host if websocket.client else "unknown")
    if not _ws_rate_limit_check(rl_key):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    # Verify token before accepting connection
    if not token:
        await websocket.close(code=4001) # Unauthorized
        return
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            await websocket.close(code=4001)
            return
    except JWTError:
        await websocket.close(code=4001)
        return

    await manager.connect(websocket)
    # #25 — per-message rate limit (anti-flood). 30 msg/10sek max.
    import time as _t
    msg_timestamps: List[float] = []
    MAX_MSG = 30
    WINDOW = 10.0
    try:
        while True:
            await websocket.receive_text()
            now = _t.time()
            msg_timestamps[:] = [t for t in msg_timestamps if now - t < WINDOW]
            if len(msg_timestamps) >= MAX_MSG:
                await websocket.send_json(
                    {"type": "error", "message": "Слишком много сообщений в секунду"}
                )
                await websocket.close(code=1008)
                break
            msg_timestamps.append(now)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket, token: Optional[str] = None):
    """Per-user notification stream. JWT cookie yoki ``?token=...`` query param."""
    token = _ws_extract_token(websocket, token)  # #16
    # Rate limit
    rl_key = _ws_rate_key_from_token(token, websocket.client.host if websocket.client else "unknown")
    if not _ws_rate_limit_check(rl_key):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    if not token:
        await websocket.close(code=4001)
        return
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            await websocket.close(code=4001)
            return
    except JWTError:
        await websocket.close(code=4001)
        return

    with SessionLocal() as db:
        user = db.query(database.User).filter_by(email=email).first()
        if not user or not user.is_active:
            await websocket.close(code=4003)
            return
        user_id = user.id

    from utils.notifications import hub as _notif_hub
    await websocket.accept()
    await _notif_hub.register(user_id, websocket)
    try:
        # Initial hello so the client can confirm the stream is live.
        await websocket.send_json({"type": "ready", "user_id": user_id})
        while True:
            # Keep-alive loop — client may send pings, we don't care about content.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"[notifications_ws] error for user={user_id}: {exc}")
    finally:
        _notif_hub.unregister(user_id, websocket)


@app.websocket("/ws/webrtc/{candidate_id}")
async def webrtc_signaling(websocket: WebSocket, candidate_id: int, token: Optional[str] = None):
    logger.info(f"[WebRTC] New connection request for candidate_id={candidate_id}, token_present={bool(token)}")
    token = _ws_extract_token(websocket, token)  # #16 — cookie support
    # Rate limit
    rl_key = _ws_rate_key_from_token(token, websocket.client.host if websocket.client else "unknown")
    if not _ws_rate_limit_check(rl_key):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    if not token:
        logger.warning(f"[WebRTC] Connection rejected: No token provided for candidate_id={candidate_id}")
        await websocket.close(code=4001)
        return

    # Identify actor (admin user vs candidate) via JWT.
    actor_type: Optional[str] = None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        role = payload.get("role")
        sub = payload.get("sub")
        logger.info(f"[WebRTC] Token decoded: role={role}, sub={sub}")

        if role == "Candidate" and isinstance(payload.get("candidate_id"), int):
            if int(payload.get("candidate_id")) != int(candidate_id):
                logger.warning(f"[WebRTC] Candidate ID mismatch: payload={payload.get('candidate_id')} != URL={candidate_id}")
                await websocket.close(code=4003)
                return
            actor_type = "candidate"
            actor_email = None
        else:
            # Admin JWT uses sub=email and role=user.role
            email = sub
            if not isinstance(email, str) or "@" not in email:
                logger.warning(f"[WebRTC] Invalid admin email in token: {email}")
                await websocket.close(code=4001)
                return
            # IMPORTANT: do not keep a DB session open for the duration of the websocket.
            # We do a one-off check and immediately close.
            with SessionLocal() as session:
                user = session.query(database.User).filter_by(email=email).first()
                if not user or not user.is_active:
                    logger.warning(f"[WebRTC] Admin not found or inactive: {email}")
                    await websocket.close(code=4001)
                    return
                if user.role not in {"SuperAdmin", "Recruiter", "Psychologist"}:
                    logger.warning(f"[WebRTC] Unauthorized role: {user.role}")
                    await websocket.close(code=4003)
                    return
            actor_type = "admin"
            actor_email = email

        logger.info(f"[WebRTC] Connection authorized: actor_type={actor_type}, candidate_id={candidate_id}")

    except JWTError as e:
        logger.error(f"[WebRTC] JWT Decode Error: {e}")
        await websocket.close(code=4001)
        return
    except Exception as e:
        logger.error(f"[WebRTC] Auth Error: {e}")
        await websocket.close(code=4001)
        return

    # Kandidat DB'da haqiqatan mavjudligini tekshirish — fantom in-memory
    # room'larni oldini olish (yo'q kandidatga ulanib bo'lmaydi).
    # Auth'dan KEYIN tekshiramiz (anonymous probe'ni oldini olish uchun).
    with SessionLocal() as session:
        cand_exists = session.query(Candidate.id).filter_by(id=candidate_id).first()
        if not cand_exists:
            logger.warning(f"[WebRTC] Candidate {candidate_id} not found in DB — closing")
            await websocket.close(code=4404, reason="Candidate not found")
            return

    await websocket.accept()
    room = _get_room(int(candidate_id))
    
    if actor_type == "admin":
        # Admin takeover qarshi himoya: agar boshqa admin allaqachon ulangan bo'lsa
        # va u BOSHQA email bo'lsa — yangi ulanishni rad etamiz (silent takeover'ni
        # oldini oladi). Bir xil admin reconnect qilsa — replace OK (yangi browser tab).
        if room.admin and room.admin_email and room.admin_email != actor_email:
            logger.warning(
                f"[WebRTC] Room {candidate_id}: takeover blocked — "
                f"existing admin '{room.admin_email}' vs new '{actor_email}'"
            )
            await websocket.close(code=4003, reason="Another admin is already in this room")
            return
        if room.admin:
            logger.info(f"[WebRTC] Room {candidate_id}: Replacing existing admin connection (same admin)")
            try:
                await room.admin.close()
            except Exception:
                pass
        room.admin = websocket
        room.admin_email = actor_email
    else:
        if room.candidate:
            logger.info(f"[WebRTC] Room {candidate_id}: Replacing existing candidate connection")
            try:
                await room.candidate.close()
            except Exception:
                pass
        room.candidate = websocket

    logger.info(f"[WebRTC] {actor_type.capitalize()} joined room {candidate_id}")

    # Flush any buffered messages destined for this side.
    try:
        for buffered in room.flush_for(actor_type):
            await websocket.send_json(buffered)
    except Exception as e:
        logger.error(f"[WebRTC] Error flushing buffer for {actor_type}: {e}")

    # Single notification: tell each side about the other (exactly once)
    other = room.other(websocket)
    if other:
        try:
            peer_type = "candidate" if actor_type == "admin" else "admin"
            # Tell the newcomer that the peer is already here
            await websocket.send_json({"type": f"{peer_type}_joined"})
            # Tell the existing peer that the newcomer arrived
            await other.send_json({"type": f"{actor_type}_joined"})
            logger.info(f"[WebRTC] Notified both sides in room {candidate_id}")
        except Exception as e:
            logger.error(f"[WebRTC] Join notification failed: {e}")

    try:
        while True:
            msg = await websocket.receive_json()
            # Forward only whitelisted signaling message types.
            msg_type = msg.get("type")
            if msg_type not in {"offer", "answer", "ice", "ready", "ping", "end"}:
                continue
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            target = room.other(websocket)
            if target:
                logger.info(f"[WebRTC] Forwarding {msg_type} from {actor_type}")
                await target.send_json(msg)
                if msg_type == "end":
                    break
            else:
                logger.info(f"[WebRTC] No peer connected, buffering {msg_type} from {actor_type}")
                # Buffer until peer connects (prevents "lost offer" when one side joins later).
                room.enqueue_for("candidate" if actor_type == "admin" else "admin", msg)
                if msg_type == "end":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup.
        if room.admin == websocket:
            room.admin = None
        if room.candidate == websocket:
            room.candidate = None
        if room.admin is None and room.candidate is None:
            webrtc_rooms.pop(int(candidate_id), None)

@app.post("/auth/login")
@limiter.limit("5/minute")
def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    # #55 Account lockout — 5 marta xato login bo'lsa 15 daq blok
    check_account_lockout(email)

    user = db.query(database.User).filter(database.User.email == email).first()
    if not user or not verify_password(password, user.password):
        # Xato urinish — tracker'ga yoziladi
        record_login_failure(email)
        # Audit log (faqat email saqlanadi, parol emas)
        try:
            from utils.audit import log_audit
            log_audit(
                db, None,
                action="login_failed",
                entity_type="auth",
                entity_label=email,
                request=request,
            )
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    # Muvaffaqiyatli login — lockout tarixi tozalanadi
    clear_login_attempts(email)

    # Track login activity
    user.login_count = (user.login_count or 0) + 1
    user.last_login = datetime.datetime.utcnow()
    db.commit()

    # Audit log — kim qaysi IP'dan kirdi
    try:
        from utils.audit import log_audit
        log_audit(
            db, user,
            action="login",
            entity_type="auth",
            entity_label=email,
            request=request,
        )
    except Exception:
        pass

    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
    )

    # JWT'ni HttpOnly cookie sifatida ham yuboramiz — XSS hujum tokenni
    # localStorage'dan ololmaydi. Frontend asta-sekin Bearer header'dan
    # cookie'ga ko'chadi (backward compat: ikkalasi ham ishlaydi).
    is_prod = os.getenv("ENVIRONMENT", "").lower() in {"production", "prod", "staging", "live"}
    response.set_cookie(
        key="admin_token",
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=is_prod,           # HTTPS-only in prod
        samesite="lax",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }


@app.post("/auth/logout")
def logout(
    response: Response,
    token: str = Depends(oauth2_scheme),
    admin_token_cookie: Optional[str] = Cookie(default=None, alias="admin_token"),
):
    """Cookie'ni o'chiradi VA token'ni revocation list'ga qo'shadi (#19).
    Endi token expiry'gacha amal qilmaydi — logout darhol yaroqsiz qiladi."""
    effective_token = admin_token_cookie or token
    if effective_token:
        try:
            payload = jwt.decode(effective_token, SECRET_KEY, algorithms=[ALGORITHM])
            jti = payload.get("jti")
            if jti:
                _revoke_token(jti)
        except JWTError:
            pass
    response.delete_cookie("admin_token")
    return {"status": "logged_out"}

@app.post("/users/register", response_model=schemas.UserSchema)
def register_user(name: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    safe_name = bleach.clean(name.strip(), tags=[], strip=True)
    safe_email = email.strip().lower()

    if not safe_name:
        raise HTTPException(status_code=400, detail="Имя не может быть пустым")
    if not safe_email:
        raise HTTPException(status_code=400, detail="Email не может быть пустым")

    validate_password_strength(password)

    try:
        existing = db.query(database.User).filter(database.User.email == safe_email).first()
        if existing:
            raise HTTPException(status_code=409, detail="Этот email уже зарегистрирован")

        db_user = database.User(
            name=safe_name,
            email=safe_email,
            password=get_password_hash(password),
            role="Recruiter"
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except HTTPException:
        raise
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Email уже существует: {safe_email}") from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка регистрации: {exc}") from exc

# --- User Management ---
@app.get("/auth/profile", response_model=schemas.UserSchema)
def get_profile(user: database.User = Depends(require_admin)):
    return user

@app.put("/auth/profile")
def update_profile(
    name: str = Form(None),
    email: str = Form(None),
    db: Session = Depends(get_db),
    user: database.User = Depends(require_admin),
):
    if name:
        user.name = bleach.clean(name.strip(), tags=[], strip=True)
    if email:
        safe_email = email.strip().lower()
        existing = db.query(database.User).filter(database.User.email == safe_email, database.User.id != user.id).first()
        if existing:
            raise HTTPException(status_code=409, detail="Этот email уже используется")
        user.email = safe_email
    db.commit()
    db.refresh(user)
    return {"status": "updated", "name": user.name, "email": user.email}

@app.post("/auth/change-password")
def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db),
    user: database.User = Depends(require_admin),
):
    if not verify_password(old_password, user.password):
        raise HTTPException(status_code=400, detail="Текущий пароль неверный")
    validate_password_strength(new_password)
    user.password = get_password_hash(new_password)
    db.commit()
    return {"status": "password_changed"}


@app.get("/users/", response_model=List[schemas.UserSchema])
def read_users(db: Session = Depends(get_db), admin: database.User = Depends(require_role(["SuperAdmin"]))):
    return db.query(database.User).order_by(database.User.id.asc()).all()

@app.post("/users/", response_model=schemas.UserSchema)
def create_user(name: str = Form(...), email: str = Form(...), password: str = Form(...), role: str = Form("Recruiter"), db: Session = Depends(get_db), admin: database.User = Depends(require_role(["SuperAdmin"]))):
    existing = db.query(database.User).filter(database.User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")

    db_user = database.User(name=name, email=email, password=get_password_hash(password), role=role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=schemas.UserSchema)
def get_user(user_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_role(["SuperAdmin"]))):
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return user

@app.put("/users/{user_id}", response_model=schemas.UserSchema)
def update_user(
    user_id: int,
    name: str = Form(None),
    email: str = Form(None),
    role: str = Form(None),
    is_active: bool = Form(None),
    db: Session = Depends(get_db),
    admin: database.User = Depends(require_role(["SuperAdmin"])),
):
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    if name is not None:
        user.name = bleach.clean(name.strip(), tags=[], strip=True)
    if email is not None:
        safe_email = email.strip().lower()
        existing = db.query(database.User).filter(database.User.email == safe_email, database.User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=409, detail="Этот email уже используется")
        user.email = safe_email
    if role is not None and role in ("SuperAdmin", "Recruiter", "Psychologist"):
        user.role = role
    if is_active is not None:
        user.is_active = is_active
    db.commit()
    db.refresh(user)
    return user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), admin: database.User = Depends(require_role(["SuperAdmin"]))):
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"status": "deleted"}

# --- Notifications ---

@app.get("/notifications/")
def get_notifications(db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    notifs = (
        db.query(database.Notification)
        .filter((database.Notification.user_id == user.id) | (database.Notification.user_id.is_(None)))
        .order_by(database.Notification.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": n.id,
            "title": n.title,
            "message": n.message,
            "type": n.type,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat() if n.created_at else None,
        }
        for n in notifs
    ]

@app.get("/notifications/unread-count")
def get_unread_count(db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    count = (
        db.query(database.Notification)
        .filter(
            (database.Notification.user_id == user.id) | (database.Notification.user_id.is_(None)),
            database.Notification.is_read == False,
        )
        .count()
    )
    return {"count": count}

@app.post("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int, db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    notif = db.query(database.Notification).filter_by(id=notification_id).first()
    if notif and (notif.user_id == user.id or notif.user_id is None):
        notif.is_read = True
        db.commit()
    return {"status": "ok"}

@app.post("/notifications/read-all")
def mark_all_read(db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    db.query(database.Notification).filter(
        (database.Notification.user_id == user.id) | (database.Notification.user_id.is_(None)),
        database.Notification.is_read == False,
    ).update({"is_read": True}, synchronize_session=False)
    db.commit()
    return {"status": "ok"}


@app.delete("/notifications/{notification_id}")
def delete_notification(notification_id: int, db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    notif = db.query(database.Notification).filter_by(id=notification_id).first()
    if notif and (notif.user_id == user.id or notif.user_id is None):
        db.delete(notif)
        db.commit()
    return {"status": "ok"}

@app.delete("/notifications/")
def delete_all_notifications(db: Session = Depends(get_db), user: database.User = Depends(require_admin)):
    db.query(database.Notification).filter(
        (database.Notification.user_id == user.id) | (database.Notification.user_id.is_(None))
    ).delete(synchronize_session=False)
    db.commit()
    return {"status": "ok"}


def create_notification(db_session, title: str, message: str, type: str = "info", user_id: int = None):
    """Helper to create a notification from anywhere in the backend."""
    notif = database.Notification(title=title, message=message, type=type, user_id=user_id)
    db_session.add(notif)
    db_session.commit()


# --- RAG / Knowledge Base ---

@app.get("/rag/status")
def rag_status(_: database.User = Depends(require_admin)):
    from utils.rag_service import get_collection_info
    return get_collection_info()

@app.get("/rag/documents")
def rag_list_documents(_: database.User = Depends(require_admin)):
    from utils.rag_service import get_all_documents
    return get_all_documents()

@app.post("/rag/documents")
def rag_add_document(
    text: str = Form(...),
    category: str = Form("general"),
    _: database.User = Depends(require_role(["SuperAdmin", "Recruiter"])),
):
    from utils.rag_service import add_document, ensure_collection
    ensure_collection()
    success = add_document(text, metadata={"category": category})
    if not success:
        raise HTTPException(status_code=500, detail="Не удалось добавить документ. Проверьте Qdrant настройки.")
    return {"status": "added"}

@app.post("/rag/documents/bulk")
def rag_bulk_upload(
    texts: List[str] = Form(...),
    category: str = Form("general"),
    _: database.User = Depends(require_role(["SuperAdmin"])),
):
    from utils.rag_service import add_document, ensure_collection
    ensure_collection()
    added = 0
    for text in texts:
        if text.strip() and add_document(text.strip(), metadata={"category": category}):
            added += 1
    return {"status": "ok", "added": added, "total": len(texts)}

@app.delete("/rag/documents/{doc_id}")
def rag_delete_document(doc_id: str, _: database.User = Depends(require_role(["SuperAdmin"]))):
    from utils.rag_service import delete_document
    delete_document(doc_id)
    return {"status": "deleted"}

@app.post("/rag/search")
def rag_search(query: str = Form(...), top_k: int = Form(3), _: database.User = Depends(require_admin)):
    from utils.rag_service import search_context
    context = search_context(query, top_k=top_k)
    return {"context": context}


# --- Feature Flags Management ---
@app.get("/features/", response_model=List[dict])
def get_features(db: Session = Depends(get_db)):
    # Any logged in user can view flags to adjust frontend state
    if db.query(database.FeatureFlag.id).count() == 0:
        _ensure_default_feature_flags(db)
    flags = db.query(database.FeatureFlag).all()
    return [{"id": f.id, "name": f.name, "is_enabled": f.is_enabled, "description": f.description} for f in flags]

@app.post("/features/toggle/{name}")
def toggle_feature(name: str, enabled: bool = Form(...), db: Session = Depends(get_db), admin: database.User = Depends(require_role(["SuperAdmin"]))):
    flag = db.query(database.FeatureFlag).filter_by(name=name).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Feature not found")
    flag.is_enabled = enabled
    db.commit()
    return {"name": name, "is_enabled": flag.is_enabled}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
