import os
import secrets
import string
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
import asyncio
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

import database
import schemas
import logic

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MEDIA_DIR = BACKEND_DIR / "media"
MEDIA_AUDIO_DIR = MEDIA_DIR / "audio"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_notification(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

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
DEFAULT_FEATURE_FLAGS = [
    {"name": "linkedin_search", "description": "Nomzod profilida LinkedIn qidiruv tugmasini ko'rsatish", "is_enabled": True},
    {"name": "pdf_export", "description": "Nomzod profilidan PDF hisobot eksportini yoqish", "is_enabled": True},
    {"name": "voice_tts", "description": "Nomzod sahifasida savolni ovozli o'qib berish (TTS)", "is_enabled": True},
    {"name": "interview_timer", "description": "Nomzod sessiyasida vaqt taymerini ko'rsatish", "is_enabled": True},
    {"name": "stress_overlay", "description": "Interview LIVE oynasida stress overlay effektini yoqish", "is_enabled": True},
    {"name": "gaze_tracking", "description": "AI Visual blokida gaze (nigoh) diagnostikasini ko'rsatish", "is_enabled": True},
    {"name": "ai_suggestions", "description": "Javobdan keyin AI tavsiya savollarini yaratish", "is_enabled": True},
    {"name": "vocal_analysis", "description": "Nomzod ovozi bo'yicha prosody va holat tahlilini yoqish", "is_enabled": True},
]

# --- WebRTC Signaling ---
class WebRTCRoom:
    def __init__(self):
        self.admin: Optional[WebSocket] = None
        self.candidate: Optional[WebSocket] = None
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

def _get_room(candidate_id: int) -> WebRTCRoom:
    with _rooms_lock:
        room = webrtc_rooms.get(candidate_id)
        if room is None:
            room = WebRTCRoom()
            webrtc_rooms[candidate_id] = room
        return room

def create_candidate_token(candidate_id: int, expires_minutes: int = 60 * 24) -> str:
    return create_access_token(
        {"sub": f"candidate:{candidate_id}", "role": "Candidate", "candidate_id": candidate_id},
        expires_delta=datetime.timedelta(minutes=expires_minutes),
    )

# --- Security Config ---
# Ensure SECRET_KEY is set via environment variable for production
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # In development, we can allow a default, but warn. In production, this should fail.
    if os.getenv("ENVIRONMENT") == "production":
        raise RuntimeError("FATAL: SECRET_KEY is not set in environment variables!")
    SECRET_KEY = "DEV_DEBUG_SECRET_ONLY_DO_NOT_USE_IN_PROD"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI Interview Backend API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Add Restricted CORS middleware
# Only allow known origins (add your production domain here)
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "https://interview.misterdev.uz",
    "https://www.interview.misterdev.uz",
    "https://interview-api.misterdev.uz",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media") # Unsecured mount removed


@app.on_event("startup")
def startup():
    if database.DATABASE_URL.startswith("sqlite"):
        database.init_db()
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

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# User context for RBAC using JWT
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
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

    file_path = MEDIA_DIR / "frames" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Rasm topilmadi")
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

    file_path = MEDIA_AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio fayl topilmadi")
    from fastapi.responses import FileResponse
    media_type = "audio/webm" if filename.endswith(".webm") else "audio/ogg" if filename.endswith(".ogg") else "audio/wav"
    return FileResponse(file_path, media_type=media_type)


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

@app.get("/candidates/", response_model=List[schemas.CandidateSchema])
def read_candidates(db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    candidates = db.query(Candidate).all()
    return candidates

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
    file_path = MEDIA_DIR / "frames" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frame file not found")

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

@app.post("/candidates/", response_model=schemas.CandidateCreateResponse)
def create_candidate(candidate: schemas.CandidateCreate, db: Session = Depends(get_db), current_user: database.User = Depends(require_role(["SuperAdmin", "Recruiter"]))):
    # Sanitize name
    safe_name = bleach.clean(candidate.name, tags=[], strip=True) if candidate.name else ""
    
    # Generate secure 16-char access token and a 6-digit PIN
    access_token = secrets.token_urlsafe(16)
    pin = "".join(secrets.choice(string.digits) for _ in range(6))
    
    db_candidate = Candidate(
        name=safe_name,
        summary=candidate.summary,
        status=candidate.status,
        access_code=access_token, # This is our long secure token
        pin_hash=get_password_hash(pin), # We hash the 6-digit PIN
        owner_id=current_user.id if current_user else None,
        answers=candidate.answers,
    )
    db.add(db_candidate)
    db.commit()
    db.refresh(db_candidate)
    
    # Important: We return the plain PIN only ONCE during creation
    # The frontend should display this to the recruiter
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
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager.broadcast({
            "type": "NOTIFICATION",
            "message": f"📢 Кандидат подключился: {candidate.name}",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        }))
        loop.close()
    except Exception:
        pass

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

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager.broadcast({
            "type": "NOTIFICATION",
            "message": f"📢 Кандидат подключился по QR: {candidate.name}",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        }))
        loop.close()
    except Exception:
        pass

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
    flag_modified(db_candidate, "answers")

    db.commit()
    db.refresh(db_candidate)
    return db_candidate

@app.delete("/candidates/{candidate_id}")
def delete_candidate(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_role(["SuperAdmin", "Recruiter"]))):
    candidate = get_candidate_or_404(db, candidate_id)
    # Delete related visual records
    db.query(database.VisualRecord).filter(database.VisualRecord.candidate_id == candidate_id).delete()
    db.delete(candidate)
    db.commit()
    return {"status": "deleted"}

# --- Chat Endpoints ---

@app.get("/chat/", response_model=List[schemas.ChatMessageSchema])
def get_chat_history(db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    return db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()

@app.post("/chat/")
def add_chat_message(msg: schemas.ChatMessageCreate, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    import datetime as dt
    import threading

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

    # If user message, generate AI response in background
    if msg.role == "user":
        def generate_ai_reply():
            ai_db = SessionLocal()
            try:
                # Build context from recent messages
                recent = ai_db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(10).all()
                context = "\n".join([f"{m.role}: {m.content}" for m in reversed(recent)])

                # Get AI response via Mistral
                try:
                    ai_text = logic.analyze_answer(
                        question=msg.content,
                        answer=context,
                        context="Ты — AI-ассистент психолога-методолога. Отвечай на русском языке. Анализируй поведение кандидатов, давай рекомендации по методологии интервью."
                    )
                except Exception:
                    ai_text = "AI сервер временно недоступен. Попробуйте позже."

                ai_msg = ChatMessage(
                    role="assistant",
                    content=ai_text,
                    timestamp=dt.datetime.now().isoformat()
                )
                ai_db.add(ai_msg)
                ai_db.commit()
            except Exception as e:
                logger.error(f"AI chat reply failed: {e}")
            finally:
                ai_db.close()

        thread = threading.Thread(target=generate_ai_reply, daemon=True)
        thread.start()

    return user_msg


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

@app.post("/logic/transcribe/")
def transcribe_audio_api(file: UploadFile = File(...), save: bool = False, _: database.User = Depends(require_admin)):
    ext = os.path.splitext(file.filename or "")[1]
    if not ext:
        # Fallback by content-type
        if file.content_type == "audio/webm":
            ext = ".webm"
        elif file.content_type == "audio/ogg":
            ext = ".ogg"
        else:
            ext = ".wav"

    # Save audio permanently if requested, otherwise use temp file
    audio_url = None
    if save:
        audio_filename = f"{secrets.token_hex(16)}{ext}"
        save_path = MEDIA_AUDIO_DIR / audio_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        tmp_path = str(save_path)
        audio_url = f"/media/audio/{audio_filename}"
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

    # If save=true, return immediately and process in background
    # Frontend will get audio_url now, text via polling later
    if save and audio_url:
        task_id = str(uuid.uuid4())

        import threading
        def background_transcribe():
            try:
                text, elapsed_ms = logic.transcribe_audio(tmp_path)
                # Store result in global settings for polling
                with SessionLocal() as tdb:
                    setting = database.GlobalSetting(key=f"stt_result_{task_id}", value={"text": text, "elapsed_ms": elapsed_ms, "audio_url": audio_url, "status": "done"})
                    tdb.merge(setting)
                    tdb.commit()
                # Broadcast to admin via WebSocket
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(manager.broadcast({
                        "type": "STT_RESULT",
                        "task_id": task_id,
                        "text": text,
                        "audio_url": audio_url,
                        "elapsed_ms": elapsed_ms,
                    }))
                    loop.close()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Background transcribe failed: {e}")
                with SessionLocal() as tdb:
                    setting = database.GlobalSetting(key=f"stt_result_{task_id}", value={"text": "", "error": str(e), "audio_url": audio_url, "status": "error"})
                    tdb.merge(setting)
                    tdb.commit()

        thread = threading.Thread(target=background_transcribe, daemon=True)
        thread.start()

        return {"text": "", "elapsed_ms": 0, "audio_url": audio_url, "task_id": task_id, "status": "processing"}

    # Sync mode (save=false) — wait for result
    try:
        text, elapsed_ms = logic.transcribe_audio(tmp_path)
        result = {"text": text, "elapsed_ms": elapsed_ms}
        if audio_url:
            result["audio_url"] = audio_url
        return result
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
    setting = db.query(GlobalSetting).filter(GlobalSetting.key == f"stt_result_{task_id}").first()
    if not setting:
        return {"status": "processing"}
    result = setting.value
    # Cleanup after reading
    db.delete(setting)
    db.commit()
    return result

@app.post("/logic/analyze/")
def analyze_answer_api(question: str, answer: str, _: database.User = Depends(require_admin)):
    try:
        analysis = logic.analyze_answer(question, answer)
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"analysis": analysis}

@app.post("/logic/ask/")
def ask_mistral_api(prompt: str, _: database.User = Depends(require_admin)):
    try:
        response = logic.ask_mistral_raw(prompt)
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"response": response}


@app.post("/logic/summary/")
def generate_summary_api(candidate_id: int, db: Session = Depends(get_db), _: database.User = Depends(require_admin)):
    candidate = get_candidate_or_404(db, candidate_id)

    try:
        summary = logic.build_interview_summary(candidate.answers)
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
def process_turn_api(
    candidate_id: int = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
    question_audio_url: Optional[str] = Form(None),
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

    # Create permanent audio storage path
    ext = os.path.splitext(file.filename or "")[1]
    if not ext:
        if file.content_type == "audio/webm":
            ext = ".webm"
        elif file.content_type == "audio/ogg":
            ext = ".ogg"
        else:
            ext = ".wav"

    audio_filename = f"{secrets.token_hex(16)}{ext}"
    save_path = MEDIA_AUDIO_DIR / audio_filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # INSTANT RESPONSE — save audio, return immediately, process everything in background
    audio_path = str(save_path)
    turn_uid = str(uuid.uuid4())
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
        "stt_ms": 0,
    }
    answers = list(candidate.answers or [])
    answers.append(basic_result.copy())
    candidate.answers = answers
    flag_modified(candidate, "answers")
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB commit failed in process-turn: {e}")

    # ALL PROCESSING IN BACKGROUND — Whisper + prosody + AI
    import threading

    def run_full_background_processing():
        analysis_db = SessionLocal()
        try:
            # Step 1: Whisper STT
            transcript = ""
            stt_ms = 0
            try:
                transcript, stt_ms = logic.transcribe_audio(audio_path)
            except Exception as e:
                logger.warning(f"Whisper STT failed: {e}")
                transcript = "(Речь не распознана)"

            # Step 2: Voice prosody (librosa)
            voice_raw = ""
            try:
                voice_raw = logic.run_voice_profiler(audio_path)
            except Exception as e:
                logger.warning(f"Voice profiler failed: {e}")

            # Step 3: AI analysis (Mistral + RAG)
            rag_ai = ""
            try:
                rag_ai = logic.analyze_answer(question, transcript)
            except Exception:
                rag_ai = "AI анализ недоступен"

            # Update the answer with all results
            cand = analysis_db.query(database.Candidate).filter_by(id=candidate_id).first()
            if cand:
                ans = list(cand.answers or [])
                for i, item in enumerate(ans):
                    if item.get("turn_uid") == turn_uid:
                        ans[i]["answer"] = transcript
                        ans[i]["ai"] = rag_ai
                        ans[i]["voice_raw"] = voice_raw
                        ans[i]["stt_ms"] = stt_ms
                        break
                cand.answers = ans
                flag_modified(cand, "answers")
                analysis_db.commit()

            # Broadcast ALL results to admin via WebSocket
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(manager.broadcast({
                    "type": "TURN_RESULT",
                    "candidate_id": candidate_id,
                    "question": question,
                    "answer": transcript,
                    "ai": rag_ai,
                    "voice_raw": voice_raw,
                    "audio_url": basic_result.get("audio_url", ""),
                    "turn_uid": turn_uid,
                    "stt_ms": stt_ms,
                }))
                loop.close()
            except Exception:
                pass

            # Telegram
            send_telegram_notification(
                f"📝 <b>Ответ проанализирован</b>\n❓ {question}\n💬 {transcript[:100]}...\n🧠 {rag_ai[:150]}"
            )
        except Exception as e:
            logger.error(f"Background analysis failed: {e}")
        finally:
            analysis_db.close()

    try:
        thread = threading.Thread(target=run_full_background_processing, daemon=True)
        thread.start()
    except Exception as e:
        logger.error(f"Failed to start AI thread: {e}")

    return basic_result

# Helper to validate password strength
def validate_password_strength(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Пароль должен содержать минимум 8 символов")
    if not re.search("[a-z]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы одну строчную букву")
    if not re.search("[0-9]", password):
        raise HTTPException(status_code=400, detail="Пароль должен содержать хотя бы одну цифру")
    return True

@app.post("/logic/analyze-frame/")
def analyze_frame_api(candidate_id: int, file: UploadFile = File(...)):
    from utils.face_analyzer import analyze_frame

    image_bytes = file.file.read()
    result = analyze_frame(image_bytes)

    # Save visual record to DB if face detected
    if result.get("face_detected"):
        try:
            with SessionLocal() as frame_db:
                record = database.VisualRecord(
                    candidate_id=candidate_id,
                    emotion=result.get("primary_emotion"),
                    stress_level=result.get("stress_level"),
                    notes=f"Взгляд: {result.get('gaze_direction', '—')}",
                )
                frame_db.add(record)
                frame_db.commit()
        except Exception as e:
            logger.warning(f"Visual record save failed: {e}")

    return result

@app.websocket("/ws/live-analysis/")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
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
    try:
        while True:
            # Keep-alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/webrtc/{candidate_id}")
async def webrtc_signaling(websocket: WebSocket, candidate_id: int, token: Optional[str] = None):
    logger.info(f"[WebRTC] New connection request for candidate_id={candidate_id}, token_present={bool(token)}")
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
        
        logger.info(f"[WebRTC] Connection authorized: actor_type={actor_type}, candidate_id={candidate_id}")

    except JWTError as e:
        logger.error(f"[WebRTC] JWT Decode Error: {e}")
        await websocket.close(code=4001)
        return
    except Exception as e:
        logger.error(f"[WebRTC] Auth Error: {e}")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    room = _get_room(int(candidate_id))
    
    if actor_type == "admin":
        if room.admin:
            logger.info(f"[WebRTC] Room {candidate_id}: Replacing existing admin connection")
            try:
                await room.admin.close()
            except Exception:
                pass
        room.admin = websocket
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
def login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(database.User).filter(database.User.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    # Track login activity
    user.login_count = (user.login_count or 0) + 1
    user.last_login = datetime.datetime.utcnow()
    db.commit()

    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user.email,
        "name": user.name,
        "role": user.role
    }

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
