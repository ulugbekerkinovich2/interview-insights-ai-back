import os
import secrets
import string
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import shutil
import tempfile
import datetime
import requests
import uuid
import re
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import bleach

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

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media") # Unsecured mount removed

@app.get("/media/frames/{filename}")
def get_protected_frame(filename: str, user: database.User = Depends(get_current_user)):
    # Only allow authenticated recruiters/admins to see frames
    if not user:
        raise HTTPException(status_code=401, detail="Rasmga kirish uchun tizimga kiring")
    file_path = MEDIA_DIR / "frames" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Rasm topilmadi")
    from fastapi.responses import FileResponse
    return FileResponse(file_path)


@app.on_event("startup")
def startup():
    if database.DATABASE_URL.startswith("sqlite"):
        database.init_db()

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
            raise HTTPException(status_code=401, detail="Seans muddati tugagan yoki tizimga kirmaganman")
        if user.role not in roles and user.role != "SuperAdmin":
            raise HTTPException(status_code=403, detail="Sizda ushbu amalni bajarish uchun ruxsat yo'q")
        return user
    return role_checker


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
def get_candidate_stats(db: Session = Depends(get_db)):
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
def read_candidates(db: Session = Depends(get_db)):
    candidates = db.query(Candidate).all()
    return candidates

@app.get("/candidates/{candidate_id}/visual", response_model=List[schemas.VisualRecordSchema])
def read_visual_records(candidate_id: int, db: Session = Depends(get_db)):
    get_candidate_or_404(db, candidate_id)
    records = (
        db.query(database.VisualRecord)
        .filter(database.VisualRecord.candidate_id == candidate_id)
        .order_by(database.VisualRecord.timestamp.asc())
        .all()
    )
    return [
        {
            "emotion": r.emotion,
            "stress_level": r.stress_level,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]

@app.get("/candidates/{candidate_id}", response_model=schemas.CandidateSchema)
def read_candidate(candidate_id: int, db: Session = Depends(get_db)):
    return get_candidate_or_404(db, candidate_id)

@app.post("/candidates/", response_model=schemas.CandidateSchema)
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
    res = schemas.CandidateSchema.from_orm(db_candidate)
    # We'll hijack the response for this specific call to include the plain pin
    # (In production, usually handled by a separate schema or notification)
    return res

@app.post("/candidates/login")
@limiter.limit("5/minute")
async def candidate_login(request: Request, access_code: str = Form(...), pin: str = Form(...), db: Session = Depends(get_db)):
    # 1. Find by long access token
    candidate = db.query(Candidate).filter(Candidate.access_code == access_code).first()
    if not candidate:
        raise HTTPException(status_code=401, detail="Noto'g'ri havola")
    
    # 2. Verify hashed PIN
    if not candidate.pin_hash or not verify_password(pin, candidate.pin_hash):
        raise HTTPException(status_code=401, detail="PIN kod noto'g'ri")
    
    # Notify HR
    send_telegram_notification(f"🚀 <b>Nomzod sissiyaga kirdi!</b>\n\n👤 Nomzod: {candidate.name}\n🆔 ID: {candidate.id}\n📍 Holat: Suhbat boshlandi")
    
    # Notify admin about candidate joining
    await manager.broadcast({
        "type": "NOTIFICATION",
        "message": f"📢 Nomzod ulandi: {candidate.name}",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    })

    return {"status": "success", "candidate_id": candidate.id, "name": candidate.name}

@app.put("/candidates/{candidate_id}", response_model=schemas.CandidateSchema)
def update_candidate(candidate_id: int, candidate: schemas.CandidateCreate, db: Session = Depends(get_db)):
    db_candidate = get_candidate_or_404(db, candidate_id)
    
    db_candidate.name = candidate.name
    db_candidate.summary = candidate.summary
    db_candidate.status = candidate.status
    db_candidate.answers = candidate.answers
    
    db.commit()
    db.refresh(db_candidate)
    return db_candidate

# --- Chat Endpoints ---

@app.get("/chat/", response_model=List[schemas.ChatMessageSchema])
def get_chat_history(db: Session = Depends(get_db)):
    return db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()

@app.post("/chat/", response_model=schemas.ChatMessageSchema)
def add_chat_message(msg: schemas.ChatMessageCreate, db: Session = Depends(get_db)):
    import datetime
    db_msg = ChatMessage(
        role=msg.role,
        content=msg.content,
        timestamp=datetime.datetime.now().isoformat()
    )
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)
    return db_msg


@app.delete("/chat/")
def clear_chat_history(db: Session = Depends(get_db)):
    db.query(ChatMessage).delete()
    db.commit()
    return {"status": "cleared"}

# --- Settings Endpoints ---

@app.get("/settings/{key}", response_model=schemas.GlobalSettingSchema)
def get_setting(key: str, db: Session = Depends(get_db)):
    setting = db.query(GlobalSetting).filter(GlobalSetting.key == key).first()
    if setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return setting

@app.post("/settings/", response_model=schemas.GlobalSettingSchema)
def set_setting(setting: schemas.GlobalSettingBase, db: Session = Depends(get_db)):
    db_setting = db.query(GlobalSetting).filter(GlobalSetting.key == setting.key).first()
    if db_setting:
        db_setting.value = setting.value
    else:
        db_setting = GlobalSetting(key=setting.key, value=setting.value)
        db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

# --- Logic Endpoints (AI) ---

@app.post("/logic/transcribe/")
async def transcribe_audio_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        text = logic.transcribe_audio(tmp_path)
        return {"text": text}
    except logic.TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/logic/analyze/")
def analyze_answer_api(question: str, answer: str):
    try:
        analysis = logic.analyze_answer(question, answer)
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"analysis": analysis}

@app.post("/logic/summary/")
def generate_summary_api(candidate_id: int, db: Session = Depends(get_db)):
    candidate = get_candidate_or_404(db, candidate_id)
    
    try:
        summary = logic.build_interview_summary(candidate.answers)
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    candidate.summary = summary
    db.commit()
    return {"summary": summary}


@app.post("/logic/process-turn/")
async def process_turn_api(
    candidate_id: int = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    candidate = get_candidate_or_404(db, candidate_id)

    # Create permanent audio storage path
    audio_filename = f"{secrets.token_hex(16)}.wav"
    save_path = MEDIA_AUDIO_DIR / audio_filename
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = logic.process_interview_turn(str(save_path), question, db=db)
        # Add audio URL to the result
        result["audio_url"] = f"/media/audio/{audio_filename}"

        answers = list(candidate.answers or [])
        answers.append(result.copy())
        candidate.answers = answers
        db.commit()
        db.refresh(candidate)

        # Notify HR with summary of the move
        safe_answer = result.get('answer', '')[:100] + "..."
        next_q = result.get('next_suggestion', 'Aniqlanmadi')
        msg = (
            f"📝 <b>Yangi javob tahlil qilindi</b>\n\n"
            f"❓ Savol: {question}\n"
            f"💬 Javob: {safe_answer}\n"
            f"🧠 AI Insight: {result.get('ai', '')[:150]}...\n\n"
            f"✨ <b>Tavsiya etilgan keyingi savol:</b>\n<i>{next_q}</i>"
        )
        send_telegram_notification(msg)

        return result
    except logic.TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except logic.AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# Helper to validate password strength
def validate_password_strength(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Parol kamida 8 ta belgidan iborat bo'lishi kerak")
    if not re.search("[a-z]", password):
        raise HTTPException(status_code=400, detail="Parolda kamida bitta kichik harf bo'lishi kerak")
    if not re.search("[0-9]", password):
        raise HTTPException(status_code=400, detail="Parolda kamida bitta raqam bo'lishi kerak")
    return True

@app.post("/logic/analyze-frame/")
async def analyze_frame_api(candidate_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    get_candidate_or_404(db, candidate_id)
    
    # Save frame to disk with UUID to prevent guessing
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    frame_filename = f"{uuid.uuid4()}{ext}"
    frame_save_path = BACKEND_DIR / "media" / "frames" / frame_filename
    frame_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(frame_save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    tmp_path = frame_save_path
    
    try:
        res = logic.analyze_visual_frame(str(tmp_path))
        
        # Bazaga saqlash
        record = database.VisualRecord(
            candidate_id=candidate_id,
            emotion=res.get("primary_emotion"),
            stress_level=res.get("stress_level"),
            notes=res.get("behavior_notes"),
            image_url=f"/media/frames/{frame_filename}"
        )
        db.add(record)
        db.commit()
        
        # Get base64 for real-time video preview on admin side
        import base64
        with open(tmp_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Adminkaga real-time yuborish (Broadcast)
        live_data = {
            "type": "LIVE_VISUAL",
            "candidate_id": candidate_id,
            "analysis": res,
            "image": f"data:image/jpeg;base64,{encoded_string}",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        await manager.broadcast(live_data)
        
        # Notify on high stress
        if res.get("stress_level") == "High":
            await manager.broadcast({
                "type": "NOTIFICATION",
                "message": f"⚠️ Diqqat! Nomzodda (ID: {candidate_id}) yuqori hayajon aniqlandi.",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })

        return res
    finally:
        # We keep the file on disk now
        pass

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

@app.post("/auth/login")
@limiter.limit("5/minute")
def login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(database.User).filter(database.User.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Email yoki parol noto'g'ri")
    
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
    validate_password_strength(password)
    existing = db.query(database.User).filter(database.User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Bu email allaqachon ro'yxatdan o'tgan")

    db_user = database.User(
        name=name, 
        email=email, 
        password=get_password_hash(password), 
        role="Recruiter"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- User Management ---
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

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), admin: database.User = Depends(require_role(["SuperAdmin"]))):
    user = db.query(database.User).filter(database.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"status": "deleted"}

# --- Feature Flags Management ---
@app.get("/features/", response_model=List[dict])
def get_features(db: Session = Depends(get_db)):
    # Any logged in user can view flags to adjust frontend state
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
