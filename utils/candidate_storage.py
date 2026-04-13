import datetime

from sqlalchemy.orm import Session

from database import Candidate, ChatMessage, GlobalSetting, SessionLocal

def load_candidate(candidate_id) -> dict:
    if not candidate_id:
        return {}
    db = SessionLocal()
    try:
        candidate = db.query(Candidate).filter(Candidate.id == int(candidate_id)).first()
        if candidate:
            return {
                "id": candidate.id,
                "name": candidate.name,
                "summary": candidate.summary,
                "status": candidate.status,
                "answers": candidate.answers
            }
        return {}
    except (ValueError, TypeError):
        return {}
    finally:
        db.close()

def save_candidate(candidate_data: dict) -> int:
    db = SessionLocal()
    try:
        candidate_id = candidate_data.get("id")
        
        # Make id clean
        if isinstance(candidate_id, str):
            candidate_id = candidate_id.strip()
            if not candidate_id:
                candidate_id = None
            else:
                try:
                    candidate_id = int(candidate_id)
                except ValueError:
                    candidate_id = None

        existing = None
        if candidate_id:
            existing = db.query(Candidate).filter(Candidate.id == candidate_id).first()
        
        if existing:
            existing.name = candidate_data.get("name", existing.name)
            existing.summary = candidate_data.get("summary", existing.summary)
            existing.status = candidate_data.get("status", existing.status)
            existing.answers = candidate_data.get("answers", existing.answers)
            db.commit()
            return existing.id
        else:
            new_candidate = Candidate(
                name=candidate_data.get("name", "Unknown"),
                summary=candidate_data.get("summary", ""),
                status=candidate_data.get("status", "interview_started"),
                answers=candidate_data.get("answers", [])
            )
            db.add(new_candidate)
            db.commit()
            db.refresh(new_candidate)
            return new_candidate.id
    finally:
        db.close()

def list_all_candidates() -> list:
    db = SessionLocal()
    try:
        candidates = db.query(Candidate).all()
        result = []
        for c in candidates:
            result.append({
                "id": c.id,
                "name": c.name,
                "summary": c.summary,
                "status": c.status,
                "answers": c.answers
            })
        result.sort(key=lambda x: x.get("id", 0))
        return result
    finally:
        db.close()

def find_candidate_by_query(query: str) -> dict:
    q = str(query).strip().lower()
    if not q:
        return {}

    db = SessionLocal()
    try:
        # Search by ID if it's a number, or by Name
        filters = []
        if q.isdigit():
            filters.append(Candidate.id == int(q))
        filters.append(Candidate.name.ilike(f"%{q}%"))
        
        from sqlalchemy import or_
        candidate = db.query(Candidate).filter(or_(*filters)).first()
        
        if candidate:
            return {
                "id": candidate.id,
                "name": candidate.name,
                "summary": candidate.summary,
                "status": candidate.status,
                "answers": candidate.answers
            }
        return {}
    finally:
        db.close()

def save_chat_message(role: str, content: str):
    db = SessionLocal()
    try:
        msg = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.datetime.now().isoformat()
        )
        db.add(msg)
        db.commit()
    finally:
        db.close()

def load_chat_history() -> list:
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()
        return [{"role": m.role, "content": m.content} for m in messages]
    finally:
        db.close()

def clear_chat_history_db():
    db = SessionLocal()
    try:
        db.query(ChatMessage).delete()
        db.commit()
    finally:
        db.close()

def save_global_setting(key: str, value):
    db = SessionLocal()
    try:
        setting = db.query(GlobalSetting).filter(GlobalSetting.key == key).first()
        if setting:
            setting.value = value
        else:
            setting = GlobalSetting(key=key, value=value)
            db.add(setting)
        db.commit()
    finally:
        db.close()

def load_global_setting(key: str, default=None):
    db = SessionLocal()
    try:
        setting = db.query(GlobalSetting).filter(GlobalSetting.key == key).first()
        return setting.value if setting else default
    finally:
        db.close()
