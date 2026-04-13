from pydantic import BaseModel, Field
from typing import List, Optional, Any

class AnswerSchema(BaseModel):
    id: int
    question: str
    answer: str
    ai: Optional[str] = None
    visual_raw: Optional[Any] = None

class CandidateBase(BaseModel):
    name: str
    summary: Optional[str] = ""
    status: Optional[str] = "interview_started"
    access_code: Optional[str] = None
    answers: List[dict] = Field(default_factory=list)

class CandidateCreate(CandidateBase):
    pass

class CandidateSchema(CandidateBase):
    id: int

    class Config:
        from_attributes = True

class CandidateCreateResponse(CandidateSchema):
    # Plain 6-digit PIN is returned only once at candidate creation time.
    pin: Optional[str] = None

class ChatMessageBase(BaseModel):
    role: str
    content: str

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageSchema(ChatMessageBase):
    id: int
    timestamp: Optional[str] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserSchema(BaseModel):
    id: int
    name: str
    email: str
    role: str

    class Config:
        from_attributes = True

class GlobalSettingBase(BaseModel):
    key: str
    value: Any

class GlobalSettingSchema(GlobalSettingBase):
    class Config:
        from_attributes = True


class VisualRecordSchema(BaseModel):
    emotion: Optional[str] = None
    stress_level: Optional[str] = None
    timestamp: str


class HealthComponentSchema(BaseModel):
    available: bool
    dialect: Optional[str] = None
    detail: Optional[str] = None


class HealthSchema(BaseModel):
    status: str
    service: str
    database: HealthComponentSchema
