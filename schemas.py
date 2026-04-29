from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

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
    filters: List[str] = Field(default_factory=list)

class CandidateCreate(CandidateBase):
    pass

class CandidateSchema(CandidateBase):
    id: int
    display_id: Optional[str] = None  # YYMMNNNN format (e.g. 26040001)

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
    is_active: bool = True
    login_count: Optional[int] = 0
    last_login: Optional[Any] = None
    created_at: Optional[Any] = None

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
    image_url: Optional[str] = None
    timestamp: str


class HealthComponentSchema(BaseModel):
    available: bool
    dialect: Optional[str] = None
    detail: Optional[str] = None


class HealthSchema(BaseModel):
    status: str
    service: str
    database: HealthComponentSchema


# --- Knowledge base (RAG) ---

class KnowledgeDocCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    category: Optional[str] = None
    language: Optional[str] = "uz"


class KnowledgeDocSchema(BaseModel):
    id: int
    title: str
    content: str
    source_type: str
    source_name: Optional[str] = None
    category: Optional[str] = None
    language: str
    approved: bool
    created_by: Optional[int] = None
    approved_by: Optional[int] = None
    created_at: Optional[Any] = None
    approved_at: Optional[Any] = None
    chunks_count: int = 0
    qdrant_indexed: bool = False

    class Config:
        from_attributes = True


class KnowledgeDocUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    content: Optional[str] = Field(default=None, min_length=1)
    category: Optional[str] = None
    language: Optional[str] = None


class KnowledgeBulkDelete(BaseModel):
    ids: List[int] = Field(default_factory=list, min_length=1)


class KnowledgeStats(BaseModel):
    total: int
    approved: int
    drafts: int
    indexed_in_qdrant: int
    chunks_total: int
    by_category: Dict[str, int]
    by_language: Dict[str, int]
    qdrant: Dict[str, Any]


class KnowledgeRetrainReport(BaseModel):
    attempted: int
    succeeded: int
    failed: int
    chunks_total: int
    failed_ids: List[int] = Field(default_factory=list)


class RetrainJobSchema(BaseModel):
    id: int
    status: str  # pending | running | completed | failed
    triggered_by: Optional[int] = None
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    total_docs: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks_total: int = 0
    current_doc_id: Optional[int] = None
    failed_ids: List[int] = Field(default_factory=list)
    error: Optional[str] = None
    progress_pct: float = 0.0

    class Config:
        from_attributes = True


class ChatHistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class KnowledgeChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    # Oldingi xabarlar (kontekst). Backend faqat oxirgi 10 tasini ishlatadi.
    history: Optional[List[ChatHistoryMessage]] = Field(default=None, max_length=40)


class KnowledgeUsedChunk(BaseModel):
    doc_id: Optional[int] = None
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    text: str
    score: float
    approved: bool


class KnowledgeChatResponse(BaseModel):
    answer: str
    role_seen: str
    # "rag" (default) for RAG answers; otherwise the executed command name
    # (save / list_drafts / approve / reject / reindex / status / my_drafts).
    action: Optional[str] = "rag"
    # Optional structured payload for admin/chat-driven actions (e.g. a
    # list of draft summaries, or the affected ``doc_id``).
    data: Optional[Any] = None
    # Source fragments referenced by the answer (visible to every user).
    # Each entry: {index, doc_id, title, [cited, score, backend, approved] — admin only}.
    sources: Optional[List[Dict[str, Any]]] = None
    # 1-based indices of [N] markers that the LLM used in its answer.
    cited_indices: Optional[List[int]] = None
    # Populated only for SuperAdmin (testing mode)
    confidence: Optional[float] = None
    used_chunks: Optional[List[KnowledgeUsedChunk]] = None
