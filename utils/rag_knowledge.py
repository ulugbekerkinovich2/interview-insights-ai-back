"""Role-aware RAG knowledge base.

Wraps the low-level Qdrant+Mistral primitives from ``rag_service`` and adds:
  * document-level chunking
  * ``approved``/``doc_id`` payload for role-based filtering
  * chat flow with psychological safety guardrails (no diagnoses / no meds)
  * per-role response shaping (super_admin sees sources + confidence + chunks)

Qdrant is optional at runtime — helpers degrade gracefully if the server is
unreachable so the service can boot before Qdrant is deployed.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .rag_service import (
    EMBED_DIM,
    QDRANT_COLLECTION,
    _embed_text,
    _get_qdrant,
    ensure_collection,
)

logger = logging.getLogger(__name__)

# --- Roles -------------------------------------------------------------------

ROLE_SUPER_ADMIN = "SuperAdmin"
ROLE_PSYCHOLOGIST = "Psychologist"
ROLE_USER = "User"


def is_super_admin(role: Optional[str]) -> bool:
    return (role or "").strip().lower() == ROLE_SUPER_ADMIN.lower()


def can_add_knowledge(role: Optional[str]) -> bool:
    r = (role or "").strip().lower()
    return r in {ROLE_SUPER_ADMIN.lower(), ROLE_PSYCHOLOGIST.lower()}


def can_approve(role: Optional[str]) -> bool:
    return is_super_admin(role)


# --- Chunking ----------------------------------------------------------------

_CHUNK_MAX_CHARS = int(os.getenv("RAG_CHUNK_MAX_CHARS", "1800"))   # ~450 tokens
_CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))  # ~60 tokens
_MIN_CHUNK_CHARS = 120

_SPLIT_RE = re.compile(r"(?:\r?\n){2,}|(?<=[\.!\?])\s+(?=[A-ZА-ЯЎҚҒҲ])", re.UNICODE)


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?:\r?\n){3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = _CHUNK_MAX_CHARS, overlap: int = _CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text on paragraph/sentence boundaries with char-based overlap."""
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    if not pieces:
        pieces = [text]

    chunks: List[str] = []
    buf = ""
    for piece in pieces:
        if not buf:
            buf = piece
            continue
        if len(buf) + 1 + len(piece) <= max_chars:
            buf = f"{buf} {piece}"
            continue
        chunks.append(buf)
        if overlap > 0 and len(buf) > overlap:
            buf = f"{buf[-overlap:]} {piece}"
        else:
            buf = piece
    if buf:
        chunks.append(buf)

    # Hard-split any chunk that is still too long (e.g. a single massive paragraph).
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue
        for i in range(0, len(c), max_chars - overlap):
            sub = c[i : i + max_chars]
            if len(sub) >= _MIN_CHUNK_CHARS or not final:
                final.append(sub)
    return final


# --- Qdrant point helpers ----------------------------------------------------

def _point_id(doc_id: int, chunk_index: int) -> int:
    # Stable per (doc_id, chunk_index) — allows idempotent re-indexing.
    return (int(doc_id) << 20) | (int(chunk_index) & 0xFFFFF)


def index_document(
    doc_id: int,
    title: str,
    content: str,
    *,
    approved: bool,
    category: Optional[str] = None,
    language: str = "uz",
    source_type: str = "text",
    source_name: Optional[str] = None,
    created_by: Optional[int] = None,
) -> Tuple[bool, int]:
    """Chunk, embed and upsert a document into Qdrant. Returns (ok, chunks_count)."""
    client = _get_qdrant()
    if client is None:
        logger.info("index_document: Qdrant unavailable — skipping upsert for doc_id=%s", doc_id)
        return False, 0

    if not ensure_collection():
        return False, 0

    chunks = chunk_text(content)
    if not chunks:
        return False, 0

    try:
        from qdrant_client.models import PointStruct
    except Exception as exc:  # pragma: no cover
        logger.warning("qdrant-client missing: %s", exc)
        return False, 0

    # Remove any stale points for this doc_id first (handles re-index).
    delete_document_points(doc_id)

    points = []
    for idx, chunk in enumerate(chunks):
        vector = _embed_text(chunk)
        if not vector:
            logger.warning("index_document: embedding failed for doc_id=%s chunk=%s", doc_id, idx)
            continue
        if len(vector) != EMBED_DIM:
            logger.warning("index_document: unexpected embedding dim %s (expected %s)", len(vector), EMBED_DIM)
        payload = {
            "doc_id": int(doc_id),
            "chunk_index": idx,
            "title": title,
            "text": chunk,
            "category": category,
            "language": language,
            "approved": bool(approved),
            "source_type": source_type,
            "source_name": source_name,
            "created_by": created_by,
        }
        points.append(PointStruct(id=_point_id(doc_id, idx), vector=vector, payload=payload))

    if not points:
        return False, 0

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        return True, len(points)
    except Exception as exc:
        logger.warning("Qdrant upsert failed for doc_id=%s: %s", doc_id, exc)
        return False, 0


def update_document_approval(doc_id: int, approved: bool) -> bool:
    """Update the ``approved`` flag on every chunk of a document without re-embedding."""
    client = _get_qdrant()
    if client is None:
        return False
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client.set_payload(
            collection_name=QDRANT_COLLECTION,
            payload={"approved": bool(approved)},
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=int(doc_id)))]),
        )
        return True
    except Exception as exc:
        logger.warning("Qdrant set_payload failed for doc_id=%s: %s", doc_id, exc)
        return False


def delete_document_points(doc_id: int) -> bool:
    client = _get_qdrant()
    if client is None:
        return False
    try:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=int(doc_id)))])
            ),
        )
        return True
    except Exception as exc:
        logger.warning("Qdrant delete by doc_id=%s failed: %s", doc_id, exc)
        return False


# --- Retrieval ---------------------------------------------------------------

def search_knowledge(
    query: str,
    *,
    top_k: int = 5,
    only_approved: bool = True,
    category: Optional[str] = None,
    language: Optional[str] = None,
    score_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Return list of {text, score, doc_id, title, chunk_index, approved, ...}."""
    client = _get_qdrant()
    if client is None:
        return []

    vector = _embed_text(query)
    if not vector:
        return []

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must: List[Any] = []
        if only_approved:
            must.append(FieldCondition(key="approved", match=MatchValue(value=True)))
        if category:
            must.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if language:
            must.append(FieldCondition(key="language", match=MatchValue(value=language)))
        qfilter = Filter(must=must) if must else None

        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.warning("search_knowledge failed: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for r in results or []:
        payload = dict(r.payload or {})
        out.append(
            {
                "score": float(r.score),
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "chunk_index": payload.get("chunk_index"),
                "approved": bool(payload.get("approved", False)),
                "category": payload.get("category"),
                "language": payload.get("language"),
            }
        )
    return out


# --- Prompting ---------------------------------------------------------------

SYSTEM_PROMPT_UZ = """Sen mijozga yordam beruvchi empatik psixologik yordamchisan.

QATTIQ QOIDALAR:
- Hech qachon aniq diagnoz qo'yma ("sizda depressiya bor" kabi gaplar taqiqlanadi).
- Hech qachon dori tavsiya qilma.
- Xavfli holat (o'z joniga qasd qilish, boshqalarga zarar, zo'ravonlik belgilari) sezilsa, darhol jonli mutaxassis yoki tezkor xizmatga murojaat qilishni tavsiya qil.
- Faqat quyida berilgan KONTEKSTga tayan. Agar kontekstda kerakli ma'lumot yo'q bo'lsa, aynan shu iborani javob qil: "Bu savol uchun yetarli ma'lumot topilmadi".
- Taxmin qilma, ma'lumotni o'zingdan to'qib qo'shma.

USLUB:
- Oddiy, insoniy, iliq til. Og'ir ilmiy jargon ishlatma.
- Qisqa va aniq javob ber.
- Javobni quyidagi tuzilma bo'yicha ber:
  1) Muammoni qisqa tushuntir.
  2) Mumkin bo'lgan sabablar.
  3) Amaliy maslahat / qadamlar.
"""


def _build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    if chunks:
        ctx_lines = []
        for i, c in enumerate(chunks, 1):
            title = c.get("title") or "Manba"
            ctx_lines.append(f"[{i}] ({title})\n{c.get('text', '').strip()}")
        context_block = "\n\n".join(ctx_lines)
    else:
        context_block = "(kontekst topilmadi)"

    return (
        "KONTEKST (tasdiqlangan bilimlar bazasidan olingan parchalar):\n"
        f"{context_block}\n\n"
        "FOYDALANUVCHI SAVOLI:\n"
        f"{query}\n\n"
        "Yuqoridagi qoidalarga qat'iy rioya qilgan holda javob ber."
    )


# --- LLM call ----------------------------------------------------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
FALLBACK_NO_CONTEXT = "Bu savol uchun yetarli ma'lumot topilmadi"


def ask_mistral(query: str, chunks: List[Dict[str, Any]], *, timeout: int = 45) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return FALLBACK_NO_CONTEXT

    if not chunks:
        # Do not call the LLM when there is zero retrieved context — fall back deterministically.
        return FALLBACK_NO_CONTEXT

    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.3,
                "max_tokens": 700,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_UZ},
                    {"role": "user", "content": _build_user_prompt(query, chunks)},
                ],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip() or FALLBACK_NO_CONTEXT
    except Exception as exc:
        logger.warning("Mistral call failed: %s", exc)
        return FALLBACK_NO_CONTEXT


# --- Chat orchestration ------------------------------------------------------

def _confidence_from_scores(scores: Iterable[float]) -> float:
    scores = list(scores)
    if not scores:
        return 0.0
    top = scores[0]
    # COSINE similarity is already in [-1,1], clamp to [0,1] and convert to percent.
    return round(max(0.0, min(1.0, top)) * 100, 1)


def _chat_via_langchain(query: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
    """Use LangChain (``ChatMistralAI``) if available, else return None."""
    try:
        from . import rag_langchain as lc
    except Exception:
        return None
    if not lc.is_available():
        return None
    if not chunks:
        return None
    user_prompt = _build_user_prompt(query, chunks)
    return lc.lc_chat(SYSTEM_PROMPT_UZ, user_prompt)


def _search_via_langchain(
    query: str, *, only_approved: bool, top_k: int, category: Optional[str], language: Optional[str]
) -> Optional[List[Dict[str, Any]]]:
    try:
        from . import rag_langchain as lc
    except Exception:
        return None
    if not lc.is_available():
        return None
    return lc.lc_search(
        query,
        only_approved=only_approved,
        top_k=top_k,
        category=category,
        language=language,
    )


def run_chat(
    query: str,
    *,
    role: str,
    top_k: int = 5,
    category: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Role-aware RAG chat. Returns a dict suitable for ``KnowledgeChatResponse``.

    Prefers the LangChain path when the package stack is available; falls back
    to direct Qdrant + Mistral HTTP calls otherwise.
    """
    admin = is_super_admin(role)

    chunks = _search_via_langchain(
        query, only_approved=not admin, top_k=top_k, category=category, language=language
    )
    used_backend = "langchain"
    if chunks is None:
        chunks = search_knowledge(
            query,
            top_k=top_k,
            only_approved=not admin,
            category=category,
            language=language,
        )
        used_backend = "direct"

    answer: Optional[str] = None
    if chunks:
        answer = _chat_via_langchain(query, chunks)
        if not answer:
            answer = ask_mistral(query, chunks)
    if not answer:
        answer = FALLBACK_NO_CONTEXT

    response: Dict[str, Any] = {
        "answer": answer,
        "role_seen": role or "anonymous",
    }

    if admin:
        response["confidence"] = _confidence_from_scores(c["score"] for c in chunks)
        response["used_chunks"] = [
            {
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
                "approved": bool(c.get("approved", False)),
            }
            for c in chunks
        ]
        response["sources"] = [
            {
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "score": c.get("score"),
                "approved": c.get("approved"),
                "backend": used_backend,
            }
            for c in chunks
        ]

    return response


# --- Chat-as-admin: intent parser -------------------------------------------

@dataclass
class Intent:
    action: str  # see the README below for the full set
    doc_id: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    query: Optional[str] = None

# Supported chat actions:
#   save              — psychologist/admin creates a draft
#   list_drafts       — admin: all pending drafts
#   my_drafts         — author's own drafts
#   approve           — admin: approve a draft and index it
#   reject            — admin: delete a draft
#   reindex           — admin: reindex one document
#   status            — admin: Qdrant status + backend info
#   stats             — admin: aggregate knowledge-base statistics
#   retrain           — admin: re-embed every approved document
#   search            — full-text search (SQL ILIKE on title/content)
#   get               — fetch one document's details
#   delete_all_drafts — admin: wipe all un-approved drafts


_RE_APPROVE = re.compile(
    r"^\s*(?:/approve|approve|tasdiqla)\s+#?(\d+)\s*$|^\s*#?(\d+)\s+(?:ni\s+)?(?:tasdiqla|approve)\s*$",
    re.IGNORECASE,
)
_RE_REJECT = re.compile(
    r"^\s*(?:/reject|reject|rad\s*et|o'?chir|delete)\s+#?(\d+)\s*$|^\s*#?(\d+)\s+(?:ni\s+)?(?:rad\s*et|o'?chir|reject|delete)\s*$",
    re.IGNORECASE,
)
_RE_REINDEX = re.compile(
    r"^\s*(?:/reindex|reindex|qayta\s+indeksla)\s+#?(\d+)\s*$|^\s*#?(\d+)\s+(?:ni\s+)?(?:reindex|qayta\s+indeksla)\s*$",
    re.IGNORECASE,
)
_RE_SAVE = re.compile(
    r"^\s*(?:/save|save|saqla|eslab\s*qol)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_RE_LIST_DRAFTS = re.compile(r"^\s*(?:/drafts|drafts|draftlar|pending|kutilayotgan)\s*$", re.IGNORECASE)
_RE_MY_DRAFTS = re.compile(r"^\s*(?:/mydrafts|my\s*drafts|mening\s*draftlarim|o'?z\s*draftlarim)\s*$", re.IGNORECASE)
_RE_STATUS = re.compile(r"^\s*(?:/status|status|qdrant|holat)\s*$", re.IGNORECASE)
_RE_STATS = re.compile(r"^\s*(?:/stats|stats|statistika|hisobot)\s*$", re.IGNORECASE)
_RE_RETRAIN = re.compile(
    r"^\s*(?:/retrain|retrain|reindex\s+all|qayta\s+train(?:\s+qil)?|hammani\s+qayta\s+indeksla)\s*$",
    re.IGNORECASE,
)
_RE_EDIT = re.compile(
    r"^\s*(?:/edit|edit|tahrirla)\s+#?(\d+)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_RE_SEARCH = re.compile(
    r"^\s*(?:/search|search|qidir|topib\s*ber)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_RE_GET = re.compile(
    r"^\s*(?:/get|get|show|ko'?rsat|ochib\s*ber)\s+#?(\d+)\s*$",
    re.IGNORECASE,
)
_RE_DELETE_ALL_DRAFTS = re.compile(
    r"^\s*(?:/deldrafts|delete\s+all\s+drafts|hamma\s+draftlarni\s+o'?chir|draftlarni\s+o'?chir)\s*$",
    re.IGNORECASE,
)


def parse_intent(text: str, *, role: str) -> Optional[Intent]:
    """Detect a command intent from free-form chat text.

    Returns ``None`` for regular RAG questions.
    """
    if not text:
        return None
    raw = text.strip()

    if is_super_admin(role):
        m = _RE_APPROVE.match(raw)
        if m:
            return Intent(action="approve", doc_id=int(m.group(1) or m.group(2)))
        m = _RE_REJECT.match(raw)
        if m:
            return Intent(action="reject", doc_id=int(m.group(1) or m.group(2)))
        m = _RE_REINDEX.match(raw)
        if m:
            return Intent(action="reindex", doc_id=int(m.group(1) or m.group(2)))
        m = _RE_EDIT.match(raw)
        if m:
            body = m.group(2).strip()
            if "|" in body:
                title, _, content = body.partition("|")
                title, content = title.strip(), content.strip()
            else:
                first, _, _rest = body.partition("\n")
                title = first.strip().rstrip(".!?:")[:200] or None
                content = body
            return Intent(
                action="edit",
                doc_id=int(m.group(1)),
                title=title or None,
                content=content or None,
            )
        if _RE_LIST_DRAFTS.match(raw):
            return Intent(action="list_drafts")
        if _RE_STATUS.match(raw):
            return Intent(action="status")
        if _RE_STATS.match(raw):
            return Intent(action="stats")
        if _RE_RETRAIN.match(raw):
            return Intent(action="retrain")
        if _RE_DELETE_ALL_DRAFTS.match(raw):
            return Intent(action="delete_all_drafts")

    # Intents available to any authenticated caller (query/get/search)
    m = _RE_GET.match(raw)
    if m:
        return Intent(action="get", doc_id=int(m.group(1)))
    m = _RE_SEARCH.match(raw)
    if m:
        return Intent(action="search", query=m.group(1).strip())

    if can_add_knowledge(role):
        if _RE_MY_DRAFTS.match(raw):
            return Intent(action="my_drafts")
        m = _RE_SAVE.match(raw)
        if m:
            body = m.group(1).strip()
            # Optional "Title | content" split; otherwise first line is title.
            if "|" in body:
                title, _, content = body.partition("|")
                title, content = title.strip(), content.strip()
            else:
                first, _, rest = body.partition("\n")
                title = first.strip().rstrip(".!?:")[:200] or body[:80]
                content = body
            if not title or not content:
                return None
            return Intent(action="save", title=title, content=content)

    return None


# --- Document parsing (uploads) ----------------------------------------------

def extract_text_from_upload(filename: str, data: bytes) -> str:
    """Best-effort text extraction from txt/pdf/docx uploads."""
    name = (filename or "").lower()
    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
            from io import BytesIO

            reader = PdfReader(BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            logger.warning("PDF parse failed: %s", exc)
            return ""

    if name.endswith(".docx"):
        try:
            import docx  # type: ignore
            from io import BytesIO

            doc = docx.Document(BytesIO(data))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text)
        except Exception as exc:
            logger.warning("DOCX parse failed: %s", exc)
            return ""

    return ""
