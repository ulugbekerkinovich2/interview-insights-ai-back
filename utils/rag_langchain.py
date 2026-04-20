"""LangChain-based RAG pipeline (optional).

Uses ``langchain-qdrant`` + ``langchain-mistralai`` when the packages and the
required env vars (``QDRANT_URL``, ``QDRANT_API_KEY``, ``MISTRAL_API_KEY``) are
present. Degrades gracefully: if any import or env lookup fails, ``is_available()``
returns ``False`` and callers should fall back to the direct HTTP path in
``rag_knowledge``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_IMPORT_OK = True
try:  # pragma: no cover — import guard
    from langchain_qdrant import QdrantVectorStore  # type: ignore
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # type: ignore
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore
except Exception as exc:  # pragma: no cover
    logger.info("LangChain stack not available: %s", exc)
    _IMPORT_OK = False


_vector_store = None
_llm = None
_embeddings = None


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _get_embeddings():
    global _embeddings
    if not _IMPORT_OK:
        return None
    if _embeddings is not None:
        return _embeddings
    api_key = _env("MISTRAL_API_KEY")
    if not api_key:
        return None
    try:
        _embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
        return _embeddings
    except Exception as exc:
        logger.warning("MistralAIEmbeddings init failed: %s", exc)
        return None


def _get_llm():
    global _llm
    if not _IMPORT_OK:
        return None
    if _llm is not None:
        return _llm
    api_key = _env("MISTRAL_API_KEY")
    if not api_key:
        return None
    try:
        _llm = ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            api_key=api_key,
            temperature=0.3,
            max_tokens=700,
        )
        return _llm
    except Exception as exc:
        logger.warning("ChatMistralAI init failed: %s", exc)
        return None


def _get_vector_store():
    global _vector_store
    if not _IMPORT_OK:
        return None
    if _vector_store is not None:
        return _vector_store

    url = _env("QDRANT_URL")
    api_key = _env("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION_NAME", "psychology")
    if not url or not api_key or api_key == "your_qdrant_key_here":
        return None

    embeddings = _get_embeddings()
    if embeddings is None:
        return None

    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )
        return _vector_store
    except Exception as exc:
        logger.warning("QdrantVectorStore init failed: %s", exc)
        return None


def is_available() -> bool:
    """Return True only when LangChain can actually answer a query end-to-end."""
    if not _IMPORT_OK:
        return False
    return _get_vector_store() is not None and _get_llm() is not None


# --- Retrieval ---------------------------------------------------------------

def lc_search(
    query: str,
    *,
    only_approved: bool,
    top_k: int = 5,
    score_threshold: float = 0.35,
    category: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    vs = _get_vector_store()
    if vs is None:
        return []

    must: List[Any] = []
    if only_approved:
        must.append(FieldCondition(key="approved", match=MatchValue(value=True)))
    if category:
        must.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if language:
        must.append(FieldCondition(key="language", match=MatchValue(value=language)))
    qfilter = Filter(must=must) if must else None

    try:
        results = vs.similarity_search_with_score(query=query, k=top_k, filter=qfilter)
    except Exception as exc:
        logger.warning("LangChain similarity search failed: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for doc, score in results:
        if score is not None and float(score) < score_threshold:
            continue
        md = dict(getattr(doc, "metadata", {}) or {})
        text = getattr(doc, "page_content", "") or md.get("text", "")
        out.append(
            {
                "score": float(score) if score is not None else 0.0,
                "text": text,
                "doc_id": md.get("doc_id"),
                "title": md.get("title"),
                "chunk_index": md.get("chunk_index"),
                "approved": bool(md.get("approved", False)),
                "category": md.get("category"),
                "language": md.get("language"),
            }
        )
    return out


# --- Chat --------------------------------------------------------------------

def lc_chat(system_prompt: str, user_prompt: str) -> Optional[str]:
    llm = _get_llm()
    if llm is None:
        return None
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(resp, "content", None)
        if isinstance(content, list):
            # Some providers return a list of parts.
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return (content or "").strip() or None
    except Exception as exc:
        logger.warning("LangChain chat invoke failed: %s", exc)
        return None
