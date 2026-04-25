"""RAG service — Qdrant vector search for interview context retrieval.

Uses Mistral embeddings (mistral-embed) to avoid OpenAI dependency.
Falls back gracefully if Qdrant is unavailable.
"""
import os
import logging
import hashlib
import threading
from typing import List, Optional

from .rag_metrics import metrics, now_ms
from .rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

# Lazy imports — don't crash if not installed
_qdrant_client = None
_qdrant_lock = threading.Lock()              # _get_qdrant() singleton uchun
_collection_ready: bool = False              # ensure_collection() idempotent flag
_collection_lock = threading.Lock()
_embed_cache: dict = {}
_embed_cache_lock = threading.Lock()         # cache iteratsiya/eviction race oldini olish
_EMBED_CACHE_MAX = 2000

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "psychology")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
EMBED_MODEL = "mistral-embed"  # Mistral's embedding model
EMBED_DIM = 1024  # mistral-embed output dimension

# Configurable retrieval thresholds (cosine similarity, [-1, 1])
_CONTEXT_SCORE_THRESHOLD = float(os.getenv("RAG_CONTEXT_SCORE_THRESHOLD", "0.3"))

# Mistral embed API rate limit. Free tier allows ~1 rps; paid tier up to ~6 rps.
_EMBED_RPS = float(os.getenv("MISTRAL_EMBED_RPS", "5"))
_EMBED_BURST = int(os.getenv("MISTRAL_EMBED_BURST", "10"))
_EMBED_WAIT_TIMEOUT = float(os.getenv("MISTRAL_EMBED_WAIT_TIMEOUT", "30"))
_embed_bucket = TokenBucket(rate=_EMBED_RPS, burst=_EMBED_BURST, name="mistral_embed")


def embed_bucket_stats() -> dict:
    return _embed_bucket.stats()


def _get_qdrant():
    """Lazy init Qdrant client — thread-safe (double-checked locking).
    Aks holda 2 ta thread bir vaqtda klient yaratib, resurs leak bo'ladi."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if not QDRANT_URL:
        return None
    with _qdrant_lock:
        # Lock olishidan oldin boshqa thread allaqachon yaratgan bo'lishi mumkin
        if _qdrant_client is not None:
            return _qdrant_client
        api_key = QDRANT_API_KEY if QDRANT_API_KEY and QDRANT_API_KEY != "your_qdrant_key_here" else None
        try:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=10)
            logger.info(f"Qdrant connected: {QDRANT_URL}")
            return _qdrant_client
        except Exception as e:
            logger.warning(f"Qdrant init failed: {e}")
            return None


def _embed_text(text: str) -> Optional[List[float]]:
    """Get embedding vector using Mistral embed API."""
    if not MISTRAL_API_KEY:
        return None

    # Cache by text hash — thread-safe read
    key = hashlib.md5(text.encode()).hexdigest()
    with _embed_cache_lock:
        cached = _embed_cache.get(key)
    if cached is not None:
        metrics.record_embed(elapsed_ms=0.0, cache_hit=True)
        return cached

    if not _embed_bucket.acquire(timeout=_EMBED_WAIT_TIMEOUT):
        logger.warning("Mistral embed rate limit exceeded; skipping embedding")
        metrics.record_embed(elapsed_ms=0.0, error=True)
        return None

    t0 = now_ms()
    try:
        import requests
        resp = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": [text]},
            timeout=15,
        )
        resp.raise_for_status()
        vector = resp.json()["data"][0]["embedding"]
        # Eviction + insert — lock ostida (avvalda RuntimeError xavfi bor edi)
        with _embed_cache_lock:
            if len(_embed_cache) >= _EMBED_CACHE_MAX:
                # Evict oldest half (FIFO order — Python 3.7+ dict ordered)
                keys = list(_embed_cache.keys())
                for k in keys[: len(keys) // 2]:
                    _embed_cache.pop(k, None)
            _embed_cache[key] = vector
        metrics.record_embed(elapsed_ms=now_ms() - t0)
        return vector
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        metrics.record_embed(elapsed_ms=now_ms() - t0, error=True)
        return None


def ensure_collection():
    """Create Qdrant collection if it doesn't exist — thread-safe + idempotent.
    Lock orqali bir vaqtda 2 ta admin approve qilsa, ikkalasi ham collection
    yaratishga urinmaydi (avvalda silent fail edi → indekslash buzilardi)."""
    global _collection_ready
    if _collection_ready:
        return True
    client = _get_qdrant()
    if not client:
        return False
    with _collection_lock:
        if _collection_ready:
            return True
        try:
            from qdrant_client.models import Distance, VectorParams
            collections = [c.name for c in client.get_collections().collections]
            if QDRANT_COLLECTION not in collections:
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")
            _collection_ready = True
            return True
        except Exception as e:
            # "already exists" xatoligi konkurent yaratish bo'lsa — bu OK
            err_str = str(e).lower()
            if "already exists" in err_str or "conflict" in err_str:
                _collection_ready = True
                return True
            logger.warning(f"Qdrant collection check failed: {e}")
            return False


def add_document(text: str, metadata: dict = None, doc_id: str = None) -> bool:
    """Add a document to Qdrant vector store."""
    client = _get_qdrant()
    if not client:
        return False

    vector = _embed_text(text)
    if not vector:
        return False

    try:
        from qdrant_client.models import PointStruct
        point_id = doc_id or hashlib.md5(text.encode()).hexdigest()
        # Use integer hash for Qdrant point ID
        int_id = int(hashlib.md5(point_id.encode()).hexdigest()[:15], 16)
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=int_id,
                    vector=vector,
                    payload={"text": text, **(metadata or {})},
                )
            ],
        )
        return True
    except Exception as e:
        logger.warning(f"Qdrant upsert failed: {e}")
        return False


def search_context(query: str, top_k: int = 3) -> str:
    """Search Qdrant for relevant context. Returns concatenated text."""
    client = _get_qdrant()
    if not client:
        return ""

    vector = _embed_text(query)
    if not vector:
        return ""

    t0 = now_ms()
    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            score_threshold=_CONTEXT_SCORE_THRESHOLD,
        )
        elapsed = now_ms() - t0
        if not results:
            metrics.record_search(elapsed_ms=elapsed, hits=0)
            return ""

        texts = []
        for r in results:
            text = r.payload.get("text", "")
            if text:
                texts.append(text)

        top = float(results[0].score) if results else 0.0
        metrics.record_search(elapsed_ms=elapsed, hits=len(texts), top_score=top)
        context = "\n---\n".join(texts)
        logger.info(
            "RAG ctx: %d docs in %.0fms (scores=%s, threshold=%.2f)",
            len(texts),
            elapsed,
            [round(r.score, 2) for r in results],
            _CONTEXT_SCORE_THRESHOLD,
        )
        return context
    except Exception as e:
        metrics.record_search(elapsed_ms=now_ms() - t0, hits=0, error=True)
        logger.warning(f"Qdrant search failed: {e}")
        return ""


def get_all_documents(limit: int = 100) -> list:
    """Get all documents from collection."""
    client = _get_qdrant()
    if not client:
        return []
    try:
        result = client.scroll(collection_name=QDRANT_COLLECTION, limit=limit)
        return [
            {"id": str(p.id), "text": p.payload.get("text", ""), "metadata": {k: v for k, v in p.payload.items() if k != "text"}}
            for p in result[0]
        ]
    except Exception as e:
        logger.warning(f"Qdrant scroll failed: {e}")
        return []


def delete_document(doc_id: str) -> bool:
    """Delete document by ID."""
    client = _get_qdrant()
    if not client:
        return False
    try:
        from qdrant_client.models import PointIdsList
        int_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)
        client.delete(collection_name=QDRANT_COLLECTION, points_selector=PointIdsList(points=[int_id]))
        return True
    except Exception as e:
        logger.warning(f"Qdrant delete failed: {e}")
        return False


def get_collection_info() -> dict:
    """Get collection stats."""
    client = _get_qdrant()
    if not client:
        return {"status": "disconnected"}
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "status": "connected",
            "collection": QDRANT_COLLECTION,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
