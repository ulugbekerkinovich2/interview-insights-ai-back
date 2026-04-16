"""RAG service — Qdrant vector search for interview context retrieval.

Uses Mistral embeddings (mistral-embed) to avoid OpenAI dependency.
Falls back gracefully if Qdrant is unavailable.
"""
import os
import logging
import hashlib
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — don't crash if not installed
_qdrant_client = None
_embed_cache: dict = {}
_EMBED_CACHE_MAX = 2000

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "psychology")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
EMBED_MODEL = "mistral-embed"  # Mistral's embedding model
EMBED_DIM = 1024  # mistral-embed output dimension


def _get_qdrant():
    """Lazy init Qdrant client."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if not QDRANT_URL or not QDRANT_API_KEY or QDRANT_API_KEY == "your_qdrant_key_here":
        return None
    try:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        logger.info(f"Qdrant connected: {QDRANT_URL}")
        return _qdrant_client
    except Exception as e:
        logger.warning(f"Qdrant init failed: {e}")
        return None


def _embed_text(text: str) -> Optional[List[float]]:
    """Get embedding vector using Mistral embed API."""
    if not MISTRAL_API_KEY:
        return None

    # Cache by text hash
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]

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
        if len(_embed_cache) >= _EMBED_CACHE_MAX:
            # Evict oldest half
            keys = list(_embed_cache.keys())
            for k in keys[:len(keys) // 2]:
                del _embed_cache[k]
        _embed_cache[key] = vector
        return vector
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    client = _get_qdrant()
    if not client:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")
        return True
    except Exception as e:
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

    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            score_threshold=0.3,
        )
        if not results:
            return ""

        texts = []
        for r in results:
            text = r.payload.get("text", "")
            if text:
                texts.append(text)

        context = "\n---\n".join(texts)
        logger.info(f"RAG: found {len(texts)} relevant docs (scores: {[round(r.score, 2) for r in results]})")
        return context
    except Exception as e:
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
