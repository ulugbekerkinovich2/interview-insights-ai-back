"""In-memory RAG metrics — retrieval hit rate, latencies, top scores.

Cheap to maintain and exposed via /knowledge/metrics for SuperAdmin.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class _Counter:
    __slots__ = ("lock", "count", "total", "min_v", "max_v", "window")

    def __init__(self, window_size: int = 200):
        self.lock = threading.Lock()
        self.count = 0
        self.total = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")
        self.window: Deque[float] = deque(maxlen=window_size)

    def add(self, value: float) -> None:
        with self.lock:
            self.count += 1
            self.total += value
            self.min_v = min(self.min_v, value)
            self.max_v = max(self.max_v, value)
            self.window.append(value)

    def snapshot(self) -> Dict[str, float]:
        with self.lock:
            if not self.count:
                return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "recent_avg": 0.0}
            return {
                "count": self.count,
                "avg": round(self.total / self.count, 3),
                "min": round(self.min_v, 3),
                "max": round(self.max_v, 3),
                "recent_avg": round(sum(self.window) / len(self.window), 3) if self.window else 0.0,
            }


class _RAGMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.search_total = 0
        self.search_hits = 0
        self.search_empty = 0
        self.search_backend_errors = 0
        self.embed_total = 0
        self.embed_errors = 0
        self.embed_cache_hits = 0
        self.embed_latency_ms = _Counter()
        self.search_latency_ms = _Counter()
        self.top_score = _Counter()
        self.chunks_returned = _Counter()

    def record_search(self, *, elapsed_ms: float, hits: int, top_score: float = 0.0, error: bool = False) -> None:
        with self.lock:
            self.search_total += 1
            if error:
                self.search_backend_errors += 1
            elif hits == 0:
                self.search_empty += 1
            else:
                self.search_hits += 1
        self.search_latency_ms.add(elapsed_ms)
        self.chunks_returned.add(hits)
        if hits > 0 and top_score:
            self.top_score.add(top_score)

    def record_embed(self, *, elapsed_ms: float, error: bool = False, cache_hit: bool = False) -> None:
        with self.lock:
            if cache_hit:
                self.embed_cache_hits += 1
                return
            self.embed_total += 1
            if error:
                self.embed_errors += 1
        if not error:
            self.embed_latency_ms.add(elapsed_ms)

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            total = self.search_total
            hit_rate = round(self.search_hits / total * 100, 1) if total else 0.0
            empty_rate = round(self.search_empty / total * 100, 1) if total else 0.0
            embed_total = self.embed_total
            err_rate = round(self.embed_errors / embed_total * 100, 1) if embed_total else 0.0
            cache_total = self.embed_cache_hits + embed_total
            cache_rate = round(self.embed_cache_hits / cache_total * 100, 1) if cache_total else 0.0
            base = {
                "search": {
                    "total": total,
                    "hits": self.search_hits,
                    "empty": self.search_empty,
                    "errors": self.search_backend_errors,
                    "hit_rate_pct": hit_rate,
                    "empty_rate_pct": empty_rate,
                },
                "embed": {
                    "total": embed_total,
                    "errors": self.embed_errors,
                    "error_rate_pct": err_rate,
                    "cache_hits": self.embed_cache_hits,
                    "cache_hit_rate_pct": cache_rate,
                },
            }
        base["latency_ms"] = {
            "embed": self.embed_latency_ms.snapshot(),
            "search": self.search_latency_ms.snapshot(),
        }
        base["quality"] = {
            "top_score": self.top_score.snapshot(),
            "chunks_returned": self.chunks_returned.snapshot(),
        }
        return base


metrics = _RAGMetrics()


def now_ms() -> float:
    return time.perf_counter() * 1000.0
