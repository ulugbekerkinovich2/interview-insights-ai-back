"""Thread-safe token bucket rate limiter.

Used to throttle external API calls (Mistral embed / chat) so we stay within
provider quotas even under concurrent interview + chat traffic.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, rate: float, burst: Optional[int] = None, name: str = "bucket"):
        self.rate = float(max(rate, 0.01))
        self.burst = int(burst) if burst and burst > 0 else max(1, int(round(self.rate)))
        self.tokens = float(self.burst)
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.name = name
        self._waited_total = 0.0
        self._denied_total = 0

    def acquire(self, tokens: int = 1, *, blocking: bool = True, timeout: float = 30.0) -> bool:
        tokens = max(1, int(tokens))
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                    self.last = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                wait = (tokens - self.tokens) / self.rate

            if not blocking:
                with self.lock:
                    self._denied_total += 1
                return False
            if time.monotonic() + wait > deadline:
                with self.lock:
                    self._denied_total += 1
                logger.warning(
                    "rate_limit %s: wait %.2fs exceeds timeout %.2fs", self.name, wait, timeout
                )
                return False

            sleep_for = min(wait, 0.5)
            time.sleep(sleep_for)
            with self.lock:
                self._waited_total += sleep_for

    def stats(self) -> dict:
        with self.lock:
            return {
                "name": self.name,
                "rate_per_sec": self.rate,
                "burst": self.burst,
                "available": round(self.tokens, 2),
                "waited_total_sec": round(self._waited_total, 2),
                "denied_total": self._denied_total,
            }
