"""AI API xarajatlarini kuzatish va chegaralash.

Har Mistral, Deepgram va Ollama chaqiruvi uchun xarajat hisoblanadi va
``cost_events`` jadvalida (yoki in-memory counter'da) saqlanadi. Admin
``/health/cost`` endpointida sutkali / oylik xarajatni ko'radi, kunlik
chegaradan oshsa AI API chaqiruvlari bloklanadi (circuit breaker).

Baholash
--------
* Mistral ``mistral-small-latest``: kiruvchi $0.2/1M, chiquvchi $0.6/1M tokens
  → ~$0.0004 per turn (o'rtacha 1500 input + 500 output tokens)
* Deepgram nova-2: $0.0043/min
* Ollama: lokal, xarajat yo'q

Chegaralar (``.env`` sozlanuvchi):
* ``AI_COST_DAILY_LIMIT_USD`` — default $10/kun
* ``AI_COST_MONTHLY_LIMIT_USD`` — default $200/oy
* Chegaraga yetilganda ``CostLimitExceeded`` xatolik qaytariladi (503)
"""
from __future__ import annotations

import datetime
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostLimitExceeded(Exception):
    """AI API xarajat chegarasi oshib ketdi."""


# --- Chegaralar ---
DAILY_LIMIT_USD = float(os.getenv("AI_COST_DAILY_LIMIT_USD", "10.0"))
MONTHLY_LIMIT_USD = float(os.getenv("AI_COST_MONTHLY_LIMIT_USD", "200.0"))
COST_TRACKING_ENABLED = os.getenv("AI_COST_TRACKING_ENABLED", "true").lower() not in ("false", "0", "no")


# --- Baholash parametrlari ---
# Mistral tokenlari uchun taxminiy chars→tokens ratio (ruscha matn uchun ~3 chars = 1 token)
MISTRAL_INPUT_PER_1M_USD = 0.20
MISTRAL_OUTPUT_PER_1M_USD = 0.60
MISTRAL_AVG_TOKEN_CHARS = 3.0

# Deepgram nova-2 tarifi
DEEPGRAM_USD_PER_MINUTE = 0.0043


@dataclass
class _Bucket:
    """Vaqt oynasi xarajatlari (sutka/oy)."""
    period: str  # "2026-04-25" yoki "2026-04"
    total_usd: float = 0.0
    calls: int = 0
    by_provider: Dict[str, float] = field(default_factory=dict)


_daily: Optional[_Bucket] = None
_monthly: Optional[_Bucket] = None
_lock = threading.Lock()


def _today_key() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _month_key() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m")


def _rotate_if_needed() -> None:
    """Sutka/oy o'zgarsa bucket ni qayta yaratadi."""
    global _daily, _monthly
    today = _today_key()
    month = _month_key()
    if _daily is None or _daily.period != today:
        _daily = _Bucket(period=today)
    if _monthly is None or _monthly.period != month:
        _monthly = _Bucket(period=month)


def estimate_mistral_cost(input_chars: int, output_chars: int) -> float:
    """Mistral Cloud API chaqiruvi uchun taxminiy xarajat."""
    in_tokens = input_chars / MISTRAL_AVG_TOKEN_CHARS
    out_tokens = output_chars / MISTRAL_AVG_TOKEN_CHARS
    return (in_tokens / 1_000_000) * MISTRAL_INPUT_PER_1M_USD + (out_tokens / 1_000_000) * MISTRAL_OUTPUT_PER_1M_USD


def estimate_deepgram_cost(audio_duration_sec: float) -> float:
    return (audio_duration_sec / 60.0) * DEEPGRAM_USD_PER_MINUTE


def check_limits() -> None:
    """Xarajat chegaralariga yetganligini tekshiradi. Oshgan bo'lsa xatolik."""
    if not COST_TRACKING_ENABLED:
        return
    with _lock:
        _rotate_if_needed()
        if _daily and _daily.total_usd >= DAILY_LIMIT_USD:
            raise CostLimitExceeded(
                f"Sutkali AI xarajat chegarasi oshib ketdi: ${_daily.total_usd:.2f} / ${DAILY_LIMIT_USD:.2f}"
            )
        if _monthly and _monthly.total_usd >= MONTHLY_LIMIT_USD:
            raise CostLimitExceeded(
                f"Oylik AI xarajat chegarasi oshib ketdi: ${_monthly.total_usd:.2f} / ${MONTHLY_LIMIT_USD:.2f}"
            )


def record(provider: str, cost_usd: float) -> None:
    """Xarajatni qayd qiladi. Provider: 'mistral', 'deepgram', 'ollama'."""
    if not COST_TRACKING_ENABLED or cost_usd <= 0:
        return
    with _lock:
        _rotate_if_needed()
        for bucket in (_daily, _monthly):
            if bucket is None:
                continue
            bucket.total_usd += cost_usd
            bucket.calls += 1
            bucket.by_provider[provider] = bucket.by_provider.get(provider, 0.0) + cost_usd

    # Ogohlantirish — chegaraning 80% iga yetilganida
    if _daily and _daily.total_usd >= DAILY_LIMIT_USD * 0.8:
        logger.warning(
            f"AI cost warning: daily ${_daily.total_usd:.2f} / ${DAILY_LIMIT_USD:.2f} (80%+)"
        )


def stats() -> dict:
    """/health/cost endpoint uchun — joriy xarajat holati."""
    with _lock:
        _rotate_if_needed()
        daily = _daily
        monthly = _monthly
        return {
            "enabled": COST_TRACKING_ENABLED,
            "limits_usd": {
                "daily": DAILY_LIMIT_USD,
                "monthly": MONTHLY_LIMIT_USD,
            },
            "today": {
                "period": daily.period if daily else None,
                "total_usd": round(daily.total_usd, 4) if daily else 0,
                "calls": daily.calls if daily else 0,
                "by_provider": {k: round(v, 4) for k, v in (daily.by_provider.items() if daily else [])},
                "remaining_usd": round(DAILY_LIMIT_USD - (daily.total_usd if daily else 0), 4),
            },
            "this_month": {
                "period": monthly.period if monthly else None,
                "total_usd": round(monthly.total_usd, 4) if monthly else 0,
                "calls": monthly.calls if monthly else 0,
                "remaining_usd": round(MONTHLY_LIMIT_USD - (monthly.total_usd if monthly else 0), 4),
            },
        }


def reset() -> None:
    """Chegaralarni qayta o'rnatish (test yoki admin uchun)."""
    global _daily, _monthly
    with _lock:
        _daily = None
        _monthly = None
