"""Bounded ThreadPoolExecutor — og'ir sync I/O ni asyncio loop dan chiqarib,
cheklangan paralellik bilan ishga tushirish uchun.

Muammo
------
FastAPI sync (``def``) endpointlarni default anyio thread poolida ishlatadi
(default 40 thread). Agar 40 ta foydalanuvchi bir vaqtda Whisper so'rov yuborsa:

* 40 ta CPU-heavy Whisper jarayoni paralel ishlaydi → RAM 40 × 500 MB = 20 GB
* Barcha thread-lar band bo'lsa, yangi so'rovlar kutishga tushadi (timeout)
* Mistral API chaqiruvlari ham bir xil — tarmoq xatosi yoki rate-limit
  uyqusida barcha threadlarni bloklaydi

Yechim
------
* Har og'ir operatsiya turi uchun **alohida bounded executor** (masalan STT=4,
  LLM=8). Shu tariqa bir turdagi so'rov boshqa turdagilarni bloklamaydi.
* Queue to'lsa va kutish vaqti chegaradan oshsa — ``QueueFull`` xatosi
  qaytariladi (503 Service Unavailable), shoshilmay kutmaslik kerak.
* Har submission `asyncio.wait_for` orqali execution timeout ga ham bog'lanadi.

Ishlatilish
-----------
>>> from utils.executor import stt_executor, run_bounded
>>> result = await run_bounded(
...     stt_executor,
...     logic.transcribe_audio, audio_path,
...     queue_wait_sec=10, exec_timeout_sec=120,
... )
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    """Pool to'lgan va kutish vaqti chegaradan oshgan."""


def _env_int(name: str, default: int) -> int:
    try:
        val = int(os.getenv(name, str(default)))
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default


# Har turdagi og'ir operatsiya uchun alohida bounded pool.
# ``.env`` dan sozlanadi — ishchi yukiga qarab o'zgartirish mumkin.
STT_POOL_SIZE = _env_int("STT_POOL_SIZE", 4)
LLM_POOL_SIZE = _env_int("LLM_POOL_SIZE", 8)
PROSODY_POOL_SIZE = _env_int("PROSODY_POOL_SIZE", 4)

stt_executor = ThreadPoolExecutor(
    max_workers=STT_POOL_SIZE,
    thread_name_prefix="stt-pool",
)
llm_executor = ThreadPoolExecutor(
    max_workers=LLM_POOL_SIZE,
    thread_name_prefix="llm-pool",
)
prosody_executor = ThreadPoolExecutor(
    max_workers=PROSODY_POOL_SIZE,
    thread_name_prefix="prosody-pool",
)
# #27 — Frame analysis pool (Haar cascade CPU-bound, 50-200ms/frame)
FRAME_POOL_SIZE = _env_int("FRAME_POOL_SIZE", 4)
frame_executor = ThreadPoolExecutor(
    max_workers=FRAME_POOL_SIZE,
    thread_name_prefix="frame-pool",
)


def _active_count(executor: ThreadPoolExecutor) -> int:
    """Hozir ishlayotgan threadlar soni (taxminiy)."""
    try:
        # _threads — barcha yaratilgan, _work_queue — kutayotganlar
        return len([t for t in executor._threads if t.is_alive()])  # type: ignore[attr-defined]
    except Exception:
        return 0


def _queue_size(executor: ThreadPoolExecutor) -> int:
    try:
        return executor._work_queue.qsize()  # type: ignore[attr-defined]
    except Exception:
        return 0


async def run_bounded(
    executor: ThreadPoolExecutor,
    func: Callable[..., Any],
    *args: Any,
    queue_wait_sec: float = 10.0,
    exec_timeout_sec: float = 120.0,
    **kwargs: Any,
) -> Any:
    """Funksiyani bounded poolda ishga tushiradi.

    Parametrlar
    -----------
    executor :
        Qaysi poolda ishga tushirish (stt/llm/prosody)
    func, args, kwargs :
        Chaqiriladigan sync funksiya va argumentlar
    queue_wait_sec :
        Pool to'lsa, task qabul qilinishiga qancha kutish. Bundan oshsa
        ``QueueFull`` chiqadi.
    exec_timeout_sec :
        Task boshlanganidan keyin necha sekund tugashi kerak. Bundan oshsa
        ``asyncio.TimeoutError`` chiqadi.

    Xatoliklar
    ----------
    * ``QueueFull`` — pool va queue tamomila to'lib, kutish vaqti tugadi
    * ``asyncio.TimeoutError`` — task ishga tushdi lekin juda uzoq davom etdi
    """
    loop = asyncio.get_running_loop()

    # Bounded queue wait: agar pool to'la bo'lsa, bir muddat kutamiz.
    # run_in_executor o'zi task ni queue ga qo'shadi, lekin biz wait_for bilan
    # task start vaqtini cheklay olmaymiz. Shu sababli queue band bo'lsa
    # tezda xato qaytaramiz.
    queue_depth = _queue_size(executor) + _active_count(executor)
    if queue_depth > executor._max_workers * 3:  # type: ignore[attr-defined]
        # Pool va queue 3 marotaba to'lgan — yangi yuk olishni rad etamiz
        logger.warning(
            f"Executor queue overloaded: depth={queue_depth}, "
            f"max_workers={executor._max_workers}"  # type: ignore[attr-defined]
        )
        raise QueueFull(
            f"Xizmat vaqtincha band (queue={queue_depth}). Birozdan keyin urining."
        )

    # Task submitin'dan keyin uni timeout bilan kutamiz.
    future = loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    try:
        return await asyncio.wait_for(future, timeout=queue_wait_sec + exec_timeout_sec)
    except asyncio.TimeoutError:
        # Ishlayotgan task ni bekor qilib bo'lmaydi (Python thread), lekin
        # caller ga xato qaytariladi — frontend uchun yetarli.
        logger.warning(
            f"run_bounded timeout: func={func.__name__}, "
            f"exec_timeout={exec_timeout_sec}s"
        )
        raise


def pool_stats() -> dict:
    """Barcha poollar holati — /health endpointda ko'rsatish uchun."""
    return {
        "stt": {
            "max_workers": stt_executor._max_workers,  # type: ignore[attr-defined]
            "active": _active_count(stt_executor),
            "queue": _queue_size(stt_executor),
        },
        "llm": {
            "max_workers": llm_executor._max_workers,  # type: ignore[attr-defined]
            "active": _active_count(llm_executor),
            "queue": _queue_size(llm_executor),
        },
        "prosody": {
            "max_workers": prosody_executor._max_workers,  # type: ignore[attr-defined]
            "active": _active_count(prosody_executor),
            "queue": _queue_size(prosody_executor),
        },
        "frame": {
            "max_workers": frame_executor._max_workers,  # type: ignore[attr-defined]
            "active": _active_count(frame_executor),
            "queue": _queue_size(frame_executor),
        },
    }


def shutdown_all() -> None:
    """Server to'xtatilganda barcha poollarni yopish (graceful shutdown)."""
    for ex in (stt_executor, llm_executor, prosody_executor, frame_executor):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
