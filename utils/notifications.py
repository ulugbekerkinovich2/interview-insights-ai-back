"""Per-user notification hub.

Persists notifications via the existing SQL model and pushes them live over a
dedicated WebSocket (``/ws/notifications``). Callable from sync FastAPI
endpoints via ``push_sync`` — we capture the FastAPI event loop once at
startup and schedule pushes onto it using ``run_coroutine_threadsafe``.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from fastapi import WebSocket
from sqlalchemy.orm import Session

import database

logger = logging.getLogger(__name__)


class NotificationHub:
    """In-memory registry of per-user WebSocket connections."""

    def __init__(self) -> None:
        self._connections: Dict[int, Set[WebSocket]] = defaultdict(set)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def has_loop(self) -> bool:
        return self._loop is not None and self._loop.is_running()

    async def register(self, user_id: int, ws: WebSocket) -> None:
        with self._lock:
            self._connections[user_id].add(ws)

    def unregister(self, user_id: int, ws: WebSocket) -> None:
        with self._lock:
            conns = self._connections.get(user_id)
            if conns and ws in conns:
                conns.discard(ws)
                if not conns:
                    self._connections.pop(user_id, None)

    def connected_users(self) -> List[int]:
        with self._lock:
            return list(self._connections.keys())

    def _targets(self, user_id: int) -> List[WebSocket]:
        with self._lock:
            return list(self._connections.get(user_id, ()))

    async def _send(self, targets: Iterable[WebSocket], payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    for conns in self._connections.values():
                        conns.discard(ws)

    def push_sync(self, user_id: int, payload: Dict[str, Any]) -> None:
        """Schedule a push from a sync context. No-op if nobody is connected."""
        targets = self._targets(user_id)
        if not targets:
            return
        if not self.has_loop():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send(targets, payload), self._loop)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("notification push scheduling failed: %s", exc)


hub = NotificationHub()


# --- Persistence + push helpers ----------------------------------------------

def _serialize(notif: database.Notification) -> Dict[str, Any]:
    return {
        "id": notif.id,
        "title": notif.title,
        "message": notif.message,
        "type": notif.type,
        "is_read": bool(notif.is_read),
        "user_id": notif.user_id,
        "created_at": notif.created_at.isoformat() if notif.created_at else None,
    }


def notify_user(
    db: Session,
    user_id: int,
    *,
    title: str,
    message: str,
    type: str = "info",
    meta: Optional[Dict[str, Any]] = None,
) -> database.Notification:
    notif = database.Notification(
        user_id=user_id,
        title=title,
        message=message,
        type=type,
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)

    payload = _serialize(notif)
    if meta:
        payload["meta"] = meta
    hub.push_sync(user_id, payload)
    return notif


def notify_role(
    db: Session,
    role: str,
    *,
    title: str,
    message: str,
    type: str = "info",
    exclude_user_id: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Create + push a notification for every active user with the given role."""
    q = db.query(database.User).filter(
        database.User.role == role,
        database.User.is_active.is_(True),
    )
    if exclude_user_id is not None:
        q = q.filter(database.User.id != exclude_user_id)

    sent = 0
    for user in q.all():
        notify_user(db, user.id, title=title, message=message, type=type, meta=meta)
        sent += 1
    return sent
