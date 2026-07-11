"""In-memory Device Code OAuth sessions for the admin UI (Grok/xAI only)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .oauth.xai import (
    DEFAULT_API_BASE_URL,
    XAIAuth,
    XAIAuthError,
    resolve_chat_base_url,
    save_token,
)

logger = logging.getLogger("grok2api.admin.oauth")

_SESSIONS: Dict[str, "DeviceSession"] = {}
_LOCK = threading.Lock()
_MAX_SESSIONS = 32
_SESSION_TTL_S = 45 * 60


@dataclass
class DeviceSession:
    id: str
    status: str = "pending"  # pending | success | error | expired
    user_code: str = ""
    verification_uri: str = ""
    verification_uri_complete: str = ""
    expires_in: int = 0
    created_at: float = field(default_factory=time.time)
    error: str = ""
    email: str = ""
    credential_path: str = ""
    using_api: bool = False
    chat_base: str = ""
    _device: Any = field(default=None, repr=False)
    _disc_token_endpoint: str = field(default="", repr=False)


def _purge_old() -> None:
    now = time.time()
    dead = [
        sid
        for sid, s in _SESSIONS.items()
        if now - s.created_at > _SESSION_TTL_S
        or (s.status in ("success", "error", "expired") and now - s.created_at > 600)
    ]
    for sid in dead:
        _SESSIONS.pop(sid, None)
    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = sorted(_SESSIONS.values(), key=lambda s: s.created_at)
        for s in oldest[: len(_SESSIONS) - _MAX_SESSIONS]:
            _SESSIONS.pop(s.id, None)


def start_device_session(
    *,
    auths_dir: Optional[Path] = None,
    using_api: bool = False,
) -> DeviceSession:
    """Start device code flow and poll in a background thread."""
    auth = XAIAuth()
    disc = auth.discover()
    device = auth.request_device_code(
        disc.device_authorization_endpoint,
        disc.token_endpoint,
    )
    sid = uuid.uuid4().hex
    sess = DeviceSession(
        id=sid,
        status="pending",
        user_code=device.user_code,
        verification_uri=device.verification_uri,
        verification_uri_complete=device.verification_uri_complete or "",
        expires_in=int(device.expires_in or 0),
        using_api=using_api,
        _device=device,
        _disc_token_endpoint=disc.token_endpoint,
    )
    with _LOCK:
        _purge_old()
        _SESSIONS[sid] = sess

    def worker() -> None:
        try:
            token = auth.wait_for_authorization(device)
            storage = auth.create_token_storage(
                token,
                token_endpoint=disc.token_endpoint,
                base_url=DEFAULT_API_BASE_URL,
                using_api=using_api,
            )
            path = save_token(storage, auths_dir)
            with _LOCK:
                cur = _SESSIONS.get(sid)
                if not cur:
                    return
                cur.status = "success"
                cur.email = storage.email or storage.sub or path.name
                cur.credential_path = str(path)
                cur.chat_base = resolve_chat_base_url(storage)
                cur.using_api = storage.using_api
            logger.info("admin device oauth success session=%s email=%s", sid, cur.email)
        except XAIAuthError as e:
            with _LOCK:
                cur = _SESSIONS.get(sid)
                if cur:
                    msg = str(e)
                    cur.status = "expired" if "expired" in msg.lower() else "error"
                    cur.error = msg
            logger.warning("admin device oauth failed session=%s: %s", sid, e)
        except Exception as e:
            with _LOCK:
                cur = _SESSIONS.get(sid)
                if cur:
                    cur.status = "error"
                    cur.error = str(e)
            logger.exception("admin device oauth error session=%s", sid)

    t = threading.Thread(target=worker, name=f"xai-oauth-{sid[:8]}", daemon=True)
    t.start()
    return sess


def get_session(session_id: str) -> Optional[DeviceSession]:
    with _LOCK:
        return _SESSIONS.get(session_id)


def session_public(sess: DeviceSession) -> Dict[str, Any]:
    url = (sess.verification_uri_complete or sess.verification_uri or "").strip()
    return {
        "session_id": sess.id,
        "status": sess.status,
        "user_code": sess.user_code,
        "verification_uri": sess.verification_uri,
        "verification_uri_complete": sess.verification_uri_complete,
        "open_url": url,
        "expires_in": sess.expires_in,
        "error": sess.error or None,
        "email": sess.email or None,
        "credential_path": sess.credential_path or None,
        "using_api": sess.using_api,
        "chat_base": sess.chat_base or None,
    }
