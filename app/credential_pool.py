"""Credential pool with round-robin selection + failure cooldown.

Design (aligned with CPA ``sdk/cliproxy/auth`` scheduler, minus the scheduler
plugin surface):

    Every ``xai-*.json`` under ``auths_dir`` becomes one ``TokenSlot``.
    ``UpstreamClient`` asks the pool for a slot per request via ``pick()``,
    which round-robins the ranks and skips any slot whose ``cooldown_until``
    is still in the future. On upstream failure, the caller reports the outcome
    with ``mark_cooldown()`` so the same slot won't be picked again during
    the cooldown window.

The pool is memory-only (no ``.cds`` persistence yet). xAI itself has no
quota-query endpoint, so cooldown is driven **passively** by the response
status/body we already receive.

Cooldown mapping (from cli-chat-proxy 429 body ``code`` field):

    ``subscription:*-exhausted``  →  24 h            (rolling free-tier tokens)
    ``resource-exhausted``        →  60 s × 2^level  (RPM window, capped 15 m)
    HTTP 401 / auth errors        →  5 min           (bad token / expired)
    other 429 / 5xx               →  60 s

Thread-safe via a single ``threading.Lock`` around all mutating operations —
picks, cooldown marks, reload, and the counters.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .oauth.xai import (
    TokenStorage,
    XAIAuthError,
    assert_xai_only,
    reject_foreign_filename,
)

logger = logging.getLogger("grok2api.credential_pool")


# Cooldown durations (seconds). Kept as module constants for now; expose as
# env-driven overrides only if operators actually need it.
COOLDOWN_QUOTA_EXHAUSTED = 24 * 60 * 60   # 24 h — free-tier rolling window
COOLDOWN_RATE_LIMIT_BASE = 60             # 60 s — RPM window, doubled per retry
COOLDOWN_RATE_LIMIT_MAX = 15 * 60         # cap 15 min
COOLDOWN_AUTH_ERROR = 5 * 60              # 5 min — token/refresh trouble
COOLDOWN_GENERIC = 60                     # 60 s — anything else 429/5xx


@dataclass
class TokenSlot:
    """One credential entry in the pool.

    ``storage`` is the loaded ``TokenStorage`` (mutated in-place when
    ``ensure_fresh_token`` runs against it). ``path`` is the on-disk
    ``xai-<email>.json`` — the pool key.
    """
    storage: TokenStorage
    path: Path
    cooldown_until: float = 0.0
    cooldown_reason: str = ""
    backoff_level: int = 0
    last_used: float = 0.0
    total_requests: int = 0
    total_errors: int = 0

    @property
    def email(self) -> str:
        return self.storage.email or self.storage.sub or ""

    def is_available(self, now: float) -> bool:
        return self.cooldown_until <= now


class CredentialPool:
    """Round-robin pool of xAI credentials with per-slot cooldown."""

    def __init__(self, auths_dir: Path) -> None:
        self._auths_dir = Path(auths_dir)
        self._lock = threading.Lock()
        self._slots: List[TokenSlot] = []
        self._next_idx = 0
        self._by_path: Dict[str, TokenSlot] = {}
        self._by_email: Dict[str, TokenSlot] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def reload(self) -> int:
        """Rescan ``auths_dir`` and rebuild the slot list.

        Preserves existing cooldown state for slots whose ``path`` still
        maps to a valid credential — new files are appended, deleted files
        are dropped, mutated files are refreshed. Returns the new size.
        """
        old_state = {slot.path.name: slot for slot in self._slots}
        loaded: List[TokenSlot] = []
        seen_paths: set = set()

        if self._auths_dir.is_dir():
            # Deterministic order: xai-current.json first (if it points at a
            # real file), then remaining xai-*.json sorted by mtime desc.
            current_target = _current_target(self._auths_dir)
            files: List[Path] = []
            if current_target and current_target.is_file():
                files.append(current_target)
                seen_paths.add(current_target.resolve())
            for path in sorted(
                self._auths_dir.glob("xai-*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if path.name == "xai-current.json":
                    continue
                rp = path.resolve()
                if rp in seen_paths:
                    continue
                files.append(path)
                seen_paths.add(rp)

            for path in files:
                storage = _try_load_storage(path)
                if storage is None:
                    continue
                prior = old_state.get(path.name)
                if prior is not None:
                    # Preserve cooldown + counters, refresh storage
                    prior.storage = storage
                    loaded.append(prior)
                else:
                    loaded.append(TokenSlot(storage=storage, path=path))

        with self._lock:
            self._slots = loaded
            self._by_path = {s.path.name: s for s in loaded}
            self._by_email = {s.email: s for s in loaded if s.email}
            # Reset RR pointer if it walked off the end
            if self._next_idx >= len(loaded):
                self._next_idx = 0

        logger.info(
            "credential pool reloaded: %d slot(s) from %s",
            len(loaded),
            self._auths_dir,
        )
        return len(loaded)

    def size(self) -> int:
        with self._lock:
            return len(self._slots)

    def available_count(self, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        with self._lock:
            return sum(1 for s in self._slots if s.is_available(now))

    # ------------------------------------------------------------------
    # Round-robin selection
    # ------------------------------------------------------------------

    def pick(self, *, exclude: Optional[set] = None) -> Optional[TokenSlot]:
        """Round-robin pick the next non-cooldown slot.

        ``exclude`` — slot ``path.name``s to skip. Callers pass the set of
        credentials already tried this request so we never charge a slot
        for a pick we're about to reject (advancing ``_next_idx`` and
        bumping ``total_requests`` before the caller filtered was the
        original bug this parameter fixes).

        Returns ``None`` if the pool is empty or every candidate slot is
        cooling down.
        """
        now = time.time()
        ex = exclude or set()
        with self._lock:
            n = len(self._slots)
            if n == 0:
                return None
            start = self._next_idx
            for i in range(n):
                idx = (start + i) % n
                slot = self._slots[idx]
                if slot.path.name in ex:
                    continue
                if slot.is_available(now):
                    self._next_idx = (idx + 1) % n
                    slot.last_used = now
                    slot.total_requests += 1
                    return slot
        return None

    def pick_any(self, *, exclude: Optional[set] = None) -> Optional[TokenSlot]:
        """Pick even if every slot is on cooldown — soonest-available wins.

        Emergency escape hatch: some client must still get a response even
        when the pool is fully cool. ``exclude`` mirrors ``pick()``.
        Prefer ``pick()`` in normal paths.
        """
        ex = exclude or set()
        with self._lock:
            candidates = [s for s in self._slots if s.path.name not in ex]
            if not candidates:
                return None
            slot = min(candidates, key=lambda s: s.cooldown_until)
            slot.last_used = time.time()
            slot.total_requests += 1
            return slot

    def primary(self) -> Optional[TokenSlot]:
        """First slot in stable order — used by init / list_models bootstrap."""
        with self._lock:
            return self._slots[0] if self._slots else None

    def get_by_path(self, path: Path) -> Optional[TokenSlot]:
        with self._lock:
            return self._by_path.get(path.name)

    def get_by_email(self, email: str) -> Optional[TokenSlot]:
        if not email:
            return None
        with self._lock:
            return self._by_email.get(email)

    # ------------------------------------------------------------------
    # Failure reporting
    # ------------------------------------------------------------------

    def mark_cooldown(
        self,
        slot: TokenSlot,
        *,
        status: int,
        body: str = "",
        error_code: str = "",
    ) -> float:
        """Record a failure and set the slot's cooldown window.

        Returns the resulting ``cooldown_until`` timestamp (0 = no cooldown).
        Caller passes the raw HTTP ``status`` and either the parsed upstream
        ``error_code`` (preferred) or the raw ``body`` — the routine derives
        the duration from those signals.
        """
        seconds, reason = _cooldown_for(status, error_code, body)
        if seconds <= 0:
            return 0.0
        now = time.time()
        with self._lock:
            slot.total_errors += 1
            if reason == "rate_limit":
                slot.backoff_level = min(slot.backoff_level + 1, 8)
                dur = min(seconds * (1 << (slot.backoff_level - 1)),
                          COOLDOWN_RATE_LIMIT_MAX)
            else:
                # Non-rate-limit reasons: fixed window, don't bump backoff
                dur = seconds
            slot.cooldown_until = now + dur
            slot.cooldown_reason = reason
        logger.warning(
            "cooldown %.0fs on %s (reason=%s status=%s code=%s)",
            dur, slot.email or slot.path.name, reason, status, error_code,
        )
        return slot.cooldown_until

    def note_success(self, slot: TokenSlot) -> None:
        """Successful request — reset backoff and clear any stale cooldown."""
        with self._lock:
            slot.backoff_level = 0
            slot.cooldown_until = 0.0
            slot.cooldown_reason = ""

    # ------------------------------------------------------------------
    # Introspection (admin UI)
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """JSON-friendly snapshot for the admin console."""
        now = time.time()
        with self._lock:
            slots = []
            for slot in self._slots:
                remaining = max(0.0, slot.cooldown_until - now)
                slots.append({
                    "email": slot.email,
                    "path": slot.path.name,
                    "available": slot.is_available(now),
                    "cooldown_until": (
                        _iso(slot.cooldown_until) if slot.cooldown_until else None
                    ),
                    "cooldown_seconds_remaining": int(remaining),
                    "cooldown_reason": slot.cooldown_reason,
                    "backoff_level": slot.backoff_level,
                    "total_requests": slot.total_requests,
                    "total_errors": slot.total_errors,
                    "last_used": _iso(slot.last_used) if slot.last_used else None,
                    "expired": slot.storage.expired,
                    "using_api": slot.storage.using_api,
                })
            return {
                "total": len(self._slots),
                "available": sum(1 for s in self._slots if s.is_available(now)),
                "in_cooldown": sum(
                    1 for s in self._slots if not s.is_available(now)
                ),
                "slots": slots,
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_storage(path: Path) -> Optional[TokenStorage]:
    """Read one xai-*.json and return TokenStorage, or None on any problem."""
    try:
        reject_foreign_filename(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert_xai_only(data, source_hint=str(path))
        return TokenStorage.from_dict(data)
    except XAIAuthError as e:
        logger.warning("pool skip %s: %s", path.name, e)
        return None
    except Exception as e:
        logger.warning("pool skip %s (load error): %s", path.name, e)
        return None


def _current_target(auths_dir: Path) -> Optional[Path]:
    """Read xai-current.json and return the pointed credential path."""
    current = auths_dir / "xai-current.json"
    if not current.is_file():
        return None
    try:
        meta = json.loads(current.read_text(encoding="utf-8"))
        p = Path(meta.get("path") or "")
        return p if p.is_file() else None
    except Exception:
        return None


def _cooldown_for(status: int, code: str, body: str) -> tuple[int, str]:
    """Map (status, error code, body) → (seconds, reason tag).

    Returns ``(0, "")`` when no cooldown should be applied.
    """
    code = (code or "").lower()
    body_lower = (body or "").lower()

    if status == 429:
        # xAI free-tier daily tokens exhausted — 24h rolling window
        if code.startswith("subscription:") and "exhausted" in code:
            return COOLDOWN_QUOTA_EXHAUSTED, "quota_exhausted"
        # RPM rate limit
        if code == "resource-exhausted" or "requests per minute" in body_lower:
            return COOLDOWN_RATE_LIMIT_BASE, "rate_limit"
        return COOLDOWN_GENERIC, "rate_limit"

    if status == 401 or status == 403:
        return COOLDOWN_AUTH_ERROR, "auth_error"

    if 500 <= status < 600:
        # Transient upstream error — brief cooldown to spread load
        return COOLDOWN_GENERIC, "server_error"

    return 0, ""


def _iso(ts: float) -> str:
    """Unix timestamp → ISO8601 UTC (no microseconds)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
