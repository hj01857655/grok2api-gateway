"""Persisted request metadata logs for the admin console.

JSONL files under ``{home}/logs/requests-YYYYMMDD.jsonl`` with day + size
rotation and retention purge. Full prompt bodies are off by default.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_DAY_RE = re.compile(r"requests-(\d{8})(?:_\d+)?\.jsonl$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y%m%d")


@dataclass
class RequestLogRecord:
    ts: str
    method: str
    path: str
    status: int
    duration_ms: float
    model: Optional[str] = None
    stream: Optional[bool] = None
    error: Optional[str] = None
    client: Optional[str] = None
    # Optional truncated body snippet when REQUEST_LOG_BODY_MAX > 0
    body_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class RequestLogStore:
    """Thread-safe JSONL append + query for request metadata."""

    root: Path
    keep_days: int = 7
    max_mb: float = 50.0
    body_max: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _current_path: Optional[Path] = field(default=None, repr=False)
    _seq: int = field(default=0, repr=False)

    def logs_dir(self) -> Path:
        d = self.root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _max_bytes(self) -> int:
        return max(1, int(self.max_mb * 1024 * 1024))

    def _base_name(self, day: Optional[str] = None) -> str:
        return f"requests-{day or _day_stamp()}.jsonl"

    def _active_path(self) -> Path:
        """Pick today's file, or size-rotated ``requests-YYYYMMDD_N.jsonl``."""
        day = _day_stamp()
        base = self.logs_dir() / self._base_name(day)
        max_b = self._max_bytes()
        if not base.exists() or base.stat().st_size < max_b:
            return base
        # Size rotate: find next free suffix
        n = 1
        while True:
            p = self.logs_dir() / f"requests-{day}_{n}.jsonl"
            if not p.exists() or p.stat().st_size < max_b:
                return p
            n += 1

    def append(self, record: RequestLogRecord | Dict[str, Any]) -> None:
        if isinstance(record, RequestLogRecord):
            payload = record.to_dict()
        else:
            payload = {k: v for k, v in record.items() if v is not None}
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            path = self._active_path()
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
            self._current_path = path

    def purge_old(self, now: Optional[datetime] = None) -> int:
        """Delete log files older than keep_days. Returns count removed."""
        now = now or _utc_now()
        cutoff = (now - timedelta(days=max(0, self.keep_days))).strftime("%Y%m%d")
        removed = 0
        d = self.logs_dir()
        with self._lock:
            for p in d.glob("requests-*.jsonl"):
                m = _DAY_RE.search(p.name)
                if not m:
                    continue
                if m.group(1) < cutoff:
                    try:
                        p.unlink()
                        removed += 1
                    except OSError:
                        pass
        return removed

    def _iter_files_newest_first(self) -> List[Path]:
        files = list(self.logs_dir().glob("requests-*.jsonl"))

        def sort_key(p: Path) -> tuple:
            m = _DAY_RE.search(p.name)
            day = m.group(1) if m else "00000000"
            # suffix index for same day (base file sorts as 0)
            if "_" in p.stem:
                try:
                    idx = int(p.stem.rsplit("_", 1)[-1])
                except ValueError:
                    idx = 0
            else:
                idx = 0
            return (day, idx)

        files.sort(key=sort_key, reverse=True)
        return files

    def query(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        path_prefix: Optional[str] = None,
        status_min: Optional[int] = None,
        status_max: Optional[int] = None,
        since: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan recent files reverse-chronologically; return page + total match estimate."""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        matched: List[Dict[str, Any]] = []
        total = 0
        # Collect newest-first by reading each file bottom-up
        for path in self._iter_files_newest_first():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for raw in reversed(lines):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if path_prefix and not str(row.get("path", "")).startswith(path_prefix):
                    continue
                st = row.get("status")
                if status_min is not None and (st is None or int(st) < status_min):
                    continue
                if status_max is not None and (st is None or int(st) > status_max):
                    continue
                if since:
                    ts = str(row.get("ts") or "")
                    if ts < since:
                        continue
                total += 1
                if total > offset and len(matched) < limit:
                    matched.append(row)
        return {
            "ok": True,
            "items": matched,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def summary(self) -> Dict[str, Any]:
        """Aggregate counts for last 1h and 24h (4xx/5xx)."""
        now = _utc_now()
        t1h = (now - timedelta(hours=1)).isoformat()
        t24h = (now - timedelta(hours=24)).isoformat()

        def empty_bucket() -> Dict[str, int]:
            return {"count": 0, "4xx": 0, "5xx": 0}

        b1 = empty_bucket()
        b24 = empty_bucket()

        for path in self._iter_files_newest_first():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = str(row.get("ts") or "")
                if ts < t24h:
                    continue
                st = int(row.get("status") or 0)
                for bucket, cutoff in ((b24, t24h), (b1, t1h)):
                    if ts >= cutoff:
                        bucket["count"] += 1
                        if 400 <= st < 500:
                            bucket["4xx"] += 1
                        elif st >= 500:
                            bucket["5xx"] += 1

        return {
            "ok": True,
            "last_1h": b1,
            "last_24h": b24,
            "logs_dir": str(self.logs_dir()),
        }


_store: Optional[RequestLogStore] = None
_store_lock = threading.Lock()


def get_request_log_store(
    root: Optional[Path] = None,
    *,
    keep_days: int = 7,
    max_mb: float = 50.0,
    body_max: int = 0,
) -> RequestLogStore:
    """Process-wide store; recreated when root changes (tests)."""
    global _store
    if root is None:
        from .config import get_settings

        s = get_settings()
        root = s.home_dir()
        keep_days = s.request_log_keep_days
        max_mb = s.request_log_max_mb
        body_max = s.request_log_body_max
    with _store_lock:
        if (
            _store is None
            or _store.root != root
            or _store.keep_days != keep_days
            or _store.max_mb != max_mb
            or _store.body_max != body_max
        ):
            _store = RequestLogStore(
                root=root,
                keep_days=keep_days,
                max_mb=max_mb,
                body_max=body_max,
            )
        return _store


def reset_request_log_store() -> None:
    global _store
    with _store_lock:
        _store = None


def record_request(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    error: Optional[str] = None,
    client: Optional[str] = None,
    body_preview: Optional[str] = None,
) -> None:
    """Append one record using settings-backed store (no-op if disabled)."""
    from .config import get_settings

    s = get_settings()
    if not s.request_log_enabled:
        return
    store = get_request_log_store()
    err = error
    if err and len(err) > 500:
        err = err[:500] + "…"
    store.append(
        RequestLogRecord(
            ts=_utc_now().isoformat(),
            method=method,
            path=path,
            status=int(status),
            duration_ms=round(float(duration_ms), 2),
            model=model,
            stream=stream,
            error=err,
            client=client,
            body_preview=body_preview,
        )
    )


def should_skip_path(path: str) -> bool:
    """Skip SPA static assets; always log /v1/* and /admin/api/*."""
    if path.startswith("/v1/") or path.startswith("/admin/api"):
        return False
    if path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return True
    # SPA assets under /admin
    if path.startswith("/admin/assets/") or path.startswith("/admin/assets"):
        return True
    if path.startswith("/admin/") and any(
        path.endswith(ext)
        for ext in (
            ".js",
            ".css",
            ".map",
            ".svg",
            ".png",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
        )
    ):
        return True
    return False
