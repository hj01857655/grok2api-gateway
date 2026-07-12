"""Tests for request log store (tmp_path, no real home dir)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.request_log import (
    RequestLogRecord,
    RequestLogStore,
    should_skip_path,
)


def _rec(
    path: str = "/v1/chat/completions",
    status: int = 200,
    ts: str | None = None,
    model: str | None = "grok-3",
) -> RequestLogRecord:
    return RequestLogRecord(
        ts=ts or datetime.now(timezone.utc).isoformat(),
        method="POST",
        path=path,
        status=status,
        duration_ms=12.5,
        model=model,
        stream=False,
    )


def test_append_and_query(tmp_path: Path):
    store = RequestLogStore(root=tmp_path, keep_days=7, max_mb=50)
    store.append(_rec(path="/v1/chat/completions", status=200))
    store.append(_rec(path="/v1/responses", status=500, model="grok-4"))
    store.append(_rec(path="/admin/api/status", status=401))

    page = store.query(limit=10, offset=0)
    assert page["ok"] is True
    assert page["total"] == 3
    assert len(page["items"]) == 3
    # newest first
    assert page["items"][0]["path"] == "/admin/api/status"

    filtered = store.query(path_prefix="/v1/", status_min=500)
    assert filtered["total"] == 1
    assert filtered["items"][0]["model"] == "grok-4"


def test_query_pagination(tmp_path: Path):
    store = RequestLogStore(root=tmp_path)
    for i in range(5):
        store.append(_rec(path=f"/v1/x{i}"))
    p0 = store.query(limit=2, offset=0)
    p1 = store.query(limit=2, offset=2)
    assert len(p0["items"]) == 2
    assert len(p1["items"]) == 2
    assert p0["total"] == 5
    ids0 = {r["path"] for r in p0["items"]}
    ids1 = {r["path"] for r in p1["items"]}
    assert ids0.isdisjoint(ids1)


def test_summary_counts(tmp_path: Path):
    store = RequestLogStore(root=tmp_path)
    now = datetime.now(timezone.utc)
    store.append(_rec(status=200, ts=now.isoformat()))
    store.append(_rec(status=404, ts=now.isoformat()))
    store.append(_rec(status=502, ts=now.isoformat()))
    # old row outside 24h should not count
    old = (now - timedelta(hours=30)).isoformat()
    store.append(_rec(status=500, ts=old))

    s = store.summary()
    assert s["last_1h"]["count"] == 3
    assert s["last_1h"]["4xx"] == 1
    assert s["last_1h"]["5xx"] == 1
    assert s["last_24h"]["count"] == 3


def test_purge_old(tmp_path: Path):
    store = RequestLogStore(root=tmp_path, keep_days=1)
    logs = store.logs_dir()
    old_day = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y%m%d")
    old_file = logs / f"requests-{old_day}.jsonl"
    old_file.write_text(
        json.dumps(_rec().to_dict()) + "\n",
        encoding="utf-8",
    )
    store.append(_rec())
    removed = store.purge_old()
    assert removed >= 1
    assert not old_file.exists()


def test_size_rotate(tmp_path: Path):
    # tiny max so second append may rotate
    store = RequestLogStore(root=tmp_path, max_mb=0.0001)  # ~100 bytes
    store.append(_rec(path="/v1/a"))
    store.append(_rec(path="/v1/b"))
    files = list(store.logs_dir().glob("requests-*.jsonl"))
    assert len(files) >= 1
    # query still sees both
    q = store.query(limit=10)
    assert q["total"] >= 2


def test_should_skip_path():
    assert should_skip_path("/v1/chat/completions") is False
    assert should_skip_path("/admin/api/status") is False
    assert should_skip_path("/admin/assets/index.js") is True
    assert should_skip_path("/health") is True
    assert should_skip_path("/admin") is False
