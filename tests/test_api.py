"""HTTP API smoke tests with mocked upstream."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.config import reload_settings
from app.main import create_app
from app.upstream import UpstreamError


def _app_client(client_key: str) -> TestClient:
    reload_settings()
    return TestClient(create_app())


def _mock_client(**methods: Any) -> MagicMock:
    """UpstreamClient mock defaulting to official wire (uses_official_wire=True)."""
    inst = MagicMock()
    inst.uses_official_wire = MagicMock(return_value=True)
    for name, value in methods.items():
        setattr(inst, name, value)
    return inst


def test_health_and_root(client_key: str):
    c = _app_client(client_key)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "POST /v1/chat/completions" in body["protocols"]
    assert "channels" not in body
    assert body["upstream_mode"] in ("auto", "oauth", "credential")
    assert c.get("/").json()["name"] == "grok2api"


def test_auth_required(client_key: str):
    c = _app_client(client_key)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_chat_completions_official(client_key: str):
    upstream_payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
    }
    with patch("app.handlers.UpstreamClient") as cls:
        cls.return_value = _mock_client(
            chat_completions=AsyncMock(return_value=upstream_payload)
        )
        c = _app_client(client_key)
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {client_key}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "pong"
    assert data["model"] == "test-model"


def test_messages_official_direct_responses(client_key: str):
    """Anthropic → /responses direct (no Chat hop)."""
    anth_upstream = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "upstream-m",
        "content": [{"type": "text", "text": "你好"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    with patch("app.handlers.UpstreamClient") as cls:
        inst = _mock_client(messages=AsyncMock(return_value=anth_upstream))
        cls.return_value = inst
        c = _app_client(client_key)
        r = c.post(
            "/v1/messages",
            headers={"x-api-key": client_key},
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "你好"
    assert data["model"] == "test-model"
    inst.messages.assert_awaited_once()
    inst.chat_completions.assert_not_called()


def test_responses_protocol_official_native(client_key: str):
    """Official wire: Responses client posts native /responses."""
    completed = {
        "id": "resp_native",
        "object": "response",
        "status": "completed",
        "model": "upstream-m",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "native-ok"}],
            }
        ],
        "output_text": "native-ok",
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }
    with patch("app.handlers.UpstreamClient") as cls:
        inst = _mock_client(responses=AsyncMock(return_value=completed))
        cls.return_value = inst
        c = _app_client(client_key)
        r = c.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {client_key}"},
            json={
                "model": "test-model",
                "input": "go",
                "max_output_tokens": 8,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "response"
    assert data["output_text"] == "native-ok"
    assert data["model"] == "test-model"
    inst.responses.assert_awaited_once()
    inst.chat_completions.assert_not_called()


def test_count_tokens_endpoints(client_key: str):
    c = _app_client(client_key)
    h = {"Authorization": f"Bearer {client_key}"}
    r1 = c.post(
        "/v1/messages/count_tokens",
        headers=h,
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "count me"}],
        },
    )
    assert r1.status_code == 200
    assert "input_tokens" in r1.json()

    r2 = c.post(
        "/v1/responses/input_tokens",
        headers=h,
        json={"model": "test-model", "input": "count me"},
    )
    assert r2.status_code == 200
    assert r2.json()["object"] == "response.input_tokens"


def test_upstream_error_openai_shape(client_key: str):
    with patch("app.handlers.UpstreamClient") as cls:
        cls.return_value = _mock_client(
            chat_completions=AsyncMock(
                side_effect=UpstreamError(502, "bad gateway", None)
            )
        )
        c = _app_client(client_key)
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {client_key}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["type"] == "upstream_error"


def test_upstream_error_anthropic_shape(client_key: str):
    """Anthropic path calls messages() directly; map UpstreamError to Anthropic envelope."""
    with patch("app.handlers.UpstreamClient") as cls:
        cls.return_value = _mock_client(
            messages=AsyncMock(
                side_effect=UpstreamError(429, "rate limited", None)
            )
        )
        c = _app_client(client_key)
        r = c.post(
            "/v1/messages",
            headers={"x-api-key": client_key},
            json={
                "model": "test-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    assert r.status_code == 429
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "rate limited" in body["error"]["message"]


def test_models_list(client_key: str):
    with patch("app.handlers.UpstreamClient") as cls:
        cls.return_value = _mock_client(
            list_models=AsyncMock(
                return_value={
                    "object": "list",
                    "data": [
                        {
                            "id": "test-model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "t",
                        }
                    ],
                }
            )
        )
        c = _app_client(client_key)
        r = c.get("/v1/models", headers={"Authorization": f"Bearer {client_key}"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "test-model"


def test_admin_status_no_channels(client_key: str):
    """Admin status is official Grok only — no channel inventory."""
    c = _app_client(client_key)
    h = {"Authorization": f"Bearer {client_key}"}
    st = c.get("/admin/api/status", headers=h)
    assert st.status_code == 200
    body = st.json()
    assert body.get("ok") is True
    assert "channels" not in body
    assert "providers_store" not in body
    assert c.get("/admin/api/channels", headers=h).status_code == 404


def test_admin_logs_and_summary(client_key: str, tmp_path, monkeypatch):
    """Request log APIs read the JSONL store; middleware may also append."""
    monkeypatch.setenv("GROK2API_DATA_DIR", str(tmp_path / "logs-data"))
    monkeypatch.setenv("REQUEST_LOG_ENABLED", "true")
    reload_settings()

    from app.request_log import RequestLogRecord, get_request_log_store, reset_request_log_store

    reset_request_log_store()
    store = get_request_log_store()
    store.append(
        RequestLogRecord(
            ts="2026-07-12T12:00:00+00:00",
            method="POST",
            path="/v1/chat/completions",
            status=200,
            duration_ms=11.0,
            model="test-model",
        )
    )

    c = _app_client(client_key)
    h = {"Authorization": f"Bearer {client_key}"}

    logs = c.get("/admin/api/logs?limit=10&path_prefix=/v1/", headers=h)
    assert logs.status_code == 200
    body = logs.json()
    assert body["ok"] is True
    assert body["total"] >= 1
    assert any(i.get("path") == "/v1/chat/completions" for i in body["items"])

    summary = c.get("/admin/api/logs/summary", headers=h)
    assert summary.status_code == 200
    assert "last_1h" in summary.json() or "last_24h" in summary.json()

    assert c.get("/admin/api/logs", headers={}).status_code == 401
    reset_request_log_store()


def test_admin_spa_or_legacy(client_key: str):
    """/admin serves SPA dist when built, else legacy admin.html."""
    c = _app_client(client_key)
    r = c.get("/admin")
    assert r.status_code == 200
    html = r.text
    assert "Grok2API" in html or "root" in html or "admin" in html.lower()

    # Client route fallback must not swallow /admin/api/*
    h = {"Authorization": f"Bearer {client_key}"}
    st = c.get("/admin/api/status", headers=h)
    assert st.status_code == 200
    assert st.json().get("ok") is True
