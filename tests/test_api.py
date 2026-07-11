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
    """UpstreamClient mock defaulting to mid-station wire (uses_official_wire=False)."""
    inst = MagicMock()
    inst.uses_official_wire = MagicMock(return_value=False)
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
    assert body["channels"]  # fixture added one managed channel
    assert c.get("/").json()["name"] == "grok2api"


def test_auth_required(client_key: str):
    c = _app_client(client_key)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_chat_completions_pass_through(client_key: str):
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
    with patch("app.products.UpstreamClient") as cls:
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


def test_messages_mid_station_pass_through(client_key: str):
    """Mid-station Anthropic: body goes to /messages as-is (no Chat convert)."""
    upstream = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "upstream-m",
        "content": [{"type": "text", "text": "你好"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    with patch("app.products.UpstreamClient") as cls:
        inst = _mock_client(messages=AsyncMock(return_value=upstream))
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
    call_body = inst.messages.await_args.args[0]
    assert "messages" in call_body
    assert call_body["max_tokens"] == 16
    # Must NOT have gone through chat_completions
    inst.chat_completions.assert_not_called()


def test_messages_official_converts_via_chat(client_key: str):
    """Official wire only speaks /responses — Anthropic converts once via Chat."""
    chat_upstream = {
        "id": "chatcmpl-a",
        "model": "u",
        "choices": [
            {
                "message": {"role": "assistant", "content": "你好"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
    with patch("app.products.UpstreamClient") as cls:
        inst = _mock_client(
            chat_completions=AsyncMock(return_value=chat_upstream)
        )
        inst.uses_official_wire = MagicMock(return_value=True)
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
    inst.chat_completions.assert_awaited_once()
    inst.messages.assert_not_called()


def test_responses_mid_station_pass_through(client_key: str):
    """Mid-station Responses: pass-through to /responses — no Chat hop."""
    upstream = {
        "id": "resp_mid",
        "object": "response",
        "status": "completed",
        "model": "upstream-m",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        "output_text": "done",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    with patch("app.products.UpstreamClient") as cls:
        inst = _mock_client(responses=AsyncMock(return_value=upstream))
        cls.return_value = inst
        c = _app_client(client_key)
        r = c.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {client_key}"},
            json={"model": "test-model", "input": "go", "max_output_tokens": 8},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "response"
    assert data["output_text"] == "done"
    assert data["model"] == "test-model"
    inst.responses.assert_awaited_once()
    inst.chat_completions.assert_not_called()
    call_body = inst.responses.await_args.args[0]
    assert "input" in call_body
    assert "messages" not in call_body


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
    with patch("app.products.UpstreamClient") as cls:
        inst = _mock_client(responses=AsyncMock(return_value=completed))
        inst.uses_official_wire = MagicMock(return_value=True)
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
    with patch("app.products.UpstreamClient") as cls:
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
    with patch("app.products.UpstreamClient") as cls:
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
    with patch("app.products.UpstreamClient") as cls:
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


def test_admin_channel_crud(client_key: str):
    """Channels only exist after admin add — not from env."""
    c = _app_client(client_key)
    h = {"Authorization": f"Bearer {client_key}"}

    listed = c.get("/admin/api/channels", headers=h)
    assert listed.status_code == 200
    before = len(listed.json()["channels"])

    created = c.post(
        "/admin/api/channels",
        headers=h,
        json={
            "name": "extra",
            "base_url": "https://extra.test/v1",
            "api_key": "extra-key",
            "models": "extra-model",
        },
    )
    assert created.status_code == 200
    cid = created.json()["channel"]["id"]

    listed2 = c.get("/admin/api/channels", headers=h).json()["channels"]
    assert len(listed2) == before + 1

    deleted = c.delete(f"/admin/api/channels/{cid}", headers=h)
    assert deleted.status_code == 200
    assert len(c.get("/admin/api/channels", headers=h).json()["channels"]) == before
