"""HTTP API smoke tests with mocked upstream."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import reload_settings
from app.main import create_app
from app.upstream import UpstreamError


def _app_client(client_key: str) -> TestClient:
    reload_settings()
    return TestClient(create_app())


def test_health_and_root(client_key: str):
    c = _app_client(client_key)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "POST /v1/chat/completions" in body["protocols"]
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
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    with patch("app.products.UpstreamClient") as cls:
        inst = cls.return_value
        inst.chat_completions = AsyncMock(return_value=upstream_payload)
        c = _app_client(client_key)
        r = c.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {client_key}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "pong"
    # client model id restored
    assert data["model"] == "test-model"


def test_messages_protocol(client_key: str):
    upstream = {
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
        inst = cls.return_value
        inst.chat_completions = AsyncMock(return_value=upstream)
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
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "你好"
    assert data["model"] == "test-model"


def test_responses_protocol(client_key: str):
    upstream = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with patch("app.products.UpstreamClient") as cls:
        inst = cls.return_value
        inst.chat_completions = AsyncMock(return_value=upstream)
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
        inst = cls.return_value
        inst.chat_completions = AsyncMock(
            side_effect=UpstreamError(502, "bad gateway", None)
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
        inst = cls.return_value
        inst.chat_completions = AsyncMock(
            side_effect=UpstreamError(429, "rate limited", None)
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
        inst = cls.return_value
        inst.list_models = AsyncMock(
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
        c = _app_client(client_key)
        r = c.get("/v1/models", headers={"Authorization": f"Bearer {client_key}"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "test-model"
