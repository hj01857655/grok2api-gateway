"""OpenAI Responses converter tests."""

from __future__ import annotations

import json

import pytest

from app.converters.responses import (
    chat_to_responses,
    responses_to_chat,
    stream_chat_to_responses,
)


def test_responses_to_chat_string_input_and_tools():
    body = {
        "model": "m",
        "instructions": "sys",
        "input": "hello",
        "max_output_tokens": 40,
        "temperature": 0.1,
        "tools": [
            {
                "type": "function",
                "name": "add",
                "description": "add nums",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}},
                },
            }
        ],
        "tool_choice": "auto",
        "stream": True,
    }
    chat = responses_to_chat(body)
    assert chat["stream"] is True
    assert chat["max_tokens"] == 40
    assert chat["messages"][0] == {"role": "system", "content": "sys"}
    assert chat["messages"][1] == {"role": "user", "content": "hello"}
    assert chat["tools"][0]["function"]["name"] == "add"
    assert chat["tool_choice"] == "auto"


def test_responses_to_chat_function_call_items():
    body = {
        "model": "m",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "add",
                "arguments": '{"a":1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "2",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "again"}],
            },
        ],
    }
    chat = responses_to_chat(body)
    assert chat["messages"][0]["role"] == "assistant"
    assert chat["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert chat["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "2",
    }
    assert chat["messages"][2]["content"] == "again"


def test_chat_to_responses_with_tools_and_reasoning():
    chat = {
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "reasoning": "why",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "function": {"name": "add", "arguments": '{"a":2}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
        "model": "upstream",
    }
    out = chat_to_responses(chat, "client-m")
    assert out["object"] == "response"
    assert out["model"] == "client-m"
    assert out["status"] == "completed"
    types = [o["type"] for o in out["output"]]
    assert types == ["reasoning", "message", "function_call"]
    assert out["output"][2]["call_id"] == "call_9"
    assert out["usage"]["total_tokens"] == 6
    assert out["output_text"] == "ok"


@pytest.mark.asyncio
async def test_stream_chat_to_responses_text():
    async def lines():
        yield json.dumps({"choices": [{"delta": {"content": "a"}}]})
        yield json.dumps(
            {
                "choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
        yield "[DONE]"

    frames = []
    async for f in stream_chat_to_responses(lines(), "m"):
        frames.append(f)
    joined = "".join(frames)
    assert "response.created" in joined
    assert "response.output_text.delta" in joined
    assert '"delta":"a"' in joined or '"delta": "a"' in joined or "delta" in joined
    assert "response.completed" in joined
    assert "ab" in joined
