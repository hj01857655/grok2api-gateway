"""Anthropic Messages converter tests."""

from __future__ import annotations

import json

import pytest

from app.converters.anthropic import (
    anthropic_to_chat,
    chat_to_anthropic,
    stream_chat_to_anthropic,
)


def test_anthropic_to_chat_system_and_tools():
    body = {
        "model": "claude-like",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "be brief"}],
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "lookup",
                "description": "look up",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
        "tool_choice": {"type": "any"},
        "stop_sequences": ["END"],
        "temperature": 0.2,
    }
    chat = anthropic_to_chat(body)
    assert chat["model"] == "claude-like"
    assert chat["messages"][0] == {"role": "system", "content": "be brief"}
    assert chat["messages"][1]["content"] == "hi"
    assert chat["max_tokens"] == 64
    assert chat["temperature"] == 0.2
    assert chat["stop"] == ["END"]
    assert chat["tools"][0]["function"]["name"] == "lookup"
    assert chat["tools"][0]["function"]["parameters"]["properties"]["q"]["type"] == "string"
    assert chat["tool_choice"] == "required"


def test_anthropic_tool_use_and_result_roundtrip_shape():
    body = {
        "model": "m",
        "max_tokens": 32,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"q": "x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result-text",
                    }
                ],
            },
        ],
    }
    chat = anthropic_to_chat(body)
    assert chat["messages"][0]["role"] == "assistant"
    assert chat["messages"][0]["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(chat["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {
        "q": "x"
    }
    assert chat["messages"][1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "result-text",
    }


def test_chat_to_anthropic_tool_use_and_thinking():
    chat = {
        "id": "chatcmpl-1",
        "model": "upstream-m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"a"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7},
    }
    out = chat_to_anthropic(chat, "client-m")
    assert out["type"] == "message"
    assert out["model"] == "client-m"
    assert out["stop_reason"] == "tool_use"
    types = [b["type"] for b in out["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert out["content"][0]["thinking"] == "think"
    assert out["content"][2]["input"] == {"q": "a"}
    assert out["usage"]["input_tokens"] == 3


@pytest.mark.asyncio
async def test_stream_chat_to_anthropic_text_and_tools():
    async def lines():
        yield json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "r1",
                        }
                    }
                ]
            }
        )
        yield json.dumps({"choices": [{"delta": {"content": "hel"}}]})
        yield json.dumps({"choices": [{"delta": {"content": "lo"}}]})
        yield json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "function": {"name": "lookup", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        yield json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"q":1}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        yield "[DONE]"

    events = []
    async for frame in stream_chat_to_anthropic(lines(), "m"):
        events.append(frame)

    joined = "".join(events)
    assert "event: message_start" in joined
    assert "thinking_delta" in joined
    assert "text_delta" in joined
    assert "input_json_delta" in joined
    assert "tool_use" in joined
    assert "message_stop" in joined
    assert "tool_use" in joined or "stop_reason" in joined
