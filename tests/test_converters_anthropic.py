"""Anthropic ↔ official Grok /responses direct converters (no Chat hop)."""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest

from app.converters.anthropic import (
    anthropic_to_responses_request,
    responses_result_to_anthropic,
    stream_responses_to_anthropic,
)


def test_anthropic_to_responses_system_and_user_text():
    body = {
        "model": "grok-3",
        "system": "sys-prompt",
        "max_tokens": 32,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": "hello"}],
    }

    req = anthropic_to_responses_request(body, stream=False)

    assert req["model"] == "grok-3"
    assert req["stream"] is False
    assert req["instructions"] == "sys-prompt"
    assert req["max_output_tokens"] == 32
    assert req["temperature"] == 0.4
    # user block → single message item with input_text
    assert req["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    # never converted through Chat
    assert "messages" not in req
    assert "max_tokens" not in req


def test_anthropic_to_responses_tool_use_and_tool_result_roundtrip():
    body = {
        "model": "grok-3",
        "messages": [
            {"role": "user", "content": "run the tool"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "sure"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search",
                        "input": {"q": "grok"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "42"}],
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "search",
                "description": "search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "search"},
    }

    req = anthropic_to_responses_request(body, stream=True, model="grok-3-mini")

    assert req["model"] == "grok-3-mini"
    items = req["input"]

    # user text (input_text) — never becomes Chat
    assert items[0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "run the tool"}],
    }

    # assistant text (output_text) before the function_call
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "sure"}],
    }

    # tool_use → flat function_call item
    fc = items[2]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "toolu_1"
    assert fc["name"] == "search"
    assert json.loads(fc["arguments"]) == {"q": "grok"}

    # tool_result → function_call_output referencing the call_id
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "toolu_1",
        "output": "42",
    }

    # tools flattened to Responses shape (not nested under "function")
    assert req["tools"][0]["type"] == "function"
    assert req["tools"][0]["name"] == "search"
    assert req["tools"][0]["parameters"]["properties"] == {"q": {"type": "string"}}

    # tool_choice mapped to Responses form
    assert req["tool_choice"] == {"type": "function", "name": "search"}


def test_anthropic_to_responses_image_block():
    body = {
        "model": "grok-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AAAA",
                        },
                    },
                ],
            }
        ],
    }

    req = anthropic_to_responses_request(body, stream=False)
    parts = req["input"][0]["content"]

    assert parts[0] == {"type": "input_text", "text": "what's this?"}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


def test_responses_result_to_anthropic_text_and_tool_use():
    resp = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "upstream-grok",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi there"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": '{"q":"grok"}',
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }

    out = responses_result_to_anthropic(resp, "client-m")

    assert out["type"] == "message"
    assert out["model"] == "client-m"
    # thinking → text → tool_use, in order
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    assert out["content"][1]["text"] == "hi there"
    assert out["content"][2] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "search",
        "input": {"q": "grok"},
    }
    assert out["stop_reason"] == "tool_use"
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_responses_result_to_anthropic_incomplete_maps_to_max_tokens():
    resp = {
        "object": "response",
        "status": "incomplete",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "truncated"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    out = responses_result_to_anthropic(resp, "m")
    assert out["stop_reason"] == "max_tokens"
    assert out["content"] == [{"type": "text", "text": "truncated"}]


@pytest.mark.asyncio
async def test_stream_responses_to_anthropic_text_and_tools():
    """Official Responses SSE → Anthropic message SSE frames."""
    events = [
        json.dumps({"type": "response.created"}),
        json.dumps(
            {"type": "response.reasoning_summary_text.delta", "delta": "think"}
        ),
        json.dumps({"type": "response.output_text.delta", "delta": "he"}),
        json.dumps({"type": "response.output_text.delta", "delta": "llo"}),
        json.dumps(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "search",
                },
            }
        ),
        json.dumps(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"q":',
            }
        ),
        json.dumps(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '"x"}',
            }
        ),
        json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "function_call"}],
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                },
            }
        ),
        "[DONE]",
    ]

    async def lines() -> AsyncIterator[str]:
        for e in events:
            yield e

    frames: List[str] = []
    async for frame in stream_responses_to_anthropic(lines(), "m"):
        frames.append(frame)

    joined = "".join(frames)
    assert "message_start" in joined
    # thinking block precedes text block
    assert joined.index("thinking_delta") < joined.index("text_delta")
    # tool arguments streamed as input_json_delta (dumps uses compact separators)
    assert '"partial_json":"{\\"q\\":"' in joined
    # tool stop reason + usage in message_delta
    assert '"stop_reason":"tool_use"' in joined
    assert '"input_tokens":3' in joined
    assert '"output_tokens":4' in joined
    assert frames[-1].startswith("event: message_stop")
