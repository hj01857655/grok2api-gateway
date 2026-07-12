"""OpenAI Responses converter tests (mid-station + official native/bridge)."""

from __future__ import annotations

import json

import pytest

from app.converters.responses import (
    chat_to_responses,
    chat_to_responses_request,
    collect_responses_completed,
    prepare_official_responses_request,
    responses_result_to_chat,
    responses_to_chat,
    stream_chat_to_responses,
    stream_responses_to_chat,
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


def test_chat_to_responses_request_for_official_wire():
    """Official token: client Chat → Responses request body (one hop)."""
    chat = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 16,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    req = chat_to_responses_request(chat, stream=True)
    assert req["model"] == "grok-3"
    assert req["stream"] is True
    assert req["instructions"] == "be brief"
    assert req["max_output_tokens"] == 16
    assert req["input"][0]["role"] == "user"
    assert req["tools"][0]["name"] == "lookup"
    assert "messages" not in req


def test_prepare_official_responses_request_native_no_chat():
    """Official token: client Responses stays Responses (sanitize only)."""
    body = {
        "model": "grok-3",
        "instructions": "sys",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "max_tokens": 32,
        "temperature": 0.2,
        "previous_response_id": "resp_keep_me",
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "name": "add",
                "description": "a",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "tool_choice": "auto",
    }
    req = prepare_official_responses_request(body, stream=True, model="resolved-m")
    assert req["model"] == "resolved-m"
    assert req["stream"] is True
    assert req["instructions"] == "sys"
    assert req["input"][0]["type"] == "message"
    assert req["max_output_tokens"] == 32
    assert "max_tokens" not in req
    # previous_response_id is the server-side continuation handle for multi-turn
    # Responses — it MUST pass through, not be sanitized away.
    assert req["previous_response_id"] == "resp_keep_me"
    assert "stream_options" not in req
    assert "messages" not in req  # never became Chat
    assert req["tools"][0]["name"] == "add"


def test_responses_result_to_chat_roundtrip():
    resp = {
        "id": "resp_1",
        "object": "response",
        "created_at": 100,
        "model": "up",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "think"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            },
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    }
    chat = responses_result_to_chat(resp, client_model="client-m")
    assert chat["object"] == "chat.completion"
    assert chat["model"] == "client-m"
    msg = chat["choices"][0]["message"]
    assert msg["content"] == "hello"
    assert msg["reasoning_content"] == "think"
    assert msg["tool_calls"][0]["id"] == "c1"
    assert chat["choices"][0]["finish_reason"] == "tool_calls"
    assert chat["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_collect_responses_completed():
    async def lines():
        yield json.dumps(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "x"}],
                },
            }
        )
        yield json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "object": "response",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )

    done = await collect_responses_completed(lines())
    assert done["id"] == "r1"
    assert done["output"][0]["type"] == "message"


@pytest.mark.asyncio
async def test_stream_responses_to_chat():
    async def lines():
        yield json.dumps({"type": "response.created", "response": {}})
        yield json.dumps(
            {"type": "response.output_text.delta", "delta": "hi"}
        )
        yield json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            }
        )

    chunks = []
    async for b in stream_responses_to_chat(lines(), client_model="m"):
        chunks.append(b.decode("utf-8"))
    joined = "".join(chunks)
    assert "chat.completion.chunk" in joined
    assert '"content": "hi"' in joined or '"content":"hi"' in joined
    assert "[DONE]" in joined


@pytest.mark.asyncio
async def test_stream_chat_to_responses_text():
    async def lines():
        yield json.dumps(
            {
                "choices": [
                    {"delta": {"role": "assistant", "content": "a"}, "index": 0}
                ]
            }
        )
        yield json.dumps(
            {
                "choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }
        )
        yield "[DONE]"

    frames = []
    async for f in stream_chat_to_responses(lines(), "m"):
        frames.append(f)
    text = "".join(frames)
    assert "response.created" in text
    assert "response.output_text.delta" in text
    assert "response.completed" in text


@pytest.mark.asyncio
async def test_stream_chat_to_responses_reasoning_and_live_tool_deltas():
    async def lines():
        yield json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "r1",
                        }
                    }
                ]
            }
        )
        yield json.dumps(
            {"choices": [{"delta": {"content": "hello"}}]}
        )
        yield json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {
                                        "name": "f",
                                        "arguments": "",
                                    },
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
                                    "function": {"arguments": '{"a":1}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        yield "[DONE]"

    frames = []
    async for f in stream_chat_to_responses(lines(), "m"):
        frames.append(f)
    text = "".join(frames)
    assert "response.reasoning_summary_text.delta" in text
    assert "response.output_text.delta" in text
    assert "response.function_call_arguments.delta" in text
    assert "response.completed" in text
