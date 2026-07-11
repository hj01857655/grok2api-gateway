"""OpenAI Responses API 鈫?OpenAI Chat Completions.

Request:  Responses /v1/responses body 鈫?Chat Completions body
Response: Chat Completions (JSON or SSE) 鈫?Responses object / SSE events
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..util import dumps, new_id, now_ts, parse_json, sse_event


# ---------------------------------------------------------------------------
# Request: Responses 鈫?Chat
# ---------------------------------------------------------------------------

def responses_to_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    messages.extend(_parse_input(body.get("input")))

    chat: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream")),
    }
    if body.get("temperature") is not None:
        chat["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        chat["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        chat["max_tokens"] = body["max_tokens"]

    tools = body.get("tools")
    if tools:
        chat["tools"] = _normalize_tools(tools)
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = body["tool_choice"]
    return chat


def _parse_input(input_val: Any) -> List[Dict[str, Any]]:
    if input_val is None:
        return []
    if isinstance(input_val, str):
        return [{"role": "user", "content": input_val}]
    if not isinstance(input_val, list):
        return [{"role": "user", "content": str(input_val)}]

    messages: List[Dict[str, Any]] = []
    for item in input_val:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue

        item_type = item.get("type")
        if item_type is None and "role" in item:
            item_type = "message"

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or new_id("call")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments")
                                if isinstance(item.get("arguments"), str)
                                else dumps(item.get("arguments") or {}),
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = dumps(output)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": output,
                }
            )
            continue

        if item_type in ("message", None) or "role" in item:
            role = item.get("role") or "user"
            content = item.get("content")
            messages.append({"role": role, "content": _content_to_chat(content)})
            continue

        # skip reasoning / unknown items
        if item_type in ("reasoning", "web_search_call"):
            continue
        messages.append({"role": "user", "content": dumps(item)})
    return messages


def _content_to_chat(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    texts: List[str] = []
    parts: List[Dict[str, Any]] = []
    has_image = False
    for c in content:
        if isinstance(c, str):
            texts.append(c)
            parts.append({"type": "text", "text": c})
            continue
        if not isinstance(c, dict):
            texts.append(str(c))
            continue
        ctype = c.get("type")
        if ctype in ("input_text", "output_text", "text"):
            t = c.get("text") or ""
            texts.append(t)
            parts.append({"type": "text", "text": t})
        elif ctype in ("input_image", "image_url", "image"):
            has_image = True
            url = ""
            iu = c.get("image_url")
            if isinstance(iu, str):
                url = iu
            elif isinstance(iu, dict):
                url = iu.get("url") or ""
            if not url:
                src = c.get("source")
                if isinstance(src, dict):
                    url = src.get("url") or ""
                elif isinstance(src, str):
                    url = src
            if not url:
                url = c.get("url") or ""
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            texts.append(dumps(c))
    if has_image:
        return parts
    return "\n".join(texts)


def _normalize_tools(tools: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t:
            out.append(t)
        elif t.get("type") == "function" or "name" in t:
            # Responses flat style: {type, name, description, parameters}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name") or "",
                        "description": t.get("description") or "",
                        "parameters": t.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


# ---------------------------------------------------------------------------
# Response: Chat 鈫?Responses (non-stream)
# ---------------------------------------------------------------------------

def chat_to_responses(chat: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    output: List[Dict[str, Any]] = []

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": new_id("rs"),
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )

    if text or not message.get("tool_calls"):
        output.append(
            {
                "type": "message",
                "id": new_id("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text or ""}],
            }
        )

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments") or "{}"
        if not isinstance(args_raw, str):
            args_raw = dumps(args_raw)
        call_id = tc.get("id") or new_id("call")
        output.append(
            {
                "type": "function_call",
                "id": new_id("fc"),
                "call_id": call_id,
                "name": fn.get("name") or "",
                "arguments": args_raw,
                "status": "completed",
            }
        )

    if not output:
        output.append(
            {
                "type": "message",
                "id": new_id("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            }
        )

    usage = chat.get("usage") or {}
    return {
        "id": new_id("resp"),
        "object": "response",
        "created_at": now_ts(),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": requested_model or chat.get("model") or "",
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
            "total_tokens": usage.get("total_tokens")
            or (
                (usage.get("prompt_tokens") or 0)
                + (usage.get("completion_tokens") or 0)
            ),
        },
        "metadata": {},
        "output_text": text,
        "finish_reason": choice.get("finish_reason"),
    }


# ---------------------------------------------------------------------------
# Response: Chat SSE → Responses SSE
# ---------------------------------------------------------------------------

async def stream_chat_to_responses(
    data_lines: AsyncIterator[str],
    requested_model: str,
) -> AsyncIterator[str]:
    """Convert Chat Completions SSE into Responses SSE events.

    Emits reasoning (if present), assistant text, then function_call items.
    Tool-call argument deltas are forwarded as they arrive from upstream.
    """
    resp_id = new_id("resp")
    created = now_ts()

    def base_response(
        status: str,
        output: Optional[List] = None,
        usage: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "status": status,
            "model": requested_model,
            "output": output or [],
        }
        if usage is not None:
            obj["usage"] = usage
        return obj

    yield sse_event(
        "response.created",
        {"type": "response.created", "response": base_response("in_progress")},
    )
    yield sse_event(
        "response.in_progress",
        {"type": "response.in_progress", "response": base_response("in_progress")},
    )

    full_text: List[str] = []
    reasoning_parts: List[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    # tool_calls: openai index → slot
    tools: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None

    reasoning_id: Optional[str] = None
    reasoning_index: Optional[int] = None
    reasoning_open = False

    item_id: Optional[str] = None
    message_index: Optional[int] = None
    message_open = False
    content_part_open = False

    next_output_index = 0

    def _start_reasoning() -> list[str]:
        nonlocal reasoning_id, reasoning_index, reasoning_open, next_output_index
        frames: list[str] = []
        if reasoning_id is not None:
            return frames
        reasoning_id = new_id("rs")
        reasoning_index = next_output_index
        next_output_index += 1
        reasoning_open = True
        frames.append(
            sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": reasoning_index,
                    "item": {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "summary": [],
                    },
                },
            )
        )
        frames.append(
            sse_event(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": reasoning_id,
                    "output_index": reasoning_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                },
            )
        )
        return frames

    def _close_reasoning() -> list[str]:
        nonlocal reasoning_open
        frames: list[str] = []
        if not reasoning_open or reasoning_id is None or reasoning_index is None:
            return frames
        text = "".join(reasoning_parts)
        frames.append(
            sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": reasoning_id,
                    "output_index": reasoning_index,
                    "summary_index": 0,
                    "text": text,
                },
            )
        )
        frames.append(
            sse_event(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": reasoning_id,
                    "output_index": reasoning_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text},
                },
            )
        )
        frames.append(
            sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": reasoning_index,
                    "item": {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "summary": [{"type": "summary_text", "text": text}],
                    },
                },
            )
        )
        reasoning_open = False
        return frames

    def _start_message() -> list[str]:
        nonlocal item_id, message_index, message_open, content_part_open, next_output_index
        frames: list[str] = []
        if item_id is not None:
            return frames
        frames.extend(_close_reasoning())
        item_id = new_id("msg")
        message_index = next_output_index
        next_output_index += 1
        message_open = True
        content_part_open = True
        frames.append(
            sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": message_index,
                    "item": {
                        "type": "message",
                        "id": item_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
        )
        frames.append(
            sse_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": item_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                },
            )
        )
        return frames

    def _close_message() -> list[str]:
        nonlocal message_open, content_part_open
        frames: list[str] = []
        if item_id is None or message_index is None:
            return frames
        text = "".join(full_text)
        if content_part_open:
            frames.append(
                sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            frames.append(
                sse_event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text},
                    },
                )
            )
            content_part_open = False
        if message_open:
            frames.append(
                sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": message_index,
                        "item": {
                            "type": "message",
                            "id": item_id,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        },
                    },
                )
            )
            message_open = False
        return frames

    async for line in data_lines:
        if line.startswith("__HTTP_ERROR__"):
            yield sse_event(
                "error",
                {"type": "error", "error": {"message": line}},
            )
            return
        if line.strip() == "[DONE]":
            break
        chunk = parse_json(line)
        if not chunk:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            usage = {
                "input_tokens": u.get("prompt_tokens") or 0,
                "output_tokens": u.get("completion_tokens") or 0,
                "total_tokens": u.get("total_tokens") or 0,
            }
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                for frame in _start_reasoning():
                    yield frame
                reasoning_parts.append(reasoning)
                yield sse_event(
                    "response.reasoning_summary_text.delta",
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": reasoning_id,
                        "output_index": reasoning_index,
                        "summary_index": 0,
                        "delta": reasoning,
                    },
                )

            if delta.get("content"):
                for frame in _start_message():
                    yield frame
                t = delta["content"]
                full_text.append(t)
                yield sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "delta": t,
                    },
                )

            for tc in delta.get("tool_calls") or []:
                oi = int(tc.get("index") or 0)
                if oi not in tools:
                    # close open text/reasoning before tool items
                    for frame in _close_message():
                        yield frame
                    for frame in _close_reasoning():
                        yield frame
                    fc_id = new_id("fc")
                    out_idx = next_output_index
                    next_output_index += 1
                    tools[oi] = {
                        "id": tc.get("id") or "",
                        "name": "",
                        "arguments": "",
                        "fc_id": fc_id,
                        "out_idx": out_idx,
                        "started": False,
                    }
                slot = tools[oi]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if not slot["started"] and (slot["name"] or slot["id"] or fn.get("arguments")):
                    slot["started"] = True
                    call_id = slot["id"] or new_id("call")
                    if not slot["id"]:
                        slot["id"] = call_id
                    yield sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": slot["out_idx"],
                            "item": {
                                "id": slot["fc_id"],
                                "type": "function_call",
                                "call_id": call_id,
                                "name": slot["name"],
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                    )
                if fn.get("arguments"):
                    if not slot["started"]:
                        slot["started"] = True
                        call_id = slot["id"] or new_id("call")
                        slot["id"] = call_id
                        yield sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": slot["out_idx"],
                                "item": {
                                    "id": slot["fc_id"],
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": slot["name"],
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            },
                        )
                    slot["arguments"] += fn["arguments"]
                    yield sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": slot["fc_id"],
                            "output_index": slot["out_idx"],
                            "delta": fn["arguments"],
                        },
                    )

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    # Finalize open items
    for frame in _close_message():
        yield frame
    for frame in _close_reasoning():
        yield frame

    # Empty assistant message if nothing was produced (valid shape)
    if item_id is None and not tools and reasoning_id is None:
        for frame in _start_message():
            yield frame
        for frame in _close_message():
            yield frame

    text = "".join(full_text)
    output: List[Dict[str, Any]] = []
    if reasoning_id is not None:
        output.append(
            {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [{"type": "summary_text", "text": "".join(reasoning_parts)}],
            }
        )
    if item_id is not None:
        output.append(
            {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    for oi in sorted(tools):
        slot = tools[oi]
        call_id = slot["id"] or new_id("call")
        fc_item = {
            "id": slot["fc_id"],
            "type": "function_call",
            "call_id": call_id,
            "name": slot["name"],
            "arguments": slot["arguments"],
            "status": "completed",
        }
        if not slot.get("started"):
            yield sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": slot["out_idx"],
                    "item": {
                        "id": slot["fc_id"],
                        "type": "function_call",
                        "call_id": call_id,
                        "name": slot["name"],
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        yield sse_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": slot["fc_id"],
                "output_index": slot["out_idx"],
                "arguments": slot["arguments"],
            },
        )
        yield sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": slot["out_idx"],
                "item": fc_item,
            },
        )
        output.append(fc_item)

    final = base_response("completed", output=output, usage=usage)
    final["output_text"] = text
    if finish_reason:
        final["finish_reason"] = finish_reason
    yield sse_event(
        "response.completed",
        {"type": "response.completed", "response": final},
    )
