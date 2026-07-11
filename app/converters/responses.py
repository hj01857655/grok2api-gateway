"""OpenAI Responses API ↔ OpenAI Chat Completions.

Request:  Responses /v1/responses body → Chat Completions body
Response: Chat Completions (JSON or SSE) → Responses object / SSE events
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..util import dumps, new_id, now_ts, parse_json, sse_event


# ---------------------------------------------------------------------------
# Request: Responses → Chat
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
# Response: Chat → Responses (non-stream)
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
    resp_id = new_id("resp")
    item_id = new_id("msg")
    created = now_ts()

    def base_response(status: str, output: Optional[List] = None, usage: Optional[Dict] = None) -> Dict[str, Any]:
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
    yield sse_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield sse_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )

    full_text: List[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    # tool_calls accumulated: oi → {id, name, arguments}
    tools: Dict[int, Dict[str, str]] = {}
    finish_reason: Optional[str] = None

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
            if delta.get("content"):
                t = delta["content"]
                full_text.append(t)
                yield sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": t,
                    },
                )
            for tc in delta.get("tool_calls") or []:
                oi = int(tc.get("index") or 0)
                slot = tools.setdefault(
                    oi, {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    text = "".join(full_text)
    yield sse_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
    )
    yield sse_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text},
        },
    )
    msg_item = {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    yield sse_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": msg_item,
        },
    )

    output: List[Dict[str, Any]] = [msg_item]
    base_idx = 1
    for oi in sorted(tools):
        slot = tools[oi]
        call_id = slot["id"] or new_id("call")
        fc_id = new_id("fc")
        fc_item = {
            "id": fc_id,
            "type": "function_call",
            "call_id": call_id,
            "name": slot["name"],
            "arguments": slot["arguments"],
            "status": "completed",
        }
        out_idx = base_idx + oi
        yield sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": out_idx,
                "item": {
                    "id": fc_id,
                    "type": "function_call",
                    "call_id": call_id,
                    "name": slot["name"],
                    "arguments": "",
                    "status": "in_progress",
                },
            },
        )
        yield sse_event(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": fc_id,
                "output_index": out_idx,
                "delta": slot["arguments"],
            },
        )
        yield sse_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": fc_id,
                "output_index": out_idx,
                "arguments": slot["arguments"],
            },
        )
        yield sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": out_idx,
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
