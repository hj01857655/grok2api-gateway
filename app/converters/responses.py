"""OpenAI Responses converters.

Routing (no double-hop):

  Mid-station (OpenAI-compat /chat/completions):
    Client Responses ↔ Chat  (this file: responses_to_chat / chat_to_responses)

  Official Grok OAuth token (POST …/responses only):
    Client Chat      → chat_to_responses_request → /responses → responses_result_to_chat
    Client Responses → prepare_official_responses_request → /responses (native, no Chat)
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from ..util import dumps, new_id, now_ts, parse_json, sse_data, sse_event


# Fields cli-chat-proxy / xAI often reject (aligned with CPA sanitize).
_OFFICIAL_DROP_KEYS = (
    "previous_response_id",
    "prompt_cache_retention",
    "safety_identifier",
    "stream_options",
)


# ---------------------------------------------------------------------------
# Request: Responses → Chat  (client Responses → mid-station Chat wire)
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
        if ctype in ("input_text", "output_text", "text", "summary_text"):
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
# Request: native Responses for official token wire (NO Chat hop)
# ---------------------------------------------------------------------------

def prepare_official_responses_request(
    body: Dict[str, Any],
    *,
    stream: bool,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Sanitize a client Responses body for official POST …/responses.

    Client already speaks Responses — do not convert through Chat.
    """
    out: Dict[str, Any] = {}
    for key, value in body.items():
        if key in _OFFICIAL_DROP_KEYS:
            continue
        out[key] = value

    if model is not None:
        out["model"] = model
    out["stream"] = stream

    # Chat-style max_tokens alias → Responses max_output_tokens
    if out.get("max_output_tokens") is None and out.get("max_tokens") is not None:
        out["max_output_tokens"] = out["max_tokens"]
    out.pop("max_tokens", None)

    tools = out.get("tools")
    if tools:
        out["tools"] = _responses_tools_for_official(tools)
    if not out.get("tools"):
        out.pop("tools", None)
        out.pop("tool_choice", None)
        out.pop("parallel_tool_calls", None)

    if "input" not in out:
        out["input"] = ""

    return out


def _responses_tools_for_official(tools: List[Any]) -> List[Dict[str, Any]]:
    """Accept flat Responses tools or nested Chat-style function tools."""
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        elif t.get("type") == "function" or "name" in t:
            item: Dict[str, Any] = {
                "type": "function",
                "name": t.get("name") or "",
                "description": t.get("description") or "",
                "parameters": t.get("parameters")
                or {"type": "object", "properties": {}},
            }
            out.append(item)
        else:
            # web_search etc. — pass through
            out.append(dict(t))
    return out


# ---------------------------------------------------------------------------
# Request: Chat → Responses  (client Chat → official token wire)
# ---------------------------------------------------------------------------

def chat_to_responses_request(body: Dict[str, Any], *, stream: bool) -> Dict[str, Any]:
    """Convert Chat Completions request into official xAI /responses body.

    Aligns with CPA prepareResponsesRequest basics: model, stream, input,
    instructions, tools; drops fields cli-chat-proxy rejects.
    """
    instructions_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").strip()
        content = msg.get("content")

        if role in ("system", "developer"):
            text = _plain_text(content)
            if text:
                instructions_parts.append(text)
            continue

        if role == "tool":
            output = content if isinstance(content, str) else dumps(content or "")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or msg.get("id") or "",
                    "output": output,
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        args = dumps(args or {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id") or new_id("call"),
                            "name": fn.get("name") or "",
                            "arguments": args,
                        }
                    )
                text = _plain_text(content)
                if text:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                continue
            input_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": _chat_content_to_responses_parts(
                        content, for_assistant=True
                    ),
                }
            )
            continue

        input_items.append(
            {
                "type": "message",
                "role": "user" if role == "user" else role,
                "content": _chat_content_to_responses_parts(
                    content, for_assistant=False
                ),
            }
        )

    if not input_items:
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            }
        ]

    out: Dict[str, Any] = {
        "model": body.get("model"),
        "input": input_items,
        "stream": stream,
    }
    if instructions_parts:
        out["instructions"] = "\n".join(instructions_parts)
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("max_tokens") is not None:
        out["max_output_tokens"] = body["max_tokens"]

    tools = body.get("tools")
    if tools:
        out["tools"] = _chat_tools_to_responses(tools)
    if body.get("tool_choice") is not None and tools:
        out["tool_choice"] = body["tool_choice"]
    return out


def _plain_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                if c.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(c.get("text") or ""))
                elif "text" in c:
                    parts.append(str(c.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _chat_content_to_responses_parts(
    content: Any, *, for_assistant: bool
) -> List[Dict[str, Any]]:
    text_type = "output_text" if for_assistant else "input_text"
    if content is None:
        return [{"type": text_type, "text": ""}]
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        return [{"type": text_type, "text": str(content)}]

    parts: List[Dict[str, Any]] = []
    for c in content:
        if isinstance(c, str):
            parts.append({"type": text_type, "text": c})
            continue
        if not isinstance(c, dict):
            parts.append({"type": text_type, "text": str(c)})
            continue
        ctype = c.get("type")
        if ctype in ("text", "input_text", "output_text") or (
            "text" in c and ctype is None
        ):
            parts.append({"type": text_type, "text": c.get("text") or ""})
        elif ctype in ("image_url", "input_image", "image"):
            url = ""
            iu = c.get("image_url")
            if isinstance(iu, str):
                url = iu
            elif isinstance(iu, dict):
                url = iu.get("url") or ""
            if not url:
                url = c.get("url") or ""
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append({"type": text_type, "text": dumps(c)})
    return parts or [{"type": text_type, "text": ""}]


def _chat_tools_to_responses(tools: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        elif t.get("type") == "function" or "name" in t:
            out.append(
                {
                    "type": "function",
                    "name": t.get("name") or "",
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
    return out


# ---------------------------------------------------------------------------
# Response: Chat → Responses (non-stream)  — mid-station client product
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
# Response: official Responses result → Chat Completions
# ---------------------------------------------------------------------------

def responses_result_to_chat(
    resp: Dict[str, Any],
    *,
    client_model: str = "",
) -> Dict[str, Any]:
    """Map a completed Responses object to Chat Completions JSON."""
    if not isinstance(resp, dict):
        resp = {}

    if resp.get("object") != "response" and isinstance(resp.get("response"), dict):
        resp = resp["response"]

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    text_parts.append(str(part.get("text") or ""))
        elif itype == "reasoning":
            for part in item.get("summary") or []:
                if isinstance(part, dict) and part.get("type") in (
                    "summary_text",
                    "reasoning_text",
                    "text",
                ):
                    reasoning_parts.append(str(part.get("text") or ""))
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in (
                    "reasoning_text",
                    "summary_text",
                    "text",
                ):
                    reasoning_parts.append(str(part.get("text") or ""))
        elif itype == "function_call":
            args = item.get("arguments") or "{}"
            if not isinstance(args, str):
                args = dumps(args)
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or new_id("call"),
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": args,
                    },
                }
            )

    if not text_parts and resp.get("output_text"):
        text_parts.append(str(resp["output_text"]))

    text = "".join(text_parts)
    reasoning = "".join(reasoning_parts)
    message: Dict[str, Any] = {"role": "assistant", "content": text if text else None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    else:
        if message["content"] is None:
            message["content"] = ""
        finish = "stop"

    status = (resp.get("status") or "").lower()
    if status in ("incomplete", "failed", "cancelled"):
        finish = "length" if status == "incomplete" else status

    usage_in = resp.get("usage") or {}
    usage = {
        "prompt_tokens": int(usage_in.get("input_tokens") or 0),
        "completion_tokens": int(usage_in.get("output_tokens") or 0),
        "total_tokens": int(
            usage_in.get("total_tokens")
            or (
                (usage_in.get("input_tokens") or 0)
                + (usage_in.get("output_tokens") or 0)
            )
        ),
    }

    return {
        "id": resp.get("id") or new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(resp.get("created_at") or now_ts()),
        "model": client_model or resp.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }


async def collect_responses_completed(
    data_lines: AsyncIterator[str],
) -> Dict[str, Any]:
    """Read official Responses SSE until response.completed; return response object."""
    final: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    output_by_index: Dict[int, Dict[str, Any]] = {}
    output_fallback: List[Dict[str, Any]] = []

    async for line in data_lines:
        if line.startswith("__HTTP_ERROR__"):
            raise RuntimeError(line)
        if line.strip() == "[DONE]":
            break
        obj = parse_json(line)
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") or ""
        if etype == "response.output_item.done":
            item = obj.get("item")
            if isinstance(item, dict):
                idx = obj.get("output_index")
                if isinstance(idx, int):
                    output_by_index[idx] = item
                else:
                    output_fallback.append(item)
        elif etype == "response.completed":
            final = (
                obj.get("response")
                if isinstance(obj.get("response"), dict)
                else obj
            )
        elif etype == "error":
            err = obj.get("error")
            if isinstance(err, dict):
                last_error = str(err.get("message") or err)
            else:
                last_error = str(err or obj)

    if final is None:
        raise RuntimeError(
            last_error
            or "official upstream stream ended without response.completed"
        )

    out = final.get("output")
    if (not out) and (output_by_index or output_fallback):
        merged: List[Dict[str, Any]] = []
        for i in sorted(output_by_index):
            merged.append(output_by_index[i])
        merged.extend(output_fallback)
        final = dict(final)
        final["output"] = merged
    return final


async def stream_responses_to_chat(
    data_lines: AsyncIterator[str],
    *,
    client_model: str,
) -> AsyncIterator[bytes]:
    """Convert official Responses SSE into Chat Completions SSE bytes."""
    chat_id = new_id("chatcmpl")
    created = now_ts()
    tool_index = 0
    tool_started: Dict[str, int] = {}
    emitted_role = False
    finish_reason: Optional[str] = None
    usage_chunk: Optional[Dict[str, Any]] = None

    def _chunk(delta: Dict[str, Any], *, finish: Optional[str] = None) -> bytes:
        body = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": client_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
        return sse_data(body).encode("utf-8")

    async for line in data_lines:
        if line.startswith("__HTTP_ERROR__"):
            yield (
                b"__HTTP_ERROR__"
                + line[len("__HTTP_ERROR__") :].encode("utf-8", errors="replace")
            )
            return
        if line.strip() == "[DONE]":
            break
        obj = parse_json(line)
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") or ""

        if etype in ("response.created", "response.in_progress"):
            if not emitted_role:
                emitted_role = True
                yield _chunk({"role": "assistant", "content": ""})
            continue

        if etype in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            delta_text = obj.get("delta") or ""
            if delta_text:
                if not emitted_role:
                    emitted_role = True
                    yield _chunk({"role": "assistant"})
                yield _chunk({"reasoning_content": delta_text})
            continue

        if etype == "response.output_text.delta":
            delta_text = obj.get("delta") or ""
            if delta_text:
                if not emitted_role:
                    emitted_role = True
                    yield _chunk({"role": "assistant", "content": ""})
                yield _chunk({"content": delta_text})
            continue

        if etype == "response.output_item.added":
            item = obj.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = str(item.get("id") or item.get("call_id") or tool_index)
                if key not in tool_started:
                    tool_started[key] = tool_index
                    idx = tool_index
                    tool_index += 1
                    if not emitted_role:
                        emitted_role = True
                        yield _chunk({"role": "assistant"})
                    yield _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": item.get("call_id")
                                    or item.get("id")
                                    or new_id("call"),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name") or "",
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    )
            continue

        if etype == "response.function_call_arguments.delta":
            item_id = str(obj.get("item_id") or "")
            idx = tool_started.get(item_id)
            if idx is None:
                idx = tool_index
                tool_started[item_id or f"anon-{idx}"] = idx
                tool_index += 1
            arg_delta = obj.get("delta") or ""
            if arg_delta:
                yield _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "function": {"arguments": arg_delta},
                            }
                        ]
                    }
                )
            continue

        if etype == "response.completed":
            resp = (
                obj.get("response")
                if isinstance(obj.get("response"), dict)
                else {}
            )
            usage_in = (resp or {}).get("usage") or {}
            if usage_in:
                usage_chunk = {
                    "prompt_tokens": int(usage_in.get("input_tokens") or 0),
                    "completion_tokens": int(usage_in.get("output_tokens") or 0),
                    "total_tokens": int(
                        usage_in.get("total_tokens")
                        or (
                            (usage_in.get("input_tokens") or 0)
                            + (usage_in.get("output_tokens") or 0)
                        )
                    ),
                }
            has_tools = any(
                isinstance(it, dict) and it.get("type") == "function_call"
                for it in (resp or {}).get("output") or []
            )
            finish_reason = "tool_calls" if has_tools else "stop"
            continue

        if etype == "error":
            err = obj.get("error")
            msg = err.get("message") if isinstance(err, dict) else str(err or obj)
            yield sse_data(
                {"error": {"message": msg, "type": "upstream_error"}}
            ).encode("utf-8")
            yield b"data: [DONE]\n\n"
            return

    if not emitted_role:
        yield _chunk({"role": "assistant", "content": ""})

    final_body: Dict[str, Any] = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": client_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage_chunk:
        final_body["usage"] = usage_chunk
    yield sse_data(final_body).encode("utf-8")
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Response: Chat SSE → Responses SSE  — mid-station client product
# ---------------------------------------------------------------------------

async def stream_chat_to_responses(
    data_lines: AsyncIterator[str],
    requested_model: str,
) -> AsyncIterator[str]:
    """Convert Chat Completions SSE into Responses SSE events."""
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
                if not slot["started"] and (
                    slot["name"] or slot["id"] or fn.get("arguments")
                ):
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

    for frame in _close_message():
        yield frame
    for frame in _close_reasoning():
        yield frame

    if item_id is None and not tools and reasoning_id is None:
        for frame in _start_message():
            yield frame
        for frame in _close_message():
            yield frame

    output: List[Dict[str, Any]] = []
    if reasoning_id is not None:
        output.append(
            {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [
                    {"type": "summary_text", "text": "".join(reasoning_parts)}
                ],
            }
        )
    if item_id is not None:
        output.append(
            {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "".join(full_text)}
                ],
            }
        )
    for oi in sorted(tools):
        slot = tools[oi]
        call_id = slot["id"] or new_id("call")
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
                "item": {
                    "id": slot["fc_id"],
                    "type": "function_call",
                    "call_id": call_id,
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                    "status": "completed",
                },
            },
        )
        output.append(
            {
                "id": slot["fc_id"],
                "type": "function_call",
                "call_id": call_id,
                "name": slot["name"],
                "arguments": slot["arguments"],
                "status": "completed",
            }
        )

    status = "completed"
    if finish_reason == "length":
        status = "incomplete"
    completed = base_response(status, output=output, usage=usage)
    yield sse_event(
        "response.completed",
        {"type": "response.completed", "response": completed},
    )
