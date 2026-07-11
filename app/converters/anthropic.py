"""Anthropic Messages API ↔ OpenAI Chat Completions.

Request:  Anthropic /v1/messages body → Chat Completions body
Response: Chat Completions (JSON or SSE) → Anthropic message / SSE events
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..util import dumps, new_id, parse_json, sse_event


# ---------------------------------------------------------------------------
# Request: Anthropic → Chat
# ---------------------------------------------------------------------------

def anthropic_to_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    system = body.get("system")
    if system:
        system_text = _system_to_text(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")
        messages.extend(_message_blocks_to_chat(role, content))

    chat: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages or [{"role": "user", "content": ""}],
        "stream": bool(body.get("stream")),
    }
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if body.get(src) is not None:
            chat[dst] = body[src]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        chat["tools"] = [_tool_to_openai(t) for t in tools if isinstance(t, dict)]
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = _tool_choice_to_openai(body["tool_choice"])
    return chat


def _system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return str(system) if system else ""


def _message_blocks_to_chat(role: str, content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return [{"role": role, "content": ""}]
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": str(content)}]

    tool_results = [
        b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    if tool_results:
        out: List[Dict[str, Any]] = []
        for block in tool_results:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": _content_to_text(block.get("content")),
                }
            )
        rest = [
            b
            for b in content
            if not (isinstance(b, dict) and b.get("type") == "tool_result")
        ]
        if rest:
            out.append({"role": role, "content": _blocks_to_openai_content(rest)})
        return out

    has_tool_use = any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    )
    if has_tool_use and role == "assistant":
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or new_id("toolu"),
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": dumps(block.get("input") or {}),
                        },
                    }
                )
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(t for t in text_parts if t) or None,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    return [{"role": role, "content": _blocks_to_openai_content(content)}]


def _blocks_to_openai_content(blocks: List[Any]) -> Any:
    parts: List[Dict[str, Any]] = []
    text_only: List[str] = []
    has_image = False
    for block in blocks:
        if isinstance(block, str):
            text_only.append(block)
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            s = str(block)
            text_only.append(s)
            parts.append({"type": "text", "text": s})
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text") or ""
            text_only.append(t)
            parts.append({"type": "text", "text": t})
        elif btype == "image":
            has_image = True
            source = block.get("source") or {}
            if source.get("type") == "base64":
                media = source.get("media_type") or "image/png"
                data = source.get("data") or ""
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    }
                )
            elif source.get("type") == "url":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": source.get("url") or ""},
                    }
                )
        elif btype == "thinking":
            continue
        else:
            t = block.get("text") or dumps(block)
            text_only.append(t)
            parts.append({"type": "text", "text": t})
    if has_image:
        return parts
    return "\n".join(t for t in text_only if t is not None)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return str(content)


def _tool_to_openai(tool: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name") or "",
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema")
            or tool.get("parameters")
            or {"type": "object", "properties": {}},
        },
    }


def _tool_choice_to_openai(tc: Any) -> Any:
    if isinstance(tc, str):
        if tc == "any":
            return "required"
        return tc
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "none":
            return "none"
        if t == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
    return "auto"


# ---------------------------------------------------------------------------
# Response: Chat → Anthropic (non-stream)
# ---------------------------------------------------------------------------

def chat_to_anthropic(chat: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_blocks: List[Dict[str, Any]] = []

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content_blocks.insert(0, {"type": "thinking", "thinking": reasoning})

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {"raw": args_raw}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or new_id("toolu"),
                "name": fn.get("name") or "",
                "input": args if isinstance(args, dict) else {"value": args},
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason")
    usage = chat.get("usage") or {}
    return {
        "id": chat.get("id") or new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": requested_model or chat.get("model") or "",
        "content": content_blocks,
        "stop_reason": _finish_to_stop(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


def _finish_to_stop(finish: Optional[str]) -> str:
    return {
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "stop": "end_turn",
        None: "end_turn",
    }.get(finish, finish or "end_turn")


# ---------------------------------------------------------------------------
# Response: Chat SSE → Anthropic SSE
# ---------------------------------------------------------------------------

async def stream_chat_to_anthropic(
    data_lines: AsyncIterator[str],
    requested_model: str,
) -> AsyncIterator[str]:
    msg_id = new_id("msg")
    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield sse_event("ping", {"type": "ping"})

    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    next_idx = 0
    thinking_idx: Optional[int] = None
    text_idx: Optional[int] = None
    tool_map: Dict[int, int] = {}  # openai tool index → block index
    open_blocks: set[int] = set()

    def start_block(block: Dict[str, Any]) -> tuple[int, str]:
        nonlocal next_idx
        bi = next_idx
        next_idx += 1
        open_blocks.add(bi)
        return bi, sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": bi,
                "content_block": block,
            },
        )

    def stop_block(bi: int) -> Optional[str]:
        if bi in open_blocks:
            open_blocks.discard(bi)
            return sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": bi},
            )
        return None

    async for line in data_lines:
        if line.startswith("__HTTP_ERROR__"):
            yield sse_event(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": line}},
            )
            return
        if line.strip() == "[DONE]":
            break
        chunk = parse_json(line)
        if not chunk:
            continue
        if chunk.get("usage"):
            u = chunk["usage"]
            input_tokens = u.get("prompt_tokens") or input_tokens
            output_tokens = u.get("completion_tokens") or output_tokens

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                if thinking_idx is None:
                    thinking_idx, frame = start_block(
                        {"type": "thinking", "thinking": ""}
                    )
                    yield frame
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": thinking_idx,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

            if delta.get("content"):
                if thinking_idx is not None and thinking_idx in open_blocks:
                    frame = stop_block(thinking_idx)
                    if frame:
                        yield frame
                if text_idx is None:
                    text_idx, frame = start_block({"type": "text", "text": ""})
                    yield frame
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_idx,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    },
                )

            for tc in delta.get("tool_calls") or []:
                oi = int(tc.get("index") or 0)
                if oi not in tool_map:
                    if thinking_idx is not None and thinking_idx in open_blocks:
                        frame = stop_block(thinking_idx)
                        if frame:
                            yield frame
                    if text_idx is not None and text_idx in open_blocks:
                        # keep text open until tools start; close when first tool arrives
                        frame = stop_block(text_idx)
                        if frame:
                            yield frame
                    fn = tc.get("function") or {}
                    bi, frame = start_block(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or new_id("toolu"),
                            "name": fn.get("name") or "",
                            "input": {},
                        }
                    )
                    tool_map[oi] = bi
                    yield frame
                bi = tool_map[oi]
                fn = tc.get("function") or {}
                if fn.get("arguments"):
                    yield sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": bi,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": fn["arguments"],
                            },
                        },
                    )

            fr = choice.get("finish_reason")
            if fr:
                stop_reason = _finish_to_stop(fr)

    for bi in sorted(open_blocks):
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": bi},
        )
    open_blocks.clear()

    # if nothing was emitted, still open/close empty text for valid shape
    if next_idx == 0:
        yield sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})
