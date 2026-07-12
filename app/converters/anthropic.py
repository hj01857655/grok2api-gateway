"""Anthropic Messages API ↔ official Grok /responses.

Direct converter — no Chat Completions hop:
  anthropic_to_responses_request  — Anthropic body → /responses payload
  responses_result_to_anthropic   — completed /responses → Anthropic message
  stream_responses_to_anthropic   — /responses SSE → Anthropic Messages SSE
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..util import dumps, new_id, parse_json, sse_event


# ---------------------------------------------------------------------------
# Request: Anthropic → Responses
# ---------------------------------------------------------------------------


def anthropic_to_responses_request(
    body: Dict[str, Any],
    *,
    stream: bool,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Map an Anthropic /v1/messages body to an official /responses payload.

    - system → instructions
    - messages[] → input[] with roles/blocks translated to Responses items
    - tools → flat function items (Responses shape)
    - max_tokens → max_output_tokens
    """
    instructions = _system_to_text(body.get("system"))

    input_items: List[Dict[str, Any]] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").strip()
        content = msg.get("content")
        input_items.extend(_anthropic_message_to_input(role, content))

    if not input_items:
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            }
        ]

    out: Dict[str, Any] = {
        "model": model if model is not None else body.get("model"),
        "input": input_items,
        "stream": stream,
    }
    if instructions:
        out["instructions"] = instructions
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("max_tokens") is not None:
        out["max_output_tokens"] = body["max_tokens"]

    tools = body.get("tools")
    if tools:
        anthropic_tools = [
            _tool_to_responses(t) for t in tools if isinstance(t, dict)
        ]
        from ..apply_patch import normalize_tools_for_xai
        from ..config import get_settings

        s = get_settings()
        if s.apply_patch_normalize:
            normalized, _saw = normalize_tools_for_xai(
                list(anthropic_tools), strip_apply_patch=s.apply_patch_strip
            )
            out["tools"] = normalized
        else:
            out["tools"] = anthropic_tools

    if body.get("tool_choice") is not None and out.get("tools"):
        tc = _tool_choice_to_responses(body["tool_choice"])
        if tc is not None:
            out["tool_choice"] = tc

    return out


def _system_to_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: List[str] = []
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return str(system)


def _anthropic_message_to_input(role: str, content: Any) -> List[Dict[str, Any]]:
    """Translate one Anthropic message into one or more Responses input items."""
    if content is None:
        text_type = "output_text" if role == "assistant" else "input_text"
        return [
            {
                "type": "message",
                "role": role,
                "content": [{"type": text_type, "text": ""}],
            }
        ]

    if isinstance(content, str):
        text_type = "output_text" if role == "assistant" else "input_text"
        return [
            {
                "type": "message",
                "role": role,
                "content": [{"type": text_type, "text": content}],
            }
        ]

    if not isinstance(content, list):
        text_type = "output_text" if role == "assistant" else "input_text"
        return [
            {
                "type": "message",
                "role": role,
                "content": [{"type": text_type, "text": str(content)}],
            }
        ]

    items: List[Dict[str, Any]] = []

    if role == "assistant":
        text_parts: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append({"type": "output_text", "text": str(block)})
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(
                    {"type": "output_text", "text": str(block.get("text") or "")}
                )
            elif btype == "tool_use":
                if text_parts:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": text_parts,
                        }
                    )
                    text_parts = []
                args = block.get("input") or {}
                if not isinstance(args, str):
                    args = dumps(args)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.get("id") or new_id("call"),
                        "name": block.get("name") or "",
                        "arguments": args,
                    }
                )
            elif btype == "thinking":
                continue
            else:
                text_parts.append(
                    {
                        "type": "output_text",
                        "text": str(block.get("text") or dumps(block)),
                    }
                )
        if text_parts:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": text_parts,
                }
            )
        if not items:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
            )
        return items

    # user / any non-assistant role
    user_parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            user_parts.append({"type": "input_text", "text": block})
            continue
        if not isinstance(block, dict):
            user_parts.append({"type": "input_text", "text": str(block)})
            continue
        btype = block.get("type")
        if btype == "text":
            user_parts.append(
                {"type": "input_text", "text": str(block.get("text") or "")}
            )
        elif btype == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64":
                media = src.get("media_type") or "image/png"
                data = src.get("data") or ""
                user_parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media};base64,{data}",
                    }
                )
            elif src.get("type") == "url":
                user_parts.append(
                    {"type": "input_image", "image_url": src.get("url") or ""}
                )
        elif btype == "tool_result":
            if user_parts:
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": user_parts,
                    }
                )
                user_parts = []
            output_text = _content_to_text(block.get("content"))
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id") or "",
                    "output": output_text,
                }
            )
        else:
            user_parts.append(
                {
                    "type": "input_text",
                    "text": str(block.get("text") or dumps(block)),
                }
            )

    if user_parts:
        items.append(
            {
                "type": "message",
                "role": role,
                "content": user_parts,
            }
        )
    if not items:
        items.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": ""}],
            }
        )
    return items


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(dumps(b))
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool_to_responses(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic {name, description, input_schema} → Responses flat function tool."""
    return {
        "type": "function",
        "name": tool.get("name") or "",
        "description": tool.get("description") or "",
        "parameters": tool.get("input_schema")
        or tool.get("parameters")
        or {"type": "object", "properties": {}},
    }


def _tool_choice_to_responses(tc: Any) -> Any:
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
            return {"type": "function", "name": tc["name"]}
    return None


# ---------------------------------------------------------------------------
# Response: /responses (non-stream) → Anthropic message
# ---------------------------------------------------------------------------


def responses_result_to_anthropic(
    resp: Dict[str, Any],
    requested_model: str,
) -> Dict[str, Any]:
    """Map a completed Responses object to Anthropic message JSON."""
    if not isinstance(resp, dict):
        resp = {}
    if resp.get("object") != "response" and isinstance(resp.get("response"), dict):
        resp = resp["response"]

    content_blocks: List[Dict[str, Any]] = []
    text_parts: List[str] = []

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
            thinking_bits: List[str] = []
            for part in item.get("summary") or []:
                if isinstance(part, dict) and part.get("type") in (
                    "summary_text",
                    "reasoning_text",
                    "text",
                ):
                    thinking_bits.append(str(part.get("text") or ""))
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in (
                    "reasoning_text",
                    "summary_text",
                    "text",
                ):
                    thinking_bits.append(str(part.get("text") or ""))
            thinking = "".join(thinking_bits)
            if thinking:
                content_blocks.append({"type": "thinking", "thinking": thinking})
        elif itype == "function_call":
            if text_parts:
                content_blocks.append({"type": "text", "text": "".join(text_parts)})
                text_parts = []
            args_raw = item.get("arguments") or "{}"
            try:
                args = (
                    json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                )
            except Exception:
                args = {"raw": args_raw}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or new_id("toolu"),
                    "name": item.get("name") or "",
                    "input": args if isinstance(args, dict) else {"value": args},
                }
            )

    if text_parts:
        content_blocks.append({"type": "text", "text": "".join(text_parts)})
    elif not content_blocks and resp.get("output_text"):
        content_blocks.append({"type": "text", "text": str(resp["output_text"])})
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    has_tools = any(b.get("type") == "tool_use" for b in content_blocks)
    status = (resp.get("status") or "").lower()
    if has_tools:
        stop_reason = "tool_use"
    elif status == "incomplete":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage_in = resp.get("usage") or {}
    cached_in = int(
        (usage_in.get("input_tokens_details") or {}).get("cached_tokens") or 0
    )
    usage_out: Dict[str, Any] = {
        "input_tokens": int(usage_in.get("input_tokens") or 0),
        "output_tokens": int(usage_in.get("output_tokens") or 0),
    }
    # Anthropic prefix-cache read metric — xAI cached_tokens maps here.
    if cached_in:
        usage_out["cache_read_input_tokens"] = cached_in
    return {
        "id": resp.get("id") or new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": requested_model or resp.get("model") or "",
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


# ---------------------------------------------------------------------------
# Response: /responses SSE → Anthropic SSE
# ---------------------------------------------------------------------------


async def stream_responses_to_anthropic(
    data_lines: AsyncIterator[str],
    requested_model: str,
) -> AsyncIterator[str]:
    """Convert official Responses SSE into Anthropic Messages SSE frames."""
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
    cached_input_tokens = 0
    stop_reason = "end_turn"
    next_idx = 0
    thinking_idx: Optional[int] = None
    text_idx: Optional[int] = None
    tool_map: Dict[str, int] = {}
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
        obj = parse_json(line)
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") or ""

        if etype in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            delta_text = obj.get("delta") or ""
            if not delta_text:
                continue
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
                    "delta": {"type": "thinking_delta", "thinking": delta_text},
                },
            )
            continue

        if etype == "response.output_text.delta":
            delta_text = obj.get("delta") or ""
            if not delta_text:
                continue
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
                    "delta": {"type": "text_delta", "text": delta_text},
                },
            )
            continue

        if etype == "response.output_item.added":
            item = obj.get("item") or {}
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            key = str(item.get("id") or item.get("call_id") or next_idx)
            if key in tool_map:
                continue
            if thinking_idx is not None and thinking_idx in open_blocks:
                frame = stop_block(thinking_idx)
                if frame:
                    yield frame
            if text_idx is not None and text_idx in open_blocks:
                frame = stop_block(text_idx)
                if frame:
                    yield frame
            bi, frame = start_block(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or new_id("toolu"),
                    "name": item.get("name") or "",
                    "input": {},
                }
            )
            tool_map[key] = bi
            if item.get("call_id"):
                tool_map[str(item["call_id"])] = bi
            if item.get("id"):
                tool_map[str(item["id"])] = bi
            yield frame
            continue

        if etype == "response.function_call_arguments.delta":
            item_id = str(obj.get("item_id") or "")
            bi = tool_map.get(item_id)
            if bi is None:
                bi, frame = start_block(
                    {
                        "type": "tool_use",
                        "id": item_id or new_id("toolu"),
                        "name": "",
                        "input": {},
                    }
                )
                tool_map[item_id or f"anon-{bi}"] = bi
                yield frame
            arg_delta = obj.get("delta") or ""
            if arg_delta:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": bi,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arg_delta,
                        },
                    },
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
                input_tokens = int(usage_in.get("input_tokens") or 0)
                output_tokens = int(usage_in.get("output_tokens") or 0)
                cached_input_tokens = int(
                    (usage_in.get("input_tokens_details") or {}).get("cached_tokens")
                    or 0
                )
            has_tools = any(
                isinstance(it, dict) and it.get("type") == "function_call"
                for it in (resp or {}).get("output") or []
            )
            stop_reason = "tool_use" if has_tools else "end_turn"
            status = ((resp or {}).get("status") or "").lower()
            if not has_tools and status == "incomplete":
                stop_reason = "max_tokens"
            continue

        if etype == "error":
            err = obj.get("error")
            msg = err.get("message") if isinstance(err, dict) else str(err or obj)
            yield sse_event(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(msg)},
                },
            )
            return

    for bi in sorted(open_blocks):
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": bi},
        )
    open_blocks.clear()

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

    stream_usage: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cached_input_tokens:
        stream_usage["cache_read_input_tokens"] = cached_input_tokens
    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": stream_usage,
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})
