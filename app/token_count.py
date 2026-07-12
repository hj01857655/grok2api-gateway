"""Local input-token estimation for count endpoints.

Upstream (iamhc / custom OpenAI-compat) typically has no official count HTTP.
We estimate from request text after converting to Chat Completions shape,
returning Anthropic / OpenAI Responses official response shapes.

Heuristic (no tiktoken dependency):
- CJK / full-width: ~1.5 chars per token
- Other scripts: ~4 chars per token
- Empty → 0
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Roughly: Latin 4 chars/token; CJK denser.
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)


def estimate_tokens(text: str) -> int:
    """Estimate token count for plain text."""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    # CJK ≈ 1 token / 1.5 chars; Latin ≈ 1 / 4
    n = int(cjk / 1.5 + other / 4.0 + 0.999)  # ceil-ish
    return max(n, 1) if text.strip() else 0


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("type")
                if t in ("text", "input_text", "output_text") and part.get("text"):
                    parts.append(str(part["text"]))
                elif t == "image_url":
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        parts.append(str(url.get("url") or ""))
                    elif url:
                        parts.append(str(url))
                    else:
                        parts.append("[image]")
                elif t in ("image", "input_image"):
                    parts.append("[image]")
                elif part.get("text"):
                    parts.append(str(part["text"]))
                else:
                    parts.append(_stringify(part))
            else:
                parts.append(_stringify(part))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"])
        return _stringify(content)
    return str(content)


def chat_body_to_text(chat: Dict[str, Any]) -> str:
    """Flatten a Chat Completions request body into countable text."""
    segments: List[str] = []

    for msg in chat.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or ""
        if role:
            segments.append(role)
        content = msg.get("content")
        text = _content_to_text(content)
        if text:
            segments.append(text)
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if fn.get("name"):
                segments.append(str(fn["name"]))
            if fn.get("arguments") is not None:
                segments.append(_stringify(fn["arguments"]))
        if msg.get("name"):
            segments.append(str(msg["name"]))
        if msg.get("tool_call_id"):
            segments.append(str(msg["tool_call_id"]))

    for tool in chat.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(fn, dict):
            continue
        if fn.get("name"):
            segments.append(str(fn["name"]))
        if fn.get("description"):
            segments.append(str(fn["description"]))
        if fn.get("parameters") is not None:
            segments.append(_stringify(fn["parameters"]))
        elif tool.get("parameters") is not None:
            segments.append(_stringify(tool["parameters"]))

    if chat.get("tool_choice") is not None and not isinstance(
        chat.get("tool_choice"), (str, type(None))
    ):
        segments.append(_stringify(chat["tool_choice"]))

    return "\n".join(s for s in segments if s)


def estimate_chat_tokens(chat: Dict[str, Any]) -> int:
    text = chat_body_to_text(chat)
    n = estimate_tokens(text)
    # Per-message framing overhead (~4 tokens each), similar to OpenAI docs.
    msg_count = len(chat.get("messages") or [])
    if msg_count:
        n += 4 * msg_count
    if chat.get("tools"):
        n += 8  # tool schema framing
    return n


def _anthropic_body_to_text(body: Dict[str, Any]) -> str:
    """Flatten an Anthropic Messages body into countable text."""
    segments: List[str] = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            segments.append(system)
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, dict) and b.get("type") == "text":
                    if b.get("text"):
                        segments.append(str(b["text"]))
                elif isinstance(b, str):
                    segments.append(b)

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or ""
        if role:
            segments.append(role)
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            segments.append(content)
            continue
        if not isinstance(content, list):
            segments.append(_stringify(content))
            continue
        for block in content:
            if isinstance(block, str):
                segments.append(block)
                continue
            if not isinstance(block, dict):
                segments.append(_stringify(block))
                continue
            btype = block.get("type")
            if btype == "text":
                if block.get("text"):
                    segments.append(str(block["text"]))
            elif btype == "image":
                segments.append("[image]")
            elif btype == "tool_use":
                if block.get("name"):
                    segments.append(str(block["name"]))
                if block.get("input") is not None:
                    segments.append(_stringify(block["input"]))
            elif btype == "tool_result":
                segments.append(_content_to_text(block.get("content")))
            elif btype == "thinking":
                continue
            elif block.get("text"):
                segments.append(str(block["text"]))
            else:
                segments.append(_stringify(block))

    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name"):
            segments.append(str(tool["name"]))
        if tool.get("description"):
            segments.append(str(tool["description"]))
        schema = tool.get("input_schema") or tool.get("parameters")
        if schema is not None:
            segments.append(_stringify(schema))

    tc = body.get("tool_choice")
    if tc is not None and not isinstance(tc, (str, type(None))):
        segments.append(_stringify(tc))

    return "\n".join(s for s in segments if s)


def estimate_anthropic_input_tokens(body: Dict[str, Any]) -> int:
    """Estimate Anthropic Messages count_tokens body → input_tokens."""
    text = _anthropic_body_to_text(body)
    n = estimate_tokens(text)
    msg_count = len(body.get("messages") or [])
    if msg_count:
        n += 4 * msg_count
    if body.get("tools"):
        n += 8
    return n


def estimate_responses_input_tokens(body: Dict[str, Any]) -> int:
    """Estimate OpenAI Responses input_tokens body."""
    from .converters.responses import responses_to_chat

    chat = responses_to_chat(body)
    chat.pop("stream", None)
    chat.pop("max_tokens", None)
    return estimate_chat_tokens(chat)


def anthropic_count_response(input_tokens: int) -> Dict[str, Any]:
    """Official Anthropic shape: POST /v1/messages/count_tokens."""
    return {"input_tokens": int(input_tokens)}


def responses_input_tokens_response(input_tokens: int) -> Dict[str, Any]:
    """Official OpenAI shape: POST /v1/responses/input_tokens."""
    return {
        "object": "response.input_tokens",
        "input_tokens": int(input_tokens),
    }
