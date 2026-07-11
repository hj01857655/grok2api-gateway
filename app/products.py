"""Product handlers: chat / responses / messages.

Rule: same protocol as upstream → pass-through; only convert on mismatch.

  Mid-station:
    Chat      → POST …/chat/completions  (pass-through)
    Responses → POST …/responses         (pass-through)
    Anthropic → POST …/messages          (pass-through)

  Official token (only /responses):
    Responses → native /responses
    Chat      → convert once Chat↔Responses
    Anthropic → convert once Anthropic↔Chat↔Responses
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Literal, Union

from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .converters import (
    anthropic_to_chat,
    chat_to_anthropic,
    stream_chat_to_anthropic,
)
from .token_count import (
    anthropic_count_response,
    estimate_anthropic_input_tokens,
    estimate_responses_input_tokens,
    responses_input_tokens_response,
)
from .upstream import UpstreamClient, UpstreamError
from .util import iter_sse_data_lines, sse_data

logger = logging.getLogger("grok2api.products")

JSONOrStream = Union[JSONResponse, StreamingResponse]
ErrorStyle = Literal["openai", "anthropic"]


def _client() -> UpstreamClient:
    return UpstreamClient()


def _error_response(
    exc: UpstreamError,
    *,
    style: ErrorStyle = "openai",
) -> JSONResponse:
    """Map upstream failures to client-protocol error envelopes."""
    if style == "anthropic":
        message = exc.body[:2000] if exc.body else "upstream error"
        if isinstance(exc.payload, dict):
            err = exc.payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                message = str(err["message"])[:2000]
            elif exc.payload.get("message"):
                message = str(exc.payload["message"])[:2000]
        return JSONResponse(
            status_code=exc.status,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message,
                },
            },
        )

    if isinstance(exc.payload, dict):
        return JSONResponse(status_code=exc.status, content=exc.payload)
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "message": exc.body[:2000],
                "type": "upstream_error",
                "code": exc.status,
            }
        },
    )


def _sse_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


async def _stream_passthrough(
    raw: AsyncIterator[bytes],
    *,
    style: ErrorStyle = "openai",
) -> AsyncIterator[bytes]:
    async for chunk in raw:
        if chunk.startswith(b"__HTTP_ERROR__"):
            raw_s = chunk.decode("utf-8", errors="replace")
            if style == "anthropic":
                yield sse_data(
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": raw_s,
                        },
                    }
                ).encode("utf-8")
            else:
                yield sse_data(
                    {
                        "type": "error",
                        "error": {"message": raw_s, "type": "upstream_error"},
                    }
                ).encode("utf-8")
            if style == "openai":
                # Chat-style streams often end with [DONE]; Responses may not.
                # Emit for Chat only when caller wants it — keep simple:
                yield b"data: [DONE]\n\n"
            return
        yield chunk


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

async def handle_chat(body: Dict[str, Any]) -> JSONOrStream:
    requested = body.get("model")
    payload = dict(body)
    stream = bool(payload.get("stream"))
    client = _client()

    if not stream:
        try:
            data = await client.chat_completions(payload)
        except UpstreamError as e:
            return _error_response(e, style="openai")
        if requested and data.get("model"):
            data["model"] = requested
        return JSONResponse(content=data)

    return StreamingResponse(
        _stream_passthrough(client.stream_chat_completions(payload)),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Responses API — pass-through both mid and official
# ---------------------------------------------------------------------------

async def handle_responses(body: Dict[str, Any]) -> JSONOrStream:
    """Responses: always same-protocol path.

    Mid-station → POST …/responses (pass-through)
    Official    → POST …/responses (native sanitize)
    """
    requested = body.get("model") or ""
    stream = bool(body.get("stream"))
    client = _client()

    if not stream:
        try:
            data = await client.responses(body)
        except UpstreamError as e:
            return _error_response(e, style="openai")
        if requested and isinstance(data, dict) and data.get("model"):
            data = dict(data)
            data["model"] = requested
        return JSONResponse(content=data)

    async def gen() -> AsyncIterator[bytes]:
        async for chunk in client.stream_responses(body):
            if chunk.startswith(b"__HTTP_ERROR__"):
                raw = chunk.decode("utf-8", errors="replace")
                yield sse_data(
                    {
                        "type": "error",
                        "error": {"message": raw, "type": "upstream_error"},
                    }
                ).encode("utf-8")
                return
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

async def handle_messages(body: Dict[str, Any]) -> JSONOrStream:
    """Anthropic product.

    Mid-station: POST …/messages pass-through (no Chat convert).
    Official: Anthropic → Chat → /responses → Chat → Anthropic (only mismatch).
    """
    requested = body.get("model") or ""
    stream = bool(body.get("stream"))
    client = _client()

    # Mid-station: same protocol, pass-through
    if not client.uses_official_wire():
        if not stream:
            try:
                data = await client.messages(body)
            except UpstreamError as e:
                return _error_response(e, style="anthropic")
            if requested and isinstance(data, dict) and data.get("model"):
                data = dict(data)
                data["model"] = requested
            return JSONResponse(content=data)

        async def gen_mid() -> AsyncIterator[bytes]:
            async for chunk in client.stream_messages(body):
                if chunk.startswith(b"__HTTP_ERROR__"):
                    raw = chunk.decode("utf-8", errors="replace")
                    yield sse_data(
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": raw,
                            },
                        }
                    ).encode("utf-8")
                    return
                yield chunk

        return StreamingResponse(
            gen_mid(),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    # Official: only /responses — convert Anthropic ↔ Chat once
    chat_body = anthropic_to_chat(body)
    display_model = requested or chat_body.get("model") or ""

    if not stream:
        try:
            chat = await client.chat_completions(chat_body)
        except UpstreamError as e:
            return _error_response(e, style="anthropic")
        return JSONResponse(content=chat_to_anthropic(chat, display_model))

    async def gen_official() -> AsyncIterator[str]:
        raw = client.stream_chat_completions(chat_body)
        lines = iter_sse_data_lines(raw)
        async for event in stream_chat_to_anthropic(lines, display_model):
            yield event

    return StreamingResponse(
        gen_official(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Count tokens (local estimate)
# ---------------------------------------------------------------------------

async def handle_messages_count_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    n = estimate_anthropic_input_tokens(body)
    return anthropic_count_response(n)


async def handle_responses_input_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    n = estimate_responses_input_tokens(body)
    return responses_input_tokens_response(n)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

async def handle_models() -> Dict[str, Any]:
    client = _client()
    data = await client.list_models()
    settings = get_settings()
    existing = {m.get("id") for m in data.get("data") or []}
    for alias, target in settings.alias_map().items():
        if alias not in existing:
            data.setdefault("data", []).append(
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "grok2api-alias",
                    "root": target,
                }
            )
    return data
