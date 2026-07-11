"""Product handlers: chat / responses / messages → upstream Chat Completions."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Literal, Union

from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .converters import (
    anthropic_to_chat,
    chat_to_anthropic,
    chat_to_responses,
    responses_to_chat,
    stream_chat_to_anthropic,
    stream_chat_to_responses,
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
        # Prefer Anthropic envelope even when upstream returned OpenAI-shaped JSON.
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


# ---------------------------------------------------------------------------
# Chat Completions (pass-through; upstream resolves custom provider by model)
# ---------------------------------------------------------------------------

async def handle_chat(body: Dict[str, Any]) -> JSONOrStream:
    requested = body.get("model")
    payload = dict(body)
    # Keep client model id; UpstreamClient routes by alias/prefix/provider map.
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

    async def gen() -> AsyncIterator[bytes]:
        async for chunk in client.stream_chat_completions(payload):
            if chunk.startswith(b"__HTTP_ERROR__"):
                raw = chunk.decode("utf-8", errors="replace")
                yield sse_data({"error": {"message": raw}}).encode("utf-8")
                yield b"data: [DONE]\n\n"
                return
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

async def handle_responses(body: Dict[str, Any]) -> JSONOrStream:
    requested = body.get("model") or ""
    chat_body = responses_to_chat(body)
    # Keep client model for multi-provider routing
    stream = bool(body.get("stream") or chat_body.get("stream"))
    client = _client()
    display_model = requested or chat_body.get("model") or ""

    if not stream:
        try:
            chat = await client.chat_completions(chat_body)
        except UpstreamError as e:
            return _error_response(e, style="openai")
        return JSONResponse(content=chat_to_responses(chat, display_model))

    async def gen() -> AsyncIterator[str]:
        raw = client.stream_chat_completions(chat_body)
        lines = iter_sse_data_lines(raw)
        async for event in stream_chat_to_responses(lines, display_model):
            yield event

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

async def handle_messages(body: Dict[str, Any]) -> JSONOrStream:
    requested = body.get("model") or ""
    chat_body = anthropic_to_chat(body)
    stream = bool(body.get("stream") or chat_body.get("stream"))
    client = _client()
    display_model = requested or chat_body.get("model") or ""

    if not stream:
        try:
            chat = await client.chat_completions(chat_body)
        except UpstreamError as e:
            return _error_response(e, style="anthropic")
        return JSONResponse(content=chat_to_anthropic(chat, display_model))

    async def gen() -> AsyncIterator[str]:
        raw = client.stream_chat_completions(chat_body)
        lines = iter_sse_data_lines(raw)
        async for event in stream_chat_to_anthropic(lines, display_model):
            yield event

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Count tokens (local estimate; official response shapes)
# ---------------------------------------------------------------------------

async def handle_messages_count_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic: POST /v1/messages/count_tokens → {input_tokens}."""
    n = estimate_anthropic_input_tokens(body)
    return anthropic_count_response(n)


async def handle_responses_input_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI Responses: POST /v1/responses/input_tokens → {object, input_tokens}."""
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
