"""Product handlers: chat / responses / messages.

Upstream is **official Grok only** (wire = POST …/responses).

  Responses → native /responses (sanitize + tools normalize)
  Chat      → convert once Chat↔Responses
  Anthropic → convert once Anthropic↔Responses (no Chat hop)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Literal, Optional, Union

from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .converters import (
    stream_responses_to_anthropic,
)
from .token_count import (
    anthropic_count_response,
    estimate_anthropic_input_tokens,
    estimate_responses_input_tokens,
    responses_input_tokens_response,
)
from .upstream import UpstreamClient, UpstreamError
from .util import iter_sse_data_lines, sse_data

logger = logging.getLogger("grok2api.handlers")

JSONOrStream = Union[JSONResponse, StreamingResponse]
ErrorStyle = Literal["openai", "anthropic"]


# Module-level singleton: one long-lived UpstreamClient serves every request.
# `conv_id` (xAI cache-affinity) is per-request state and is threaded through
# each method call, NOT baked into the instance.
_upstream_client: Optional[UpstreamClient] = None


def _client() -> UpstreamClient:
    """Process-wide UpstreamClient — lazy on first call, reused after."""
    global _upstream_client
    if _upstream_client is None:
        _upstream_client = UpstreamClient()
    return _upstream_client


def reset_upstream_client() -> None:
    """Test hook: drop the cached client so the next _client() rebuilds."""
    global _upstream_client
    _upstream_client = None


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

def _maybe_local_apply_patch(data: Dict[str, Any], *, protocol: str) -> None:
    """Optional local disk apply for apply_patch tool calls."""
    s = get_settings()
    if not s.apply_patch_local:
        return
    from .apply_patch import maybe_local_apply_from_response

    result = maybe_local_apply_from_response(
        data,
        protocol=protocol,  # type: ignore[arg-type]
        root=s.apply_patch_root(),
        enabled=True,
    )
    if result is not None:
        logger.info("apply_patch local: %s", result.as_tool_output())


async def handle_chat(body: Dict[str, Any], *, conv_id: str = "") -> JSONOrStream:
    requested = body.get("model")
    payload = dict(body)
    stream = bool(payload.get("stream"))
    client = _client()

    if not stream:
        try:
            data = await client.chat_completions(payload, conv_id=conv_id)
        except UpstreamError as e:
            return _error_response(e, style="openai")
        if requested and data.get("model"):
            data["model"] = requested
        _maybe_local_apply_patch(data, protocol="chat")
        return JSONResponse(content=data)

    return StreamingResponse(
        _stream_passthrough(client.stream_chat_completions(payload, conv_id=conv_id)),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ---------------------------------------------------------------------------
# Responses API — official /responses only
# ---------------------------------------------------------------------------

async def handle_responses(body: Dict[str, Any], *, conv_id: str = "") -> JSONOrStream:
    """Official Grok: sanitize tools for xAI + optional local apply_patch."""
    requested = body.get("model") or ""
    stream = bool(body.get("stream"))
    client = _client()

    if not stream:
        try:
            data = await client.responses(body, conv_id=conv_id)
        except UpstreamError as e:
            return _error_response(e, style="openai")
        if requested and isinstance(data, dict) and data.get("model"):
            data = dict(data)
            data["model"] = requested
        if isinstance(data, dict):
            _maybe_local_apply_patch(data, protocol="responses")
        return JSONResponse(content=data)

    async def gen() -> AsyncIterator[bytes]:
        async for chunk in client.stream_responses(body, conv_id=conv_id):
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

async def handle_messages(body: Dict[str, Any], *, conv_id: str = "") -> JSONOrStream:
    """Anthropic → /responses → Anthropic (single conversion, no Chat hop)."""
    requested = body.get("model") or ""
    stream = bool(body.get("stream"))
    client = _client()
    display_model = requested

    if not stream:
        try:
            data = await client.messages(body, conv_id=conv_id)
        except UpstreamError as e:
            return _error_response(e, style="anthropic")
        if requested and isinstance(data, dict):
            data = dict(data)
            data["model"] = requested
        return JSONResponse(content=data)

    async def gen_official() -> AsyncIterator[str]:
        raw = client.stream_messages(body, conv_id=conv_id)
        lines = iter_sse_data_lines(raw)
        async for event in stream_responses_to_anthropic(lines, display_model):
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
# Embeddings — OpenAI-compatible passthrough (xAI /v1/embeddings)
# ---------------------------------------------------------------------------

async def handle_embeddings(body: Dict[str, Any]) -> JSONResponse:
    """Passthrough to xAI /v1/embeddings.

    Body/response are already OpenAI-shaped on both sides — no protocol
    conversion, just forward and map upstream errors to the OpenAI envelope.
    """
    client = _client()
    try:
        data = await client.embeddings(body)
    except UpstreamError as e:
        return _error_response(e, style="openai")
    return JSONResponse(content=data)


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
