from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import require_client_auth
from .handlers import (
    handle_chat,
    handle_embeddings,
    handle_messages,
    handle_messages_count_tokens,
    handle_models,
    handle_responses,
    handle_responses_input_tokens,
)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_client_auth)])


def _ensure_json(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    return body


def _conv_id(request: Request) -> str:
    """xAI cache affinity: propagate client-supplied `x-grok-conv-id`.

    Official docs (docs.x.ai/.../prompt-caching): setting this header keeps a
    conversation pinned to the server that holds its prefix cache. We just
    passthrough — do not generate one, so clients that don't opt in keep the
    stateless behavior.
    """
    return (request.headers.get("x-grok-conv-id") or "").strip()


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    try:
        return await handle_models()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = _ensure_json(await request.json())
    try:
        return await handle_chat(body, conv_id=_conv_id(request))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/responses")
async def responses_api(request: Request):
    body = _ensure_json(await request.json())
    try:
        return await handle_responses(body, conv_id=_conv_id(request))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/responses/input_tokens")
async def responses_input_tokens(request: Request) -> Dict[str, Any]:
    """OpenAI Responses: count input tokens (local estimate)."""
    body = _ensure_json(await request.json())
    try:
        return await handle_responses_input_tokens(body)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/messages")
async def anthropic_messages(request: Request):
    body = _ensure_json(await request.json())
    try:
        return await handle_messages(body, conv_id=_conv_id(request))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/messages/count_tokens")
async def anthropic_count_tokens(request: Request) -> Dict[str, Any]:
    """Anthropic Messages: count input tokens (local estimate)."""
    body = _ensure_json(await request.json())
    try:
        return await handle_messages_count_tokens(body)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/embeddings")
async def embeddings(request: Request):
    """xAI /v1/embeddings — OpenAI-compatible passthrough."""
    body = _ensure_json(await request.json())
    try:
        return await handle_embeddings(body)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
