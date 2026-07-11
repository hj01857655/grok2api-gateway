from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import require_client_auth
from .products import (
    handle_chat,
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
        return await handle_chat(body)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/responses")
async def responses_api(request: Request):
    body = _ensure_json(await request.json())
    try:
        return await handle_responses(body)
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
        return await handle_messages(body)
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
