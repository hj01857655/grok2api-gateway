from __future__ import annotations

from fastapi import Header, HTTPException, Request

from .config import get_settings


def _extract_client_key(
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return auth
    return None


async def require_client_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
    """If GROK2API_API_KEY is set, require Bearer or x-api-key match."""
    settings = get_settings()
    expected = (settings.grok2api_api_key or "").strip()
    if not expected:
        return
    got = _extract_client_key(authorization, x_api_key)
    if not got or got != expected:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid or missing API key. Use Authorization: Bearer <key> or x-api-key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )
