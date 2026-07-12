"""Starlette middleware: append request metadata after response.

Skips SPA static assets; always records /v1/* and /admin/api/*.
Body is never fully stored unless REQUEST_LOG_BODY_MAX > 0 (preview only).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .request_log import record_request, should_skip_path

logger = logging.getLogger("grok2api.request_log")


def _extract_model_and_stream(body: bytes, content_type: str) -> tuple[Optional[str], Optional[bool]]:
    if not body or "json" not in (content_type or "").lower():
        return None, None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    model = data.get("model")
    if model is not None:
        model = str(model)
    stream = data.get("stream")
    if stream is not None:
        stream = bool(stream)
    return model, stream


def _body_preview(body: bytes, max_len: int) -> Optional[str]:
    if max_len <= 0 or not body:
        return None
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return None
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


class RequestLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if should_skip_path(path):
            return await call_next(request)

        from .config import get_settings

        settings = get_settings()
        if not settings.request_log_enabled:
            return await call_next(request)

        model: Optional[str] = None
        stream: Optional[bool] = None
        preview: Optional[str] = None
        body_max = settings.request_log_body_max

        # Shallow parse JSON body for model/stream (buffered once by Starlette)
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                ct = request.headers.get("content-type") or ""
                model, stream = _extract_model_and_stream(body, ct)
                preview = _body_preview(body, body_max)
            except Exception:
                pass

        t0 = time.perf_counter()
        status = 500
        err: Optional[str] = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as exc:
            err = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            client = None
            if request.client:
                client = request.client.host
            try:
                record_request(
                    method=request.method,
                    path=path,
                    status=status,
                    duration_ms=duration_ms,
                    model=model,
                    stream=stream,
                    error=err,
                    client=client,
                    body_preview=preview,
                )
            except Exception as log_exc:
                logger.debug("request log append failed: %s", log_exc)
