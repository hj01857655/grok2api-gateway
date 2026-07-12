from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .admin_routes import router as admin_router
from .config import get_settings
from .request_log_middleware import RequestLogMiddleware
from .routes import router

logger = logging.getLogger("grok2api")


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = FastAPI(
        title="Grok2API",
        description=(
            "Self-built gateway: Chat Completions / Responses / Anthropic Messages. "
            "Upstream is official Grok only (Device Code / xai-*.json). "
            ".env holds process settings (host/port/door key)."
        ),
        version=__version__,
    )
    # Outer CORS, then request log (innermost runs first on the way in).
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(admin_router)

    @app.get("/health")
    async def health():
        s = get_settings()
        info = {
            "ok": True,
            "version": __version__,
            "upstream_mode": s.upstream_mode,
            "effective_upstream_mode": s.effective_upstream_mode(),
            "upstream_key_configured": s.has_official_credential(),
            "oauth_auths_dir": str(s.auths_dir()),
            "client_auth_required": bool(s.grok2api_api_key),
            "admin": "/admin",
            "protocols": [
                "POST /v1/chat/completions",
                "POST /v1/responses",
                "POST /v1/responses/input_tokens",
                "POST /v1/messages",
                "POST /v1/messages/count_tokens",
                "GET  /v1/models",
            ],
            "oauth_login": "python -m app.oauth.login",
            "note": (
                "Official Grok only. Add credentials at /admin (Device Code or import). "
                ".env is process settings only. "
                "UPSTREAM_MODE=auto|oauth|credential."
            ),
        }
        try:
            from .oauth.xai import load_token, resolve_chat_base_url

            ts = load_token(path=s.oauth_token_path(), auths_dir=s.auths_dir())
            if ts:
                info["oauth_email"] = ts.email or ts.sub
                info["oauth_chat_base"] = resolve_chat_base_url(ts)
                info["oauth_using_api"] = ts.using_api
                info["upstream_base_url"] = resolve_chat_base_url(ts)
            else:
                info["oauth_credential"] = (
                    "missing — Device Code: python -m app.oauth.login; "
                    "or import xai-*.json via /admin"
                )
        except Exception as e:
            info["oauth_error"] = str(e)
        return info

    @app.get("/")
    async def root():
        return {
            "name": "grok2api",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "admin": "/admin",
            "oauth_login": "python -m app.oauth.login",
        }

    return app


app = create_app()
