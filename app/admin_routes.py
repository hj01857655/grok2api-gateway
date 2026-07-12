"""Admin UI + API — official Grok credentials only.

  · Official Grok: Device Code / import xai-*.json → ~/.grok2api/auths
  · .env: process settings only (host/port/door key)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .admin_oauth import get_session, session_public, start_device_session
from .config import get_settings, reload_settings
from .oauth.xai import (
    XAIAuthError,
    import_credential,
    import_credentials,
    list_xai_credentials,
    load_token,
    parse_xai_credential,
    reject_foreign_filename,
    resolve_chat_base_url,
    save_token,
)

router = APIRouter(tags=["admin"])

PROVIDER = "xai"
PROVIDER_LABEL = "Grok / xAI"


def _extract_key(
    authorization: str | None,
    x_api_key: str | None,
    x_admin_key: str | None,
) -> str | None:
    for raw in (x_admin_key, x_api_key):
        if raw and str(raw).strip():
            return str(raw).strip()
    if authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return auth
    return None


async def require_admin(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
) -> None:
    settings = get_settings()
    expected = (settings.grok2api_api_key or "").strip()
    if not expected:
        return
    got = _extract_key(authorization, x_api_key, x_admin_key)
    if not got or got != expected:
        raise HTTPException(status_code=401, detail="admin auth required")


def _home() -> Path:
    return get_settings().home_dir()


def _credential_status() -> Dict[str, Any]:
    """Status for admin UI — official Grok credentials only."""
    s = get_settings()
    auths = s.auths_dir()
    ts = load_token(path=s.oauth_token_path(), auths_dir=auths)
    files = list_xai_credentials(auths)

    current: Optional[Dict[str, Any]] = None
    chat_base = ""
    if ts:
        chat_base = resolve_chat_base_url(ts)
        current = {
            "provider": PROVIDER,
            "provider_label": PROVIDER_LABEL,
            "email": ts.email or None,
            "sub": ts.sub or None,
            "expired": ts.expired or None,
            "using_api": ts.using_api,
            "base_url": ts.base_url,
            "chat_base": chat_base,
            "has_access": bool(ts.access_token),
            "has_refresh": bool(ts.refresh_token),
            "auth_kind": ts.auth_kind,
            "type": "xai",
        }

    return {
        "ok": True,
        "version": __version__,
        "provider": PROVIDER,
        "provider_label": PROVIDER_LABEL,
        "upstream_mode": s.upstream_mode,
        "effective_upstream_mode": s.effective_upstream_mode(),
        "upstream_base_url": chat_base,
        "upstream_key_configured": bool(ts and ts.access_token),
        "oauth_auths_dir": str(auths),
        "oauth_current": current,
        "oauth_files": files,
        "admin_auth_required": bool(s.grok2api_api_key),
        "notes": [
            "Official Grok only — add credentials here (Device Code or import), not from .env.",
            f"Credentials store → {auths}",
            ".env only: HOST / PORT / GROK2API_API_KEY / UPSTREAM_MODE / timeouts.",
            "UPSTREAM_MODE=auto|oauth|credential.",
        ],
    }


class ImportBody(BaseModel):
    json_text: str = Field(default="", description="Grok/xAI credential JSON only (type=xai)")
    using_api: Optional[bool] = None


class DeviceStartBody(BaseModel):
    using_api: bool = False


class SelectBody(BaseModel):
    name: str = Field(..., description="Filename under auths dir, e.g. xai-user@x.ai.json")
    using_api: Optional[bool] = None


class UsingApiBody(BaseModel):
    using_api: bool = True


_STATIC_DIR = Path(__file__).parent / "static"
_ADMIN_DIST = _STATIC_DIR / "admin-dist"
_LEGACY_ADMIN = _STATIC_DIR / "admin.html"


def _spa_index() -> Path | None:
    idx = _ADMIN_DIST / "index.html"
    return idx if idx.is_file() else None


@router.get("/admin", include_in_schema=False)
@router.get("/admin/", include_in_schema=False)
async def admin_page():
    """Serve React SPA when built; fall back to legacy admin.html."""
    spa = _spa_index()
    if spa is not None:
        return FileResponse(spa, media_type="text/html")
    if _LEGACY_ADMIN.is_file():
        return HTMLResponse(_LEGACY_ADMIN.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Admin UI not built</h1><p>Run <code>npm run build</code> in <code>admin-ui/</code>.</p>",
        status_code=503,
    )


@router.get("/admin/assets/{asset_path:path}", include_in_schema=False)
async def admin_assets(asset_path: str):
    """Vite build assets under /admin/assets/*."""
    base = (_ADMIN_DIST / "assets").resolve()
    target = (base / asset_path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(target)


@router.get("/admin/api/status")
async def admin_status(_: None = Depends(require_admin)) -> Dict[str, Any]:
    return _credential_status()


@router.get("/admin/api/logs")
async def admin_logs(
    _: None = Depends(require_admin),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    path_prefix: Optional[str] = Query(default=None),
    status_min: Optional[int] = Query(default=None),
    status_max: Optional[int] = Query(default=None),
    since: Optional[str] = Query(default=None, description="ISO timestamp lower bound"),
) -> Dict[str, Any]:
    from .request_log import get_request_log_store

    store = get_request_log_store()
    return store.query(
        limit=limit,
        offset=offset,
        path_prefix=path_prefix,
        status_min=status_min,
        status_max=status_max,
        since=since,
    )


@router.get("/admin/api/logs/summary")
async def admin_logs_summary(_: None = Depends(require_admin)) -> Dict[str, Any]:
    from .request_log import get_request_log_store

    return get_request_log_store().summary()


@router.get("/admin/api/models")
async def admin_models(_: None = Depends(require_admin)) -> Dict[str, Any]:
    from .handlers import handle_models

    try:
        data = await handle_models()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"models upstream failed: {e}") from e
    return {"ok": True, "models": data}


# ── Official Grok credentials ──────────────────────────────────────────────


@router.post("/admin/api/import")
async def admin_import(
    body: ImportBody,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """Paste-import Grok/xAI credential only. Rejects other providers."""
    raw = (body.json_text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="json_text is required")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
    s = get_settings()
    try:
        storage = parse_xai_credential(data, source_hint="admin-paste")
        if body.using_api is not None:
            storage.using_api = body.using_api
        path = save_token(storage, s.auths_dir())
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "provider": PROVIDER,
        "path": str(path),
        "email": storage.email or storage.sub,
        "using_api": storage.using_api,
        "chat_base": resolve_chat_base_url(storage),
        "status": _credential_status(),
    }


@router.post("/admin/api/import/file")
async def admin_import_file(
    file: UploadFile = File(...),
    using_api: Optional[str] = Form(default=None),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """Upload-import Grok/xAI credential only. Filename + type both gated."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    fname = file.filename or "upload.json"
    try:
        reject_foreign_filename(fname)
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON file: {e}") from e
    s = get_settings()
    try:
        storage = parse_xai_credential(data, source_hint=fname)
        if using_api is not None and str(using_api).strip().lower() in ("1", "true", "yes", "on"):
            storage.using_api = True
        elif using_api is not None and str(using_api).strip().lower() in ("0", "false", "no", "off"):
            storage.using_api = False
        path = save_token(storage, s.auths_dir())
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "provider": PROVIDER,
        "path": str(path),
        "email": storage.email or storage.sub,
        "using_api": storage.using_api,
        "chat_base": resolve_chat_base_url(storage),
        "status": _credential_status(),
    }


@router.post("/admin/api/import/path")
async def admin_import_path(
    path: str = Form(...),
    using_api: Optional[str] = Form(default=None),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """Import from server-side path (Grok/xAI only)."""
    s = get_settings()
    ua: Optional[bool] = None
    if using_api is not None:
        ua = str(using_api).strip().lower() in ("1", "true", "yes", "on")
    try:
        paths = import_credentials(
            path,
            auths_dir=s.auths_dir(),
            using_api=ua,
            set_current=True,
        )
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "provider": PROVIDER,
        "imported": [str(p) for p in paths],
        "status": _credential_status(),
    }


@router.post("/admin/api/select")
async def admin_select(
    body: SelectBody,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    s = get_settings()
    name = Path(body.name).name
    if not name.startswith("xai-") or not name.endswith(".json") or name == "xai-current.json":
        raise HTTPException(
            status_code=400,
            detail="name must be an xai-*.json Grok credential file",
        )
    try:
        reject_foreign_filename(name)
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    src = s.auths_dir() / name
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {name}")
    try:
        path = import_credential(
            src,
            auths_dir=s.auths_dir(),
            using_api=body.using_api,
            set_current=True,
        )
    except XAIAuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    ts = load_token(auths_dir=s.auths_dir())
    return {
        "ok": True,
        "provider": PROVIDER,
        "path": str(path),
        "email": (ts.email if ts else None),
        "status": _credential_status(),
    }


@router.post("/admin/api/using-api")
async def admin_set_using_api(
    body: UsingApiBody,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    s = get_settings()
    ts = load_token(path=s.oauth_token_path(), auths_dir=s.auths_dir())
    if not ts:
        raise HTTPException(status_code=404, detail="no current Grok OAuth credential")
    ts.using_api = body.using_api
    path = save_token(ts, s.auths_dir())
    return {
        "ok": True,
        "provider": PROVIDER,
        "path": str(path),
        "using_api": ts.using_api,
        "chat_base": resolve_chat_base_url(ts),
        "status": _credential_status(),
    }


@router.post("/admin/api/oauth/device/start")
async def admin_device_start(
    body: DeviceStartBody | None = None,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """Start Grok/xAI Device Code login (only vendor supported)."""
    body = body or DeviceStartBody()
    s = get_settings()
    try:
        sess = start_device_session(auths_dir=s.auths_dir(), using_api=body.using_api)
    except XAIAuthError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"device start failed: {e}") from e
    return {"ok": True, "provider": PROVIDER, **session_public(sess)}


@router.get("/admin/api/oauth/device/{session_id}")
async def admin_device_status(
    session_id: str,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found or expired")
    out = session_public(sess)
    out["provider"] = PROVIDER
    if sess.status == "success":
        out["status_full"] = _credential_status()
    return out


@router.post("/admin/api/reload-settings")
async def admin_reload(_: None = Depends(require_admin)) -> Dict[str, Any]:
    reload_settings()
    return {"ok": True, "status": _credential_status()}


# SPA client routes last — must not register before /admin/api/*
@router.get("/admin/{spa_path:path}", include_in_schema=False)
async def admin_spa_fallback(spa_path: str):
    """SPA client routes — never shadow /admin/api/*."""
    if spa_path == "api" or spa_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="not found")
    candidate = (_ADMIN_DIST / spa_path).resolve()
    dist_root = _ADMIN_DIST.resolve()
    if (
        candidate.is_file()
        and str(candidate).startswith(str(dist_root))
        and spa_path != "index.html"
    ):
        return FileResponse(candidate)
    spa = _spa_index()
    if spa is not None:
        return FileResponse(spa, media_type="text/html")
    raise HTTPException(status_code=404, detail="admin spa not built")
