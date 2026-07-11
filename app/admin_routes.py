"""Admin UI + API — Grok credentials AND mid-station channels.

Both kinds of upstream inventory are managed here:
  · Official Grok: Device Code / import xai-*.json → ~/.grok2api/auths
  · Mid-station channels: base URL + key + models → ~/.grok2api/providers.json

.env only holds gateway process settings (host/port/door key), not accounts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from . import channel_store
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
    """Status for admin UI — Grok credentials + managed channels."""
    s = get_settings()
    auths = s.auths_dir()
    ts = load_token(path=s.oauth_token_path(), auths_dir=auths)
    files = list_xai_credentials(auths)
    channels = channel_store.list_public(s.home_dir())

    current: Optional[Dict[str, Any]] = None
    if ts:
        current = {
            "provider": PROVIDER,
            "provider_label": PROVIDER_LABEL,
            "email": ts.email or None,
            "sub": ts.sub or None,
            "expired": ts.expired or None,
            "using_api": ts.using_api,
            "base_url": ts.base_url,
            "chat_base": resolve_chat_base_url(ts),
            "has_access": bool(ts.access_token),
            "has_refresh": bool(ts.refresh_token),
            "auth_kind": ts.auth_kind,
            "type": "xai",
        }

    first_base = channels[0]["base_url"] if channels else ""
    return {
        "ok": True,
        "version": __version__,
        "provider": PROVIDER,
        "provider_label": PROVIDER_LABEL,
        "upstream_mode": s.upstream_mode,
        "effective_upstream_mode": s.effective_upstream_mode(),
        "upstream_base_url": first_base,
        "upstream_key_configured": any(c.get("key_configured") for c in channels)
        or bool(ts and ts.access_token),
        "channels": channels,
        "custom_providers": channels,  # alias for older UI
        "providers_store": str(s.providers_store_path()),
        "oauth_auths_dir": str(auths),
        "oauth_current": current,
        "oauth_files": files,
        "admin_auth_required": bool(s.grok2api_api_key),
        "notes": [
            "Accounts and channels exist only after you add them here — not from .env.",
            f"Mid-station channels → {s.providers_store_path()}",
            f"Official Grok credentials → {auths}",
            ".env only: HOST / PORT / GROK2API_API_KEY / UPSTREAM_MODE / timeouts.",
            "UPSTREAM_MODE=auto|compat|oauth|credential "
            "(auto = official if present, else managed channels).",
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


class ChannelBody(BaseModel):
    name: str = Field(..., description="Channel display name, e.g. iamhc")
    base_url: str = Field(..., description="OpenAI-compatible base URL ending with /v1")
    api_key: str = Field(..., description="Upstream API key for this channel")
    models: Union[str, List[Any], None] = Field(
        default=None,
        description="Comma-separated model ids, or list of names / {name,alias}",
    )
    prefix: str = Field(default="", description="Optional route prefix pin")


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_page() -> HTMLResponse:
    html_path = Path(__file__).parent / "static" / "admin.html"
    if not html_path.is_file():
        return HTMLResponse("<h1>admin.html missing</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/admin/api/status")
async def admin_status(_: None = Depends(require_admin)) -> Dict[str, Any]:
    return _credential_status()


# ── Mid-station channels ───────────────────────────────────────────────────


@router.get("/admin/api/channels")
async def list_channels(_: None = Depends(require_admin)) -> Dict[str, Any]:
    s = get_settings()
    return {
        "ok": True,
        "store": str(s.providers_store_path()),
        "channels": channel_store.list_public(s.home_dir()),
    }


@router.post("/admin/api/channels")
async def add_channel(
    body: ChannelBody,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    s = get_settings()
    try:
        created = channel_store.add_provider(
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            models=body.models,
            prefix=body.prefix,
            root=s.home_dir(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    reload_settings()
    return {"ok": True, "channel": created, "status": _credential_status()}


@router.delete("/admin/api/channels/{channel_id}")
async def delete_channel(
    channel_id: str,
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    s = get_settings()
    ok = channel_store.delete_provider(channel_id, root=s.home_dir())
    if not ok:
        raise HTTPException(status_code=404, detail=f"channel not found: {channel_id}")
    reload_settings()
    return {"ok": True, "deleted": channel_id, "status": _credential_status()}


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
