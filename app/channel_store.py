"""Managed mid-station / custom OpenAI-compatible channels.

Channels exist only after being added via admin (or API).
Persisted at: ~/.grok2api/providers.json

This is intentionally separate from .env — env is for gateway process
settings only (host/port/door key), not upstream inventory.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .providers import CompatModel, CompatProvider, providers_summary

logger = logging.getLogger("grok2api.channel_store")

_DEFAULT_ROOT = Path.home() / ".grok2api"
_STORE_NAME = "providers.json"


def data_dir(root: Optional[Path] = None) -> Path:
    base = root or _DEFAULT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def providers_path(root: Optional[Path] = None) -> Path:
    return data_dir(root) / _STORE_NAME


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    return s or f"channel-{uuid.uuid4().hex[:8]}"


def _parse_models(raw: Any) -> List[CompatModel]:
    out: List[CompatModel] = []
    if not raw:
        return out
    if isinstance(raw, str):
        for part in raw.split(","):
            m = part.strip()
            if m:
                out.append(CompatModel(name=m, alias=m))
        return out
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(CompatModel(name=item.strip(), alias=item.strip()))
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or "").strip()
            if not name:
                continue
            alias = str(item.get("alias") or item.get("client") or name).strip() or name
            out.append(CompatModel(name=name, alias=alias))
    return out


def _row_to_provider(row: Dict[str, Any]) -> Optional[CompatProvider]:
    if not isinstance(row, dict) or row.get("disabled"):
        return None
    base = str(row.get("base_url") or "").strip()
    if not base:
        return None
    name = str(row.get("name") or row.get("id") or "").strip() or _slug("channel")
    return CompatProvider(
        name=name,
        base_url=base,
        api_key=str(row.get("api_key") or "").strip(),
        prefix=str(row.get("prefix") or name).strip(),
        models=_parse_models(row.get("models")),
        disabled=False,
    )


def load_raw(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = providers_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("providers store unreadable %s: %s", path, exc)
        return []
    if isinstance(data, dict):
        data = data.get("providers") or data.get("channels") or []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def save_raw(rows: List[Dict[str, Any]], root: Optional[Path] = None) -> Path:
    path = providers_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"providers": rows}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_providers(root: Optional[Path] = None) -> List[CompatProvider]:
    """Active mid-station channels only — never invents defaults from env."""
    out: List[CompatProvider] = []
    for row in load_raw(root):
        p = _row_to_provider(row)
        if p and p.api_key:
            out.append(p)
        elif p and not p.api_key:
            logger.warning("channel %s has no api_key — skipped for routing", p.name)
    return out


def list_public(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Admin-facing list (masks secrets)."""
    public = []
    for row in load_raw(root):
        key = str(row.get("api_key") or "")
        public.append(
            {
                "id": str(row.get("id") or row.get("name") or ""),
                "name": str(row.get("name") or ""),
                "prefix": str(row.get("prefix") or row.get("name") or "") or None,
                "base_url": str(row.get("base_url") or ""),
                "key_configured": bool(key),
                "key_hint": (key[:4] + "…" + key[-4:]) if len(key) >= 12 else ("***" if key else ""),
                "models": [
                    {"name": m.upstream_id(), "alias": m.client_id()}
                    for m in _parse_models(row.get("models"))
                ],
                "disabled": bool(row.get("disabled")),
            }
        )
    return public


def add_provider(
    *,
    name: str,
    base_url: str,
    api_key: str,
    models: Any = None,
    prefix: str = "",
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    name = (name or "").strip()
    base_url = (base_url or "").strip().rstrip("/")
    api_key = (api_key or "").strip()
    if not name:
        raise ValueError("name is required")
    if not base_url:
        raise ValueError("base_url is required")
    if not api_key:
        raise ValueError("api_key is required")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise ValueError("base_url must start with http:// or https://")

    rows = load_raw(root)
    pid = _slug(name)
    # unique id
    existing_ids = {str(r.get("id") or r.get("name") or "") for r in rows}
    if pid in existing_ids:
        pid = f"{pid}-{uuid.uuid4().hex[:6]}"

    model_list = _parse_models(models)
    row = {
        "id": pid,
        "name": name,
        "prefix": (prefix or name).strip(),
        "base_url": base_url,
        "api_key": api_key,
        "models": [
            {"name": m.upstream_id(), "alias": m.client_id()} for m in model_list
        ],
        "disabled": False,
    }
    rows.append(row)
    save_raw(rows, root)
    return {
        "id": pid,
        "name": name,
        "base_url": base_url,
        "prefix": row["prefix"],
        "models": row["models"],
        "key_configured": True,
    }


def delete_provider(provider_id: str, root: Optional[Path] = None) -> bool:
    pid = (provider_id or "").strip()
    if not pid:
        return False
    rows = load_raw(root)
    new_rows = [
        r
        for r in rows
        if str(r.get("id") or "") != pid and str(r.get("name") or "") != pid
    ]
    if len(new_rows) == len(rows):
        return False
    save_raw(new_rows, root)
    return True


def summary_for_settings(root: Optional[Path] = None) -> List[dict]:
    return providers_summary(load_providers(root))
