"""OpenAI-compatible multi-provider routing (CPA openai-compatibility style).

Like CLIProxyAPI: you can use **custom OpenAI-compatible APIs** (iamhc, OpenRouter,
NewAPI, …) **and/or** official Grok OAuth — custom does NOT require a Grok account.

Config sources:
1. OPENAI_COMPATIBILITY — path to JSON file, or inline JSON array
2. Legacy single upstream: XAI_BASE_URL + XAI_API_KEY / VOYA_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("grok2api.providers")

# Project root (…/grok2api) for relative OPENAI_COMPATIBILITY paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CompatModel:
    """One model exposed by a custom provider."""

    name: str  # upstream model id sent to the API
    alias: str = ""  # client-visible id (default = name)

    def client_id(self) -> str:
        return (self.alias or self.name).strip()

    def upstream_id(self) -> str:
        return self.name.strip()


@dataclass
class CompatProvider:
    """One OpenAI-compatible upstream (iamhc, OpenRouter, NewAPI, …)."""

    name: str
    base_url: str
    api_key: str = ""
    prefix: str = ""  # optional: require "prefix/model" to pin this provider
    models: List[CompatModel] = field(default_factory=list)
    disabled: bool = False

    def display_name(self) -> str:
        return (self.name or "default").strip() or "default"

    def normalized_base(self) -> str:
        return self.base_url.rstrip("/")


@dataclass
class UpstreamRoute:
    """Resolved target for one request."""

    provider: str
    base_url: str
    api_key: str
    model: str  # upstream model id
    client_model: str  # original client-visible id


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in ("1", "true", "yes", "on")


def _parse_model_entry(raw: Any) -> Optional[CompatModel]:
    if isinstance(raw, str):
        name = raw.strip()
        return CompatModel(name=name, alias=name) if name else None
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("id") or "").strip()
    if not name:
        return None
    alias = str(raw.get("alias") or raw.get("client") or name).strip() or name
    return CompatModel(name=name, alias=alias)


def _parse_provider(raw: Dict[str, Any]) -> Optional[CompatProvider]:
    name = str(raw.get("name") or raw.get("id") or "").strip()
    base = str(raw.get("base_url") or raw.get("base-url") or raw.get("baseUrl") or "").strip()
    if not base:
        return None
    if not name:
        name = "compat"

    api_key = str(
        raw.get("api_key") or raw.get("api-key") or raw.get("apiKey") or ""
    ).strip()
    if not api_key:
        entries = (
            raw.get("api_key_entries")
            or raw.get("api-key-entries")
            or raw.get("apiKeys")
            or []
        )
        if isinstance(entries, list):
            for ent in entries:
                if isinstance(ent, str) and ent.strip():
                    api_key = ent.strip()
                    break
                if isinstance(ent, dict):
                    k = str(
                        ent.get("api_key") or ent.get("api-key") or ent.get("apiKey") or ""
                    ).strip()
                    if k:
                        api_key = k
                        break
    api_key = _expand_env(api_key)

    models: List[CompatModel] = []
    for m in raw.get("models") or []:
        parsed = _parse_model_entry(m)
        if parsed:
            models.append(parsed)

    prefix = str(raw.get("prefix") or "").strip()
    disabled = _as_bool(raw.get("disabled"))
    return CompatProvider(
        name=name,
        base_url=base,
        api_key=api_key,
        prefix=prefix,
        models=models,
        disabled=disabled,
    )


def _expand_env(value: str) -> str:
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return (os.environ.get(value[2:-1]) or "").strip()
    if value.startswith("$") and len(value) > 1 and value[1:].replace("_", "").isalnum():
        return (os.environ.get(value[1:]) or "").strip()
    return value


def _resolve_compat_path(text: str) -> Optional[Path]:
    """Resolve OPENAI_COMPATIBILITY path (cwd, then project root)."""
    candidates = [
        Path(text).expanduser(),
        _PROJECT_ROOT / text,
        Path.cwd() / text,
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return None


def load_compat_providers(raw: str) -> List[CompatProvider]:
    """Parse OPENAI_COMPATIBILITY: file path or inline JSON."""
    text = (raw or "").strip()
    if not text:
        return []

    data: Any = None
    path = _resolve_compat_path(text)
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("OPENAI_COMPATIBILITY file %s invalid: %s", path, exc)
            return []
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "OPENAI_COMPATIBILITY is neither a readable path nor JSON: %s",
                text[:80],
            )
            return []

    if isinstance(data, dict):
        data = (
            data.get("providers")
            or data.get("openai-compatibility")
            or data.get("openai_compatibility")
            or [data]
        )
    if not isinstance(data, list):
        return []

    out: List[CompatProvider] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        p = _parse_provider(item)
        if p and not p.disabled:
            out.append(p)
    return out


def build_provider_list(
    *,
    compat_raw: str,
    default_base_url: str,
    default_api_key: str,
    default_models: List[str],
    default_name: str = "default",
) -> List[CompatProvider]:
    """Merge explicit compat providers + legacy single default upstream."""
    providers = load_compat_providers(compat_raw)

    default_base = (default_base_url or "").rstrip("/")
    default_key = (default_api_key or "").strip()

    has_default_base = any(
        p.normalized_base() == default_base for p in providers if default_base
    )

    if default_base and default_key and not has_default_base:
        models = [CompatModel(name=m, alias=m) for m in default_models if m]
        providers.insert(
            0,
            CompatProvider(
                name=default_name,
                base_url=default_base,
                api_key=default_key,
                models=models,
            ),
        )
    elif default_base and default_key and providers and not any(p.api_key for p in providers):
        for p in providers:
            if not p.api_key:
                p.api_key = default_key

    for p in providers:
        if not p.api_key:
            env_candidates = [
                f"{p.name.upper().replace('-', '_')}_API_KEY",
                "VOYA_API_KEY",
                "OPENAI_API_KEY",
                "NEWAPI_API_KEY",
                "XAI_API_KEY",
            ]
            for env_name in env_candidates:
                alt = (os.environ.get(env_name) or "").strip()
                if alt:
                    p.api_key = alt
                    break

    return [p for p in providers if p.base_url]


def resolve_route(
    providers: List[CompatProvider],
    *,
    model: Optional[str],
    global_aliases: Dict[str, str],
    default_models: List[str],
) -> UpstreamRoute:
    """Pick provider + rewrite model id for an upstream call."""
    client_model = (model or "").strip()
    if not client_model:
        client_model = default_models[0] if default_models else "grok-3"

    mapped = global_aliases.get(client_model, client_model)

    # prefix/model pin: "iamhc/DeepSeek-V4-Pro"
    if "/" in mapped and providers:
        head, rest = mapped.split("/", 1)
        head_l = head.lower()
        for p in providers:
            pin = (p.prefix or p.name or "").strip().lower()
            if pin and pin == head_l:
                upstream = _match_provider_model(p, rest) or rest
                return UpstreamRoute(
                    provider=p.display_name(),
                    base_url=p.normalized_base(),
                    api_key=p.api_key,
                    model=upstream,
                    client_model=client_model,
                )

    for p in providers:
        upstream = _match_provider_model(p, mapped)
        if upstream is not None:
            return UpstreamRoute(
                provider=p.display_name(),
                base_url=p.normalized_base(),
                api_key=p.api_key,
                model=upstream,
                client_model=client_model,
            )

    for p in providers:
        if p.api_key:
            return UpstreamRoute(
                provider=p.display_name(),
                base_url=p.normalized_base(),
                api_key=p.api_key,
                model=mapped,
                client_model=client_model,
            )

    if providers:
        p = providers[0]
        return UpstreamRoute(
            provider=p.display_name(),
            base_url=p.normalized_base(),
            api_key=p.api_key,
            model=mapped,
            client_model=client_model,
        )

    raise RuntimeError(
        "No custom upstream configured. Set XAI_BASE_URL + XAI_API_KEY "
        "(or VOYA_API_KEY), or OPENAI_COMPATIBILITY providers. "
        "Official Grok is separate: UPSTREAM_MODE=oauth (Device Code) or "
        "UPSTREAM_MODE=credential (import xai-*.json)."
    )


def _match_provider_model(p: CompatProvider, model: str) -> Optional[str]:
    if not p.models:
        return None
    m = model.strip()
    for entry in p.models:
        if entry.client_id() == m or entry.upstream_id() == m:
            return entry.upstream_id()
    return None


def list_client_models(providers: List[CompatProvider]) -> List[Tuple[str, str]]:
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for p in providers:
        owned = f"compat:{p.display_name()}"
        if p.models:
            for entry in p.models:
                cid = entry.client_id()
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append((cid, owned))
    return out


def providers_summary(providers: List[CompatProvider]) -> List[Dict[str, Any]]:
    return [
        {
            "name": p.display_name(),
            "base_url": p.normalized_base(),
            "prefix": p.prefix or None,
            "key_configured": bool(p.api_key),
            "models": [
                {"name": m.upstream_id(), "alias": m.client_id()} for m in p.models
            ],
        }
        for p in providers
    ]
