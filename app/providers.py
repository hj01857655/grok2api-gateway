"""OpenAI-compatible multi-provider routing.

Providers (mid-station / custom APIs) come from the admin-managed store
(~/.grok2api/providers.json via channel_store). This module only routes
and rewrites model ids — it does NOT bootstrap channels from .env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
        "No mid-station channels configured. "
        "Open /admin → add a channel (base URL + API key + models). "
        "Official Grok is separate: Device Code login or import xai-*.json."
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
