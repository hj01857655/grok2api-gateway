"""Upstream client.

Modes (UPSTREAM_MODE):
  auto       — official Grok if credential present, else managed channels
  compat     — managed mid-station channels only (admin-added)
  oauth      — official Grok via Device Code OAuth login
  credential — official Grok via imported credential files (xai-*.json)

Routing rule: same protocol → pass-through; mismatch → convert once.

  Mid-station (OpenAI-compat / NewAPI style):
    · Client Chat      → POST …/chat/completions   (pass-through)
    · Client Responses → POST …/responses          (pass-through)
    · Client Anthropic → POST …/messages           (pass-through)
    No protocol conversion. Only model rewrite + auth headers.

  Official token (CPA xai_executor — only speaks /responses):
    · Client Responses → POST …/responses          (native, sanitize)
    · Client Chat      → Chat→Responses once
    · Client Anthropic → Anthropic→Chat→Responses once
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .config import Settings, get_settings
from .converters.responses import (
    chat_to_responses_request,
    collect_responses_completed,
    prepare_official_responses_request,
    responses_result_to_chat,
    stream_responses_to_chat,
)
from .providers import UpstreamRoute
from .util import iter_sse_data_lines, now_ts

logger = logging.getLogger("grok2api.upstream")


class UpstreamError(Exception):
    def __init__(self, status: int, body: str, payload: Any = None) -> None:
        self.status = status
        self.body = body
        self.payload = payload
        super().__init__(f"upstream {status}: {body[:300]}")


class UpstreamClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._timeout = httpx.Timeout(self.settings.upstream_timeout)
        self._token_storage = None  # official TokenStorage (oauth | credential)

        if self.settings.is_official_mode():
            self._init_official()
        else:
            providers = self.settings.compat_providers()
            if not providers or not any(p.api_key for p in providers):
                raise RuntimeError(
                    "No mid-station channels configured. "
                    "Open /admin → add a channel (base URL + API key + models). "
                    f"Store: {self.settings.providers_store_path()}  |  "
                    "Official Grok (optional): Device Code login or import xai-*.json."
                )

    def _init_official(self) -> None:
        from .oauth.xai import ensure_fresh_token, load_token

        mode = self.settings.upstream_mode
        ts = load_token(
            path=self.settings.oauth_token_path(),
            auths_dir=self.settings.auths_dir(),
        )
        if not ts or not ts.access_token:
            if mode == "oauth":
                hint = "python -m app.oauth.login  (Device Code)"
            else:
                hint = (
                    "import xai-*.json via /admin or: "
                    "python -m app.oauth.login --import path"
                )
            raise RuntimeError(
                f"UPSTREAM_MODE={mode} but no credential found. "
                f"{hint}  (auths dir: {self.settings.auths_dir()})"
            )
        try:
            ts = ensure_fresh_token(ts, auths_dir=self.settings.auths_dir())
        except Exception as exc:
            logger.warning("token refresh skipped/failed: %s", exc)
        self._token_storage = ts
        logger.info(
            "upstream official mode=%s email=%s base=%s using_api=%s "
            "(wire=/responses only; convert only when client ≠ Responses)",
            mode,
            ts.email or ts.sub or "?",
            self._official_base_url(),
            ts.using_api,
        )

    def uses_official_wire(self) -> bool:
        """True when traffic goes to official /responses (not mid-station)."""
        return (
            self.settings.is_official_mode() and self._token_storage is not None
        )

    def _official_base_url(self) -> str:
        from .oauth.xai import resolve_chat_base_url

        return resolve_chat_base_url(self._token_storage).rstrip("/")

    def _official_headers(self, *, stream: bool = False) -> Dict[str, str]:
        from .oauth.xai import ensure_fresh_token, oauth_request_headers

        try:
            self._token_storage = ensure_fresh_token(
                self._token_storage,
                auths_dir=self.settings.auths_dir(),
            )
        except Exception as exc:
            logger.warning("token ensure_fresh failed: %s", exc)
        return oauth_request_headers(self._token_storage, stream=stream)

    def _route_for(self, model: Optional[str] = None) -> UpstreamRoute:
        return self.settings.resolve_upstream(model)

    def _url(self, base: str, path: str) -> str:
        base = base.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _mid_headers(self, route: UpstreamRoute, *, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {route.api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def _prepare_mid_body(
        self,
        body: Dict[str, Any],
        *,
        stream: bool,
        route: Optional[UpstreamRoute] = None,
    ) -> Dict[str, Any]:
        """Pass-through body with model rewrite only — no protocol conversion."""
        out = dict(body)
        if route is None:
            route = self._route_for(out.get("model"))
        out["model"] = route.model
        out["stream"] = stream
        if not out.get("tools"):
            out.pop("tools", None)
            out.pop("tool_choice", None)
        return out

    def credential_info(self) -> Dict[str, Any]:
        if self.uses_official_wire():
            ts = self._token_storage
            return {
                "mode": self.settings.effective_upstream_mode(),
                "email": ts.email,
                "using_api": ts.using_api,
                "base_url": self._official_base_url(),
                "wire": "/responses",
                "expired": ts.expired,
            }
        providers = self.settings.providers_public()
        default_base = providers[0].get("base_url") if providers else ""
        return {
            "mode": "compat",
            "base_url": default_base,
            "wire": "pass-through (chat|responses|messages)",
            "key_configured": any(p.get("key_configured") for p in providers),
            "providers": providers,
        }

    async def list_models(self) -> Dict[str, Any]:
        if self.uses_official_wire():
            data = await self._list_models_single(
                self._official_base_url(),
                self._official_headers(stream=False),
            )
            return data

        merged: Dict[str, Dict[str, Any]] = {}
        ts = now_ts()
        for p in self.settings.compat_providers():
            if not p.api_key:
                continue
            headers = {
                "Authorization": f"Bearer {p.api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        self._url(p.normalized_base(), "/models"),
                        headers=headers,
                    )
                    if resp.status_code >= 400:
                        logger.warning(
                            "list_models provider=%s status=%s",
                            p.display_name(),
                            resp.status_code,
                        )
                        continue
                    data = resp.json()
                    for m in data.get("data") or []:
                        if not isinstance(m, dict):
                            continue
                        mid = m.get("id")
                        if mid and mid not in merged:
                            m = dict(m)
                            m.setdefault("owned_by", f"compat:{p.display_name()}")
                            merged[mid] = m
            except Exception as exc:
                logger.warning("list_models provider=%s failed: %s", p.display_name(), exc)

        from .providers import list_client_models

        for mid, owned in list_client_models(self.settings.compat_providers()):
            if mid not in merged:
                merged[mid] = {
                    "id": mid,
                    "object": "model",
                    "created": ts,
                    "owned_by": owned,
                }
        for alias in self.settings.alias_map():
            if alias not in merged:
                merged[alias] = {
                    "id": alias,
                    "object": "model",
                    "created": ts,
                    "owned_by": "grok2api-alias",
                }
        for mid in self.settings.default_model_list():
            if mid not in merged:
                merged[mid] = {
                    "id": mid,
                    "object": "model",
                    "created": ts,
                    "owned_by": "grok2api",
                }

        if not merged:
            return self._fallback_models()
        return {"object": "list", "data": list(merged.values())}

    async def _list_models_single(self, base: str, headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self._url(base, "/models"), headers=headers)
                if resp.status_code >= 400:
                    logger.warning("list_models upstream %s — using fallback", resp.status_code)
                    return self._fallback_models()
                data = resp.json()
                if not data.get("data"):
                    return self._fallback_models()
                return data
        except Exception as exc:
            logger.warning("list_models failed: %s — using fallback", exc)
            return self._fallback_models()

    def _fallback_models(self) -> Dict[str, Any]:
        ts = now_ts()
        models = list(self.settings.default_model_list())
        for alias in self.settings.alias_map():
            if alias not in models:
                models.append(alias)
        from .providers import list_client_models

        for mid, _ in list_client_models(self.settings.compat_providers()):
            if mid not in models:
                models.append(mid)
        if self.uses_official_wire():
            owned = f"xai-{self.settings.effective_upstream_mode()}"
        else:
            owned = "grok2api"
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "created": ts, "owned_by": owned}
                for mid in models
            ],
        }

    # ------------------------------------------------------------------
    # Mid-station pass-through (no protocol conversion)
    # ------------------------------------------------------------------

    async def _mid_post_json(
        self, path: str, body: Dict[str, Any], *, stream: bool = False
    ) -> Dict[str, Any]:
        route = self._route_for(body.get("model"))
        payload = self._prepare_mid_body(body, stream=stream, route=route)
        headers = self._mid_headers(route, stream=False)
        url = self._url(route.base_url, path)
        logger.info(
            "upstream mid pass-through path=%s provider=%s model=%s→%s stream=false",
            path,
            route.provider,
            route.client_model or body.get("model"),
            payload.get("model"),
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                except Exception:
                    err = None
                raise UpstreamError(resp.status_code, resp.text, err)
            return resp.json()

    async def _mid_stream(
        self, path: str, body: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        route = self._route_for(body.get("model"))
        payload = self._prepare_mid_body(body, stream=True, route=route)
        headers = self._mid_headers(route, stream=True)
        url = self._url(route.base_url, path)
        logger.info(
            "upstream mid pass-through path=%s provider=%s model=%s→%s stream=true",
            path,
            route.provider,
            route.client_model or body.get("model"),
            payload.get("model"),
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    yield (
                        b"__HTTP_ERROR__"
                        + str(resp.status_code).encode()
                        + b"__"
                        + err
                    )
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk

    # ------------------------------------------------------------------
    # Official wire helpers (POST /responses)
    # ------------------------------------------------------------------

    def _raise_from_http_error_marker(self, msg: str) -> None:
        if msg.startswith("__HTTP_ERROR__"):
            rest = msg[len("__HTTP_ERROR__") :]
            status_s, _, body_s = rest.partition("__")
            try:
                status = int(status_s)
            except ValueError:
                status = 502
            payload_err = None
            try:
                import json

                payload_err = json.loads(body_s)
            except Exception:
                pass
            raise UpstreamError(status, body_s, payload_err)
        raise UpstreamError(502, msg, None)

    async def _stream_official_responses_bytes(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        headers = self._official_headers(stream=True)
        url = self._url(self._official_base_url(), "/responses")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    yield (
                        b"__HTTP_ERROR__"
                        + str(resp.status_code).encode()
                        + b"__"
                        + err
                    )
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk

    def _official_chat_body(
        self, body: Dict[str, Any], *, stream: bool
    ) -> tuple[Dict[str, Any], str]:
        """Chat client → Responses payload (one convert)."""
        client_model = str(body.get("model") or "")
        model = self.settings.resolve_model(body.get("model"))
        chat = dict(body)
        chat["model"] = model
        chat["stream"] = stream
        if not chat.get("tools"):
            chat.pop("tools", None)
            chat.pop("tool_choice", None)
        payload = chat_to_responses_request(chat, stream=stream)
        return payload, client_model or model

    async def _official_chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload, client_model = self._official_chat_body(body, stream=True)
        logger.info(
            "upstream official convert chat→responses model=%s→%s stream=collect",
            client_model,
            payload.get("model"),
        )
        try:
            completed = await collect_responses_completed(
                iter_sse_data_lines(self._stream_official_responses_bytes(payload))
            )
        except RuntimeError as e:
            self._raise_from_http_error_marker(str(e))
            raise
        return responses_result_to_chat(completed, client_model=client_model)

    async def _official_stream_chat_completions(
        self, body: Dict[str, Any]
    ) -> AsyncIterator[bytes]:
        payload, client_model = self._official_chat_body(body, stream=True)
        logger.info(
            "upstream official convert chat→responses model=%s→%s stream=true",
            client_model,
            payload.get("model"),
        )
        async for out in stream_responses_to_chat(
            iter_sse_data_lines(self._stream_official_responses_bytes(payload)),
            client_model=client_model,
        ):
            yield out

    def _official_responses_body(
        self, body: Dict[str, Any], *, stream: bool
    ) -> tuple[Dict[str, Any], str]:
        client_model = str(body.get("model") or "")
        model = self.settings.resolve_model(body.get("model"))
        payload = prepare_official_responses_request(
            body, stream=stream, model=model
        )
        return payload, client_model or model

    # ------------------------------------------------------------------
    # Public: Chat Completions
    # ------------------------------------------------------------------

    async def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self.uses_official_wire():
            return await self._official_chat_completions(body)
        return await self._mid_post_json("/chat/completions", body, stream=False)

    async def stream_chat_completions(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        if self.uses_official_wire():
            async for chunk in self._official_stream_chat_completions(body):
                yield chunk
            return
        async for chunk in self._mid_stream("/chat/completions", body):
            yield chunk

    # ------------------------------------------------------------------
    # Public: Responses (mid = pass-through; official = native /responses)
    # ------------------------------------------------------------------

    async def responses(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if self.uses_official_wire():
            payload, client_model = self._official_responses_body(body, stream=True)
            logger.info(
                "upstream official client=responses wire=/responses model=%s→%s "
                "stream=collect (native)",
                client_model,
                payload.get("model"),
            )
            try:
                completed = await collect_responses_completed(
                    iter_sse_data_lines(
                        self._stream_official_responses_bytes(payload)
                    )
                )
            except RuntimeError as e:
                self._raise_from_http_error_marker(str(e))
                raise
            if client_model and isinstance(completed, dict):
                completed = dict(completed)
                completed["model"] = client_model
            return completed

        # Mid-station: pass-through to /responses (no Chat hop)
        return await self._mid_post_json("/responses", body, stream=False)

    async def stream_responses(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        if self.uses_official_wire():
            payload, client_model = self._official_responses_body(body, stream=True)
            logger.info(
                "upstream official client=responses wire=/responses model=%s→%s "
                "stream=true (native)",
                client_model,
                payload.get("model"),
            )
            async for chunk in self._stream_official_responses_bytes(payload):
                yield chunk
            return

        async for chunk in self._mid_stream("/responses", body):
            yield chunk

    # ------------------------------------------------------------------
    # Public: Anthropic Messages (mid = pass-through; official = convert)
    # ------------------------------------------------------------------

    async def messages(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Anthropic non-stream.

        Mid-station: POST …/messages pass-through.
        Official: Anthropic → Chat → /responses → Chat → Anthropic (in products).
        """
        if self.uses_official_wire():
            # products layer already converted to Chat when calling chat_completions
            raise RuntimeError(
                "official Anthropic uses chat_completions after anthropic_to_chat; "
                "do not call messages() on official wire"
            )
        return await self._mid_post_json("/messages", body, stream=False)

    async def stream_messages(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        if self.uses_official_wire():
            raise RuntimeError(
                "official Anthropic uses stream_chat_completions after anthropic_to_chat"
            )
        async for chunk in self._mid_stream("/messages", body):
            yield chunk
