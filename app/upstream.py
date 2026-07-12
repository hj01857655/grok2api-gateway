"""Upstream client — official Grok only.

UPSTREAM_MODE:
  auto       — official Grok if credential present (else error)
  oauth      — Device Code OAuth credential
  credential — imported xai-*.json

Official wire speaks only POST .../responses:
  - Client Responses -> native /responses (sanitize)
  - Client Chat      -> Chat->Responses once
  - Client Anthropic -> Anthropic->Responses direct (no Chat hop)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .config import Settings, get_settings
from .converters.anthropic import (
    anthropic_to_responses_request,
    responses_result_to_anthropic,
)
from .converters.responses import (
    chat_to_responses_request,
    collect_responses_completed,
    prepare_official_responses_request,
    responses_result_to_chat,
    stream_responses_to_chat,
)
from .util import iter_sse_data_lines, now_ts

logger = logging.getLogger("grok2api.upstream")


class UpstreamError(Exception):
    def __init__(self, status: int, body: str, payload: Any = None) -> None:
        self.status = status
        self.body = body
        self.payload = payload
        super().__init__(f"upstream {status}: {body[:300]}")


class UpstreamClient:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        conv_id: str = "",
    ) -> None:
        self.settings = settings or get_settings()
        self._timeout = httpx.Timeout(self.settings.upstream_timeout)
        self._token_storage = None
        # x-grok-conv-id: xAI cache-affinity handle; higher hit rate when the
        # same conversation reuses one id (docs.x.ai prompt-caching guide).
        self._conv_id = (conv_id or "").strip()
        self._init_official()

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
            elif mode == "credential":
                hint = (
                    "import xai-*.json via /admin or: "
                    "python -m app.oauth.login --import path"
                )
            else:
                hint = (
                    "Device Code: python -m app.oauth.login  |  "
                    "or import xai-*.json via /admin"
                )
            raise RuntimeError(
                f"Official Grok credential required (UPSTREAM_MODE={mode}). "
                f"{hint}  (auths dir: {self.settings.auths_dir()})"
            )
        try:
            ts = ensure_fresh_token(ts, auths_dir=self.settings.auths_dir())
        except Exception as exc:
            logger.warning("token refresh skipped/failed: %s", exc)
        self._token_storage = ts
        logger.info(
            "upstream official mode=%s email=%s base=%s using_api=%s "
            "(wire=/responses only; convert only when client != Responses)",
            self.settings.effective_upstream_mode(),
            ts.email or ts.sub or "?",
            self._official_base_url(),
            ts.using_api,
        )

    def uses_official_wire(self) -> bool:
        """Always True once credential loaded — gateway is official Grok only."""
        return self._token_storage is not None

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
        return oauth_request_headers(
            self._token_storage, stream=stream, session_id=self._conv_id
        )

    def _url(self, base: str, path: str) -> str:
        base = base.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def credential_info(self) -> Dict[str, Any]:
        ts = self._token_storage
        return {
            "mode": self.settings.effective_upstream_mode(),
            "email": ts.email if ts else None,
            "using_api": ts.using_api if ts else None,
            "base_url": self._official_base_url() if ts else "",
            "wire": "/responses",
            "expired": ts.expired if ts else None,
        }

    async def list_models(self) -> Dict[str, Any]:
        return await self._list_models_single(
            self._official_base_url(),
            self._official_headers(stream=False),
        )

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
        owned = f"xai-{self.settings.effective_upstream_mode()}"
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "created": ts, "owned_by": owned}
                for mid in models
            ],
        }

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
        """Chat client -> Responses payload (one convert)."""
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
            "upstream official convert chat->responses model=%s->%s stream=collect",
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
            "upstream official convert chat->responses model=%s->%s stream=true",
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

    async def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._official_chat_completions(body)

    async def stream_chat_completions(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        async for chunk in self._official_stream_chat_completions(body):
            yield chunk

    async def responses(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload, client_model = self._official_responses_body(body, stream=True)
        logger.info(
            "upstream official client=responses wire=/responses model=%s->%s "
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

    async def stream_responses(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        payload, client_model = self._official_responses_body(body, stream=True)
        logger.info(
            "upstream official client=responses wire=/responses model=%s->%s "
            "stream=true (native)",
            client_model,
            payload.get("model"),
        )
        async for chunk in self._stream_official_responses_bytes(payload):
            yield chunk

    def _official_anthropic_body(
        self, body: Dict[str, Any], *, stream: bool
    ) -> tuple[Dict[str, Any], str]:
        """Anthropic client -> Responses payload (one convert, no Chat hop)."""
        client_model = str(body.get("model") or "")
        model = self.settings.resolve_model(body.get("model"))
        payload = anthropic_to_responses_request(body, stream=stream, model=model)
        return payload, client_model or model

    async def messages(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload, client_model = self._official_anthropic_body(body, stream=True)
        logger.info(
            "upstream official convert anthropic->responses model=%s->%s stream=collect",
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
        return responses_result_to_anthropic(completed, client_model)

    async def stream_messages(self, body: Dict[str, Any]) -> AsyncIterator[bytes]:
        payload, _client_model = self._official_anthropic_body(body, stream=True)
        logger.info(
            "upstream official convert anthropic->responses model=%s->%s stream=true",
            _client_model,
            payload.get("model"),
        )
        async for chunk in self._stream_official_responses_bytes(payload):
            yield chunk

    async def embeddings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """xAI /v1/embeddings — OpenAI-compatible passthrough (no conversion).

        Body shape and response shape match OpenAI's embeddings API:
        request  {model, input, dimensions?, ...}
        response {object: "list", model, data: [{index, embedding, object}], usage}
        """
        headers = self._official_headers(stream=False)
        url = self._url(self._official_base_url(), "/embeddings")
        logger.info(
            "upstream official client=embeddings wire=/embeddings model=%s (passthrough)",
            body.get("model") or "?",
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    payload = None
                    try:
                        payload = resp.json()
                    except Exception:
                        pass
                    raise UpstreamError(resp.status_code, resp.text, payload)
                return resp.json()
        except httpx.HTTPError as exc:
            raise UpstreamError(502, str(exc), None) from exc
