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

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .config import Settings, get_settings
from .credential_pool import CredentialPool, TokenSlot
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


# HTTP statuses that justify trying another credential when the pool has one.
_RETRYABLE_STATUS: frozenset = frozenset({429, 401, 403, 500, 502, 503, 504})


def _parse_error_code(raw: Any) -> str:
    """Extract xAI ``code`` field (e.g. ``subscription:free-usage-exhausted``).

    Accepts raw ``bytes`` / ``str`` bodies; returns empty string on any parse
    failure so callers can safely feed the result into the cooldown mapper.
    """
    if not raw:
        return ""
    try:
        if isinstance(raw, (bytes, bytearray)):
            obj = json.loads(raw)
        elif isinstance(raw, str):
            obj = json.loads(raw)
        elif isinstance(raw, dict):
            obj = raw
        else:
            return ""
        if isinstance(obj, dict):
            return str(obj.get("code") or "")
    except Exception:
        pass
    return ""


class UpstreamClient:
    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """Long-lived client: no per-request state on the instance.

        ``conv_id`` (xAI cache-affinity) is a REQUEST attribute — pass it to
        the individual method calls, do NOT store it here. A single client
        is expected to be reused across many concurrent requests.
        """
        self.settings = settings or get_settings()
        self._timeout = httpx.Timeout(self.settings.upstream_timeout)
        # Shared httpx client — avoids per-request TCP + TLS handshake. The
        # AsyncClient is created here (sync ok) and lazily binds to the
        # running loop on first request. Pair with `aclose()` on shutdown.
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=self._timeout)
        # Credential pool — every ``xai-*.json`` under ``auths_dir`` becomes
        # one slot. Request-time selection is round-robin with per-slot
        # cooldown on 429/401/5xx (see ``credential_pool.py``).
        self._pool: CredentialPool = self._init_pool()

    async def aclose(self) -> None:
        """Close the shared httpx client (call from an app shutdown hook)."""
        await self._http.aclose()

    def _init_pool(self) -> CredentialPool:
        pool = CredentialPool(self.settings.auths_dir())
        if pool.size() == 0:
            mode = self.settings.upstream_mode
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
        primary = pool.primary()
        logger.info(
            "upstream pool loaded: %d slot(s), primary=%s using_api=%s",
            pool.size(),
            (primary.email if primary else "?"),
            (primary.storage.using_api if primary else None),
        )
        return pool

    def reload_pool(self) -> int:
        """Rescan ``auths_dir`` and rebuild the slot list; preserves cooldowns."""
        return self._pool.reload()

    def uses_official_wire(self) -> bool:
        """True while the pool holds at least one loaded credential."""
        return self._pool.size() > 0

    def _headers_for_slot(
        self,
        slot: TokenSlot,
        *,
        endpoint: str = "responses",
        conv_id: str = "",
    ) -> Dict[str, str]:
        """Refresh the slot's token if near expiry, then build endpoint headers.

        ``endpoint`` — one of ``"responses"`` / ``"models"`` / ``"embeddings"``
        (matches ``oauth_request_headers``). ``conv_id`` maps to
        ``x-grok-conv-id`` for xAI cache-affinity.
        """
        from .oauth.xai import ensure_fresh_token, oauth_request_headers

        try:
            slot.storage = ensure_fresh_token(
                slot.storage,
                auths_dir=self.settings.auths_dir(),
            )
        except Exception as exc:
            logger.warning(
                "ensure_fresh_token failed for %s: %s",
                slot.email or slot.path.name,
                exc,
            )
        return oauth_request_headers(
            slot.storage,
            endpoint=endpoint,
            session_id=(conv_id or "").strip(),
        )

    def _base_for_slot(self, slot: TokenSlot) -> str:
        from .oauth.xai import resolve_chat_base_url

        return resolve_chat_base_url(slot.storage).rstrip("/")

    def _url(self, base: str, path: str) -> str:
        base = base.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _pick_slot(self, tried: set) -> Optional[TokenSlot]:
        """Pick the next non-cooldown slot, skipping any already tried this request.

        Falls back to ``pool.pick_any`` (soonest-available) when everything is
        on cooldown so we still respond rather than hard-fail. Returns
        ``None`` when the pool is empty or every slot has already been tried.
        """
        slot = self._pool.pick()
        if slot is not None and slot.path.name not in tried:
            return slot
        slot = self._pool.pick_any()
        if slot is not None and slot.path.name not in tried:
            return slot
        return None

    def credential_info(self) -> Dict[str, Any]:
        """Compact credential summary — ``/health`` and legacy admin views."""
        primary = self._pool.primary()
        if primary is None:
            return {
                "mode": self.settings.effective_upstream_mode(),
                "email": None,
                "using_api": None,
                "base_url": "",
                "wire": "/responses",
                "expired": None,
                "pool_size": 0,
                "pool_available": 0,
            }
        return {
            "mode": self.settings.effective_upstream_mode(),
            "email": primary.storage.email,
            "using_api": primary.storage.using_api,
            "base_url": self._base_for_slot(primary),
            "wire": "/responses",
            "expired": primary.storage.expired,
            "pool_size": self._pool.size(),
            "pool_available": self._pool.available_count(),
        }

    def pool_status(self) -> Dict[str, Any]:
        """Full pool snapshot for the admin console (per-slot detail)."""
        return self._pool.status()

    async def list_models(self) -> Dict[str, Any]:
        """List models — one shot against the primary slot (else local fallback).

        ``/v1/models`` is safe to answer from a single credential and doesn't
        justify multi-slot retry; if it fails, return the local fallback so
        clients still discover the models we advertise.
        """
        slot = self._pool.pick() or self._pool.pick_any() or self._pool.primary()
        if slot is None:
            return self._fallback_models()
        try:
            headers = self._headers_for_slot(slot, endpoint="models")
            base = self._base_for_slot(slot)
            resp = await self._http.get(self._url(base, "/models"), headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "list_models upstream %s on %s — using fallback",
                    resp.status_code,
                    slot.email or slot.path.name,
                )
                self._pool.mark_cooldown(
                    slot,
                    status=resp.status_code,
                    body=resp.text,
                    error_code=_parse_error_code(resp.content),
                )
                return self._fallback_models()
            data = resp.json()
            if not data.get("data"):
                return self._fallback_models()
            self._pool.note_success(slot)
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
        self, payload: Dict[str, Any], *, conv_id: str = ""
    ) -> AsyncIterator[bytes]:
        """POST /responses with per-slot retry on connection-time failures.

        Retry only when the failure lands **before** we start streaming bytes
        to the client — once upstream chunks have flowed downstream, we
        can't rewind. Every failure is recorded via ``pool.mark_cooldown`` so
        subsequent requests naturally avoid the exhausted credential.
        """
        pool_size = self._pool.size()
        max_attempts = min(pool_size, 3) if pool_size > 0 else 1
        tried: set = set()

        for attempt in range(max_attempts):
            slot = self._pick_slot(tried)
            if slot is None:
                yield (
                    b"__HTTP_ERROR__503__"
                    + b'{"error":"no credentials available"}'
                )
                return
            tried.add(slot.path.name)

            try:
                headers = self._headers_for_slot(
                    slot, endpoint="responses", conv_id=conv_id
                )
                base = self._base_for_slot(slot)
                url = self._url(base, "/responses")
            except Exception as exc:
                logger.warning(
                    "headers/base build failed for %s: %s",
                    slot.email or slot.path.name,
                    exc,
                )
                self._pool.mark_cooldown(
                    slot, status=401, error_code="prepare_failed"
                )
                if attempt + 1 < max_attempts:
                    continue
                yield (
                    b"__HTTP_ERROR__500__"
                    + str(exc).encode("utf-8", errors="replace")
                )
                return

            try:
                async with self._http.stream(
                    "POST", url, headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        error_code = _parse_error_code(err)
                        self._pool.mark_cooldown(
                            slot,
                            status=resp.status_code,
                            body=err.decode("utf-8", errors="replace"),
                            error_code=error_code,
                        )
                        if (
                            attempt + 1 < max_attempts
                            and resp.status_code in _RETRYABLE_STATUS
                        ):
                            logger.info(
                                "retrying next credential (attempt %d/%d) after %d on %s",
                                attempt + 2,
                                max_attempts,
                                resp.status_code,
                                slot.email or slot.path.name,
                            )
                            continue
                        yield (
                            b"__HTTP_ERROR__"
                            + str(resp.status_code).encode()
                            + b"__"
                            + err
                        )
                        return

                    self._pool.note_success(slot)
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
                    return
            except httpx.HTTPError as exc:
                logger.warning(
                    "stream error on %s: %s",
                    slot.email or slot.path.name,
                    exc,
                )
                self._pool.mark_cooldown(slot, status=502)
                if attempt + 1 < max_attempts:
                    continue
                yield (
                    b"__HTTP_ERROR__502__"
                    + str(exc).encode("utf-8", errors="replace")
                )
                return

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

    async def _official_chat_completions(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> Dict[str, Any]:
        payload, client_model = self._official_chat_body(body, stream=True)
        logger.info(
            "upstream official convert chat->responses model=%s->%s client_stream=false",
            client_model,
            payload.get("model"),
        )
        try:
            completed = await collect_responses_completed(
                iter_sse_data_lines(
                    self._stream_official_responses_bytes(payload, conv_id=conv_id)
                )
            )
        except RuntimeError as e:
            self._raise_from_http_error_marker(str(e))
            raise
        return responses_result_to_chat(completed, client_model=client_model)

    async def _official_stream_chat_completions(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> AsyncIterator[bytes]:
        payload, client_model = self._official_chat_body(body, stream=True)
        logger.info(
            "upstream official convert chat->responses model=%s->%s client_stream=true",
            client_model,
            payload.get("model"),
        )
        async for out in stream_responses_to_chat(
            iter_sse_data_lines(
                self._stream_official_responses_bytes(payload, conv_id=conv_id)
            ),
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

    async def chat_completions(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> Dict[str, Any]:
        return await self._official_chat_completions(body, conv_id=conv_id)

    async def stream_chat_completions(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> AsyncIterator[bytes]:
        async for chunk in self._official_stream_chat_completions(
            body, conv_id=conv_id
        ):
            yield chunk

    async def responses(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> Dict[str, Any]:
        payload, client_model = self._official_responses_body(body, stream=True)
        logger.info(
            "upstream official client=responses wire=/responses model=%s->%s "
            "client_stream=false (native)",
            client_model,
            payload.get("model"),
        )
        try:
            completed = await collect_responses_completed(
                iter_sse_data_lines(
                    self._stream_official_responses_bytes(payload, conv_id=conv_id)
                )
            )
        except RuntimeError as e:
            self._raise_from_http_error_marker(str(e))
            raise
        if client_model and isinstance(completed, dict):
            completed = dict(completed)
            completed["model"] = client_model
        return completed

    async def stream_responses(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> AsyncIterator[bytes]:
        payload, client_model = self._official_responses_body(body, stream=True)
        logger.info(
            "upstream official client=responses wire=/responses model=%s->%s "
            "client_stream=true (native)",
            client_model,
            payload.get("model"),
        )
        async for chunk in self._stream_official_responses_bytes(
            payload, conv_id=conv_id
        ):
            yield chunk

    def _official_anthropic_body(
        self, body: Dict[str, Any], *, stream: bool
    ) -> tuple[Dict[str, Any], str]:
        """Anthropic client -> Responses payload (one convert, no Chat hop)."""
        client_model = str(body.get("model") or "")
        model = self.settings.resolve_model(body.get("model"))
        payload = anthropic_to_responses_request(body, stream=stream, model=model)
        return payload, client_model or model

    async def messages(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> Dict[str, Any]:
        payload, client_model = self._official_anthropic_body(body, stream=True)
        logger.info(
            "upstream official convert anthropic->responses model=%s->%s client_stream=false",
            client_model,
            payload.get("model"),
        )
        try:
            completed = await collect_responses_completed(
                iter_sse_data_lines(
                    self._stream_official_responses_bytes(payload, conv_id=conv_id)
                )
            )
        except RuntimeError as e:
            self._raise_from_http_error_marker(str(e))
            raise
        return responses_result_to_anthropic(completed, client_model)

    async def stream_messages(
        self, body: Dict[str, Any], *, conv_id: str = ""
    ) -> AsyncIterator[bytes]:
        payload, _client_model = self._official_anthropic_body(body, stream=True)
        logger.info(
            "upstream official convert anthropic->responses model=%s->%s client_stream=true",
            _client_model,
            payload.get("model"),
        )
        async for chunk in self._stream_official_responses_bytes(
            payload, conv_id=conv_id
        ):
            yield chunk

    async def embeddings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """xAI /v1/embeddings — OpenAI-compatible passthrough with per-slot retry.

        Request/response shapes match OpenAI's embeddings API on both sides;
        we forward the body untouched and only intercept transport-level
        failures to feed the cooldown model.
        """
        logger.info(
            "upstream official client=embeddings wire=/embeddings model=%s (passthrough)",
            body.get("model") or "?",
        )
        pool_size = self._pool.size()
        max_attempts = min(pool_size, 3) if pool_size > 0 else 1
        tried: set = set()
        last_error: Optional[UpstreamError] = None

        for attempt in range(max_attempts):
            slot = self._pick_slot(tried)
            if slot is None:
                break
            tried.add(slot.path.name)

            try:
                headers = self._headers_for_slot(slot, endpoint="embeddings")
                base = self._base_for_slot(slot)
                url = self._url(base, "/embeddings")
                resp = await self._http.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    payload = None
                    try:
                        payload = resp.json()
                    except Exception:
                        pass
                    error_code = _parse_error_code(resp.content)
                    self._pool.mark_cooldown(
                        slot,
                        status=resp.status_code,
                        body=resp.text,
                        error_code=error_code,
                    )
                    last_error = UpstreamError(
                        resp.status_code, resp.text, payload
                    )
                    if (
                        attempt + 1 < max_attempts
                        and resp.status_code in _RETRYABLE_STATUS
                    ):
                        continue
                    raise last_error
                self._pool.note_success(slot)
                return resp.json()
            except UpstreamError:
                raise
            except httpx.HTTPError as exc:
                self._pool.mark_cooldown(slot, status=502)
                last_error = UpstreamError(502, str(exc), None)
                if attempt + 1 < max_attempts:
                    continue
                raise last_error from exc

        if last_error is not None:
            raise last_error
        raise UpstreamError(503, "no credentials available", None)
