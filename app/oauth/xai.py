"""xAI / Grok CLI OAuth — port of CLIProxyAPI internal/auth/xai.

Device Code flow (RFC 8628) using the public Grok CLI client_id.
This is for **official xAI/Grok accounts only** — not custom models (iamhc/voya),
and not Gemini / OpenAI / Claude / other CPA providers.

Account management + import accept Grok/xAI credentials only.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse

import httpx

logger = logging.getLogger("grok2api.oauth.xai")

# ---------------------------------------------------------------------------
# Constants — keep aligned with CLIProxyAPI internal/auth/xai/types.go
# ---------------------------------------------------------------------------

DEFAULT_API_BASE_URL = "https://api.x.ai/v1"
CLI_CHAT_PROXY_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
ISSUER = "https://auth.x.ai"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
# Public Grok CLI OAuth client ID (same as CPA)
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
# Match Grok Build's OAuth scope so the signed access_token carries
# conversations:read/write — required for xAI's server-side conversation
# store and cache affinity via x-grok-conv-id.
SCOPE = (
    "openid profile email offline_access "
    "grok-cli:access api:access "
    "conversations:read conversations:write"
)
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_POLL_INTERVAL_S = 5.0
MAX_POLL_DURATION_S = 30 * 60
HTTP_TIMEOUT_S = 30.0
REFRESH_LEAD_S = 5 * 60

# Grok CLI identity headers for chat-proxy (aligns with real Grok Build traffic).
XAI_TOKEN_AUTH_HEADER = "X-XAI-Token-Auth"
XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
XAI_CLIENT_VERSION_HEADER = "x-grok-client-version"
XAI_CLIENT_VERSION_VALUE = "0.2.93"
XAI_USER_AGENT = (
    "grok-pager/0.2.93 grok-shell/0.2.93 (windows; x86_64)"
)
# POST /responses only (observed on Grok Build). GET /models does NOT carry these.
XAI_AUTHENTICATE_RESPONSE_HEADER = "x-authenticateresponse"
XAI_AUTHENTICATE_RESPONSE_VALUE = "authenticate-response"
XAI_CLIENT_IDENTIFIER_HEADER = "x-grok-client-identifier"
XAI_CLIENT_IDENTIFIER_VALUE = "grok-pager"

# Canonical provider for this product (single vendor)
PROVIDER = "xai"
PROVIDER_LABEL = "Grok / xAI"

# Only these type values are accepted on import (CPA uses "xai")
_XAI_TYPE_ALIASES = frozenset({"xai", "grok", "xai-oauth", "xai_oauth"})
# Explicit reject list so a Gemini/OpenAI file never sneaks in
_FOREIGN_TYPES = frozenset(
    {
        "gemini",
        "gemini-cli",
        "gemini_cli",
        "openai",
        "codex",
        "claude",
        "anthropic",
        "kimi",
        "antigravity",
        "vertex",
        "vertex-ai",
        "qwen",
        "iflow",
        "github",
        "copilot",
        "deepseek",
        "mistral",
        "azure",
    }
)
# Filenames that clearly belong to other CPA providers
_FOREIGN_NAME_PREFIXES = (
    "gemini-",
    "gemini_cli-",
    "openai-",
    "codex-",
    "claude-",
    "anthropic-",
    "kimi-",
    "antigravity-",
    "vertex-",
    "qwen-",
    "iflow-",
)


class XAIAuthError(Exception):
    """OAuth / device-code / import failures."""


@dataclass
class Discovery:
    device_authorization_endpoint: str
    token_endpoint: str


@dataclass
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str = ""
    expires_in: int = 0
    interval: int = 5
    token_endpoint: str = ""

    @property
    def open_url(self) -> str:
        return (self.verification_uri_complete or self.verification_uri or "").strip()


@dataclass
class TokenData:
    access_token: str
    refresh_token: str = ""
    id_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 0
    expire: str = ""  # RFC3339
    email: str = ""
    subject: str = ""

    def expired_or_near(self, lead_s: float = REFRESH_LEAD_S) -> bool:
        if not self.expire:
            return False
        try:
            exp = datetime.fromisoformat(self.expire.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return (exp.timestamp() - now.timestamp()) <= lead_s
        except Exception:
            return False


@dataclass
class TokenStorage:
    """Persistable credential file (similar to CPA auths/xai-*.json)."""

    type: str = "xai"
    auth_kind: str = "oauth"
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 0
    expired: str = ""
    last_refresh: str = ""
    email: str = ""
    sub: str = ""
    base_url: str = DEFAULT_API_BASE_URL
    token_endpoint: str = ""
    # when False (OAuth default in CPA): chat uses cli-chat-proxy + CLI headers
    using_api: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_token_data(self) -> TokenData:
        return TokenData(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            id_token=self.id_token,
            token_type=self.token_type,
            expires_in=self.expires_in,
            expire=self.expired,
            email=self.email,
            subject=self.sub,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": "xai",  # always canonical
            "auth_kind": self.auth_kind or "oauth",
            "provider": PROVIDER,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expired": self.expired,
            "last_refresh": self.last_refresh,
            "email": self.email,
            "sub": self.sub,
            "base_url": self.base_url,
            "token_endpoint": self.token_endpoint,
            "using_api": self.using_api,
        }
        # do not let extra override type/provider
        extra = {
            k: v
            for k, v in self.extra.items()
            if k not in ("type", "provider", "access_token", "refresh_token")
        }
        d.update(extra)
        d["type"] = "xai"
        d["provider"] = PROVIDER
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenStorage":
        known = {
            "type",
            "auth_kind",
            "access_token",
            "refresh_token",
            "id_token",
            "token_type",
            "expires_in",
            "expired",
            "last_refresh",
            "email",
            "sub",
            "base_url",
            "token_endpoint",
            "using_api",
        }
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in data.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        kwargs.setdefault("type", "xai")
        kwargs.setdefault("auth_kind", "oauth")
        kwargs.setdefault("base_url", DEFAULT_API_BASE_URL)
        if "using_api" in kwargs:
            kwargs["using_api"] = _as_bool(kwargs["using_api"])
        if "expires_in" in kwargs:
            try:
                kwargs["expires_in"] = int(kwargs["expires_in"] or 0)
            except (TypeError, ValueError):
                kwargs["expires_in"] = 0
        # force canonical type
        kwargs["type"] = "xai"
        ts = cls(**{k: v for k, v in kwargs.items() if k != "extra"})
        ts.extra = extra
        return ts


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")


def default_auths_dir() -> Path:
    return Path.home() / ".grok2api" / "auths"


def credential_filename(email: str = "", subject: str = "") -> str:
    email = _sanitize_segment(email)
    if email:
        return f"xai-{email}.json"
    subject = _sanitize_segment(subject)
    if subject:
        return f"xai-{subject}.json"
    return f"xai-{int(time.time() * 1000)}.json"


def _sanitize_segment(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    out = []
    for ch in value:
        if ch.isalnum() or ch in "@._-":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def is_xai_type(type_val: Any) -> bool:
    t = str(type_val or "").strip().lower()
    return (not t) or (t in _XAI_TYPE_ALIASES)


def is_foreign_type(type_val: Any) -> bool:
    t = str(type_val or "").strip().lower()
    return t in _FOREIGN_TYPES


def reject_foreign_filename(path: Union[str, Path]) -> None:
    """Raise if basename looks like another CPA provider credential."""
    name = Path(path).name.lower()
    for prefix in _FOREIGN_NAME_PREFIXES:
        if name.startswith(prefix):
            raise XAIAuthError(
                f"refusing import: filename {Path(path).name!r} looks like a "
                f"non-Grok provider credential (only Grok/xAI is supported)"
            )
    # explicit non-xai names
    if re.match(r"^(gemini|openai|codex|claude|anthropic|kimi|vertex)\b", name):
        raise XAIAuthError(
            f"refusing import: filename {Path(path).name!r} is not a Grok/xAI credential"
        )


def assert_xai_only(data: Dict[str, Any], *, source_hint: str = "") -> None:
    """Hard gate: only Grok/xAI credential objects may pass."""
    if not isinstance(data, dict):
        raise XAIAuthError("credential JSON must be an object")
    type_val = str(data.get("type") or "").strip().lower()
    provider = str(data.get("provider") or "").strip().lower()
    # provider field if present must be xai/grok
    if provider and provider not in _XAI_TYPE_ALIASES and provider not in ("", "xai", "grok"):
        raise XAIAuthError(
            f"refusing: provider={provider!r} is not Grok/xAI "
            f"(only Grok is supported). source={source_hint or '?'}"
        )
    if is_foreign_type(type_val):
        raise XAIAuthError(
            f"refusing: type={type_val!r} is not Grok/xAI "
            f"(only type=xai is supported). source={source_hint or '?'}"
        )
    if type_val and type_val not in _XAI_TYPE_ALIASES:
        raise XAIAuthError(
            f"refusing: unknown type={type_val!r}; "
            f"only xai/grok credentials are supported. source={source_hint or '?'}"
        )


def _validate_oauth_endpoint(raw_url: str, field: str) -> str:
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise XAIAuthError(f"xai discovery {field} is empty")
    if not raw_url.startswith("https://"):
        raise XAIAuthError(f"xai discovery {field} must use https: {raw_url!r}")
    try:
        host = (urlparse(raw_url).hostname or "").lower()
    except Exception as e:
        raise XAIAuthError(f"xai discovery {field} is invalid: {e}") from e
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise XAIAuthError(f"xai discovery {field} host {host!r} is not on x.ai")
    return raw_url


def _parse_jwt_identity(id_token: str) -> Tuple[str, str]:
    if not id_token or id_token.count(".") < 2:
        return "", ""
    payload = id_token.split(".")[1]
    pad = "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + pad)
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return "", ""
    email = str(claims.get("email") or "").strip()
    sub = str(claims.get("sub") or "").strip()
    return email, sub


def _build_token_data(
    access_token: str,
    refresh_token: str = "",
    id_token: str = "",
    token_type: str = "Bearer",
    expires_in: int = 0,
    email: str = "",
    subject: str = "",
) -> TokenData:
    email2, sub2 = _parse_jwt_identity(id_token)
    email = email or email2
    subject = subject or sub2
    expire = ""
    if expires_in and expires_in > 0:
        ts = datetime.now(timezone.utc).timestamp() + expires_in
        expire = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return TokenData(
        access_token=access_token.strip(),
        refresh_token=(refresh_token or "").strip(),
        id_token=(id_token or "").strip(),
        token_type=(token_type or "Bearer").strip() or "Bearer",
        expires_in=int(expires_in or 0),
        expire=expire,
        email=email,
        subject=subject,
    )


def _extract_access_token(data: Dict[str, Any]) -> str:
    """CPA filestore-compatible: top-level or nested token.access_token."""
    at = str(data.get("access_token") or "").strip()
    if at:
        return at
    token = data.get("token")
    if isinstance(token, dict):
        return str(token.get("access_token") or "").strip()
    return ""


def parse_xai_credential(
    data: Dict[str, Any],
    *,
    source_hint: str = "",
) -> TokenStorage:
    """Parse a CPA-compatible auth JSON; **Grok/xAI only**.

    Accepts files shaped like CLIProxyAPI ``auths/xai-*.json``:
    ``type=xai``, access_token/refresh_token, email/sub, base_url, …
    Rejects every other provider.
    """
    assert_xai_only(data, source_hint=source_hint)

    access = _extract_access_token(data)
    refresh = str(data.get("refresh_token") or "").strip()
    if not access and not refresh:
        token = data.get("token")
        if isinstance(token, dict):
            refresh = str(token.get("refresh_token") or "").strip()
    if not access and not refresh:
        raise XAIAuthError(
            f"xai credential missing access_token and refresh_token "
            f"(source={source_hint or '?'})"
        )

    normalized = dict(data)
    if access and not normalized.get("access_token"):
        normalized["access_token"] = access
    if refresh and not normalized.get("refresh_token"):
        normalized["refresh_token"] = refresh
    normalized["type"] = "xai"
    normalized["provider"] = PROVIDER
    if not str(normalized.get("auth_kind") or "").strip():
        normalized["auth_kind"] = "oauth"

    id_token = str(normalized.get("id_token") or "").strip()
    if id_token:
        em, su = _parse_jwt_identity(id_token)
        if em and not str(normalized.get("email") or "").strip():
            normalized["email"] = em
        if su and not str(normalized.get("sub") or "").strip():
            normalized["sub"] = su

    if not normalized.get("expired") and normalized.get("expire"):
        normalized["expired"] = normalized.get("expire")

    for drop in ("path",):
        normalized.pop(drop, None)

    storage = TokenStorage.from_dict(normalized)
    storage.type = "xai"
    if not storage.base_url:
        storage.base_url = DEFAULT_API_BASE_URL
    return storage


def load_xai_credential_file(path: Path) -> TokenStorage:
    path = Path(path)
    if not path.is_file():
        raise XAIAuthError(f"credential file not found: {path}")
    reject_foreign_filename(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise XAIAuthError(f"invalid JSON in {path}: {e}") from e
    if (
        isinstance(data, dict)
        and "path" in data
        and not data.get("access_token")
        and not data.get("refresh_token")
    ):
        target = Path(str(data.get("path") or ""))
        if target.is_file():
            return load_xai_credential_file(target)
        raise XAIAuthError(f"{path.name} is a pointer file but target missing: {target}")
    return parse_xai_credential(data, source_hint=str(path))


def import_credential(
    source: Union[str, Path],
    *,
    auths_dir: Optional[Path] = None,
    using_api: Optional[bool] = None,
    set_current: bool = True,
    refresh_if_needed: bool = False,
) -> Path:
    """Import a single **Grok/xAI** credential JSON into auths dir.

    - Accepts CPA ``xai-*.json`` and our own token files.
    - Rejects Gemini / OpenAI / Claude / … by type **and** filename.
    - Returns destination path under ``auths_dir`` (always named ``xai-*.json``).
    """
    source = Path(source).expanduser()
    if not source.is_file():
        raise XAIAuthError(f"import source is not a file: {source}")
    reject_foreign_filename(source)

    storage = load_xai_credential_file(source)
    if using_api is not None:
        storage.using_api = using_api

    if refresh_if_needed:
        try:
            storage = ensure_fresh_token(storage, auths_dir=auths_dir, save=False)
        except Exception as exc:
            logger.warning("import refresh skipped: %s", exc)

    if set_current:
        return save_token(storage, auths_dir)

    auths_dir = Path(auths_dir or default_auths_dir())
    auths_dir.mkdir(parents=True, exist_ok=True)
    name = credential_filename(storage.email, storage.sub)
    path = auths_dir / name
    path.write_text(
        json.dumps(storage.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return path


def import_credentials(
    source: Union[str, Path],
    *,
    auths_dir: Optional[Path] = None,
    using_api: Optional[bool] = None,
    set_current: bool = True,
) -> List[Path]:
    """Import one file or every **xai** credential under a directory.

    Directory mode only picks ``xai-*.json`` (skips ``xai-current.json``).
    Non-xai files are never imported.
    """
    source = Path(source).expanduser()
    if source.is_file():
        return [
            import_credential(
                source,
                auths_dir=auths_dir,
                using_api=using_api,
                set_current=set_current,
            )
        ]
    if not source.is_dir():
        raise XAIAuthError(f"import source not found: {source}")

    # Only xai-* names — never scan gemini-*/openai-*/claude-* into the list
    candidates = sorted(p for p in source.glob("xai-*.json") if p.name != "xai-current.json")

    if not candidates:
        raise XAIAuthError(
            f"no Grok/xAI credentials found under {source} "
            f"(only xai-*.json / type=xai is supported; other providers are ignored)"
        )

    saved: List[Path] = []
    errors: List[str] = []
    for i, path in enumerate(candidates):
        try:
            is_last = i == len(candidates) - 1
            dest = import_credential(
                path,
                auths_dir=auths_dir,
                using_api=using_api,
                set_current=set_current and is_last,
            )
            saved.append(dest)
        except XAIAuthError as e:
            errors.append(f"{path.name}: {e}")
            logger.warning("skip import %s: %s", path, e)

    if not saved:
        raise XAIAuthError(
            "no Grok credentials imported:\n  " + "\n  ".join(errors or ["(none)"])
        )
    return saved


def list_xai_credentials(auths_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List only valid Grok/xAI credential files under auths dir."""
    auths_dir = Path(auths_dir or default_auths_dir())
    out: List[Dict[str, Any]] = []
    if not auths_dir.is_dir():
        return out
    for p in sorted(auths_dir.glob("xai-*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.name == "xai-current.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            assert_xai_only(data, source_hint=p.name)
            out.append(
                {
                    "name": p.name,
                    "provider": PROVIDER,
                    "provider_label": PROVIDER_LABEL,
                    "type": "xai",
                    "email": data.get("email") or data.get("sub") or "",
                    "has_access": bool(data.get("access_token")),
                    "has_refresh": bool(data.get("refresh_token")),
                    "expired": data.get("expired") or "",
                    "using_api": bool(data.get("using_api")),
                    "mtime": int(p.stat().st_mtime),
                }
            )
        except XAIAuthError:
            logger.warning("skip non-xai file in auths: %s", p.name)
        except Exception:
            logger.warning("skip unreadable auths file: %s", p.name)
    return out


class XAIAuth:
    """Device-code OAuth + refresh for official Grok / xAI accounts."""

    def __init__(self, timeout: float = HTTP_TIMEOUT_S) -> None:
        self._timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, follow_redirects=True)

    def discover(self) -> Discovery:
        with self._client() as client:
            resp = client.get(DISCOVERY_URL, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                raise XAIAuthError(f"xai discovery failed: {resp.status_code} {resp.text[:300]}")
            data = resp.json()
        device_ep = _validate_oauth_endpoint(
            data.get("device_authorization_endpoint") or "",
            "device_authorization_endpoint",
        )
        token_ep = _validate_oauth_endpoint(
            data.get("token_endpoint") or "",
            "token_endpoint",
        )
        return Discovery(
            device_authorization_endpoint=device_ep,
            token_endpoint=token_ep,
        )

    def start_device_flow(self) -> DeviceCode:
        disc = self.discover()
        return self.request_device_code(disc.device_authorization_endpoint, disc.token_endpoint)

    def request_device_code(
        self,
        device_authorization_endpoint: str,
        token_endpoint: str = "",
    ) -> DeviceCode:
        form = urlencode({"client_id": CLIENT_ID, "scope": SCOPE})
        with self._client() as client:
            resp = client.post(
                device_authorization_endpoint.strip(),
                content=form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            body = resp.text
            if resp.status_code != 200:
                raise XAIAuthError(
                    f"xai device code request failed: {resp.status_code} {body[:400]}"
                )
            data = resp.json()

        device_code = (data.get("device_code") or "").strip()
        user_code = (data.get("user_code") or "").strip()
        verification_uri = (data.get("verification_uri") or "").strip()
        verification_uri_complete = (data.get("verification_uri_complete") or "").strip()
        if not device_code:
            raise XAIAuthError("xai device code: response missing device_code")
        if not user_code:
            raise XAIAuthError("xai device code: response missing user_code")
        if not verification_uri and not verification_uri_complete:
            raise XAIAuthError("xai device code: response missing verification URI")

        return DeviceCode(
            device_code=device_code,
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri_complete,
            expires_in=int(data.get("expires_in") or 0),
            interval=int(data.get("interval") or DEFAULT_POLL_INTERVAL_S),
            token_endpoint=(token_endpoint or "").strip(),
        )

    def wait_for_authorization(self, device: DeviceCode) -> TokenData:
        return self.poll_for_token(device)

    def poll_for_token(self, device: DeviceCode) -> TokenData:
        token_endpoint = (device.token_endpoint or "").strip()
        if not token_endpoint:
            token_endpoint = self.discover().token_endpoint

        interval = float(device.interval or DEFAULT_POLL_INTERVAL_S)
        if interval < DEFAULT_POLL_INTERVAL_S:
            interval = DEFAULT_POLL_INTERVAL_S

        deadline = time.time() + MAX_POLL_DURATION_S
        if device.expires_in and device.expires_in > 0:
            deadline = min(deadline, time.time() + device.expires_in)

        first = True
        while True:
            if not first and time.time() > deadline:
                raise XAIAuthError("xai device code expired")
            first = False

            token, err, next_interval, cont = self._exchange_device_code(
                token_endpoint, device.device_code, interval
            )
            if token is not None:
                return token
            if not cont:
                raise XAIAuthError(err or "xai device token failed")
            interval = next_interval
            time.sleep(interval)

    def _exchange_device_code(
        self,
        token_endpoint: str,
        device_code: str,
        interval: float,
    ) -> Tuple[Optional[TokenData], Optional[str], float, bool]:
        form = urlencode(
            {
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "device_code": device_code.strip(),
                "client_id": CLIENT_ID,
            }
        )
        with self._client() as client:
            resp = client.post(
                token_endpoint.strip(),
                content=form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            try:
                data = resp.json()
            except Exception:
                return None, f"xai device token: bad json {resp.text[:300]}", interval, False

        err = (data.get("error") or "").strip()
        if err:
            if err == "authorization_pending":
                return None, None, interval, True
            if err == "slow_down":
                return None, None, interval + DEFAULT_POLL_INTERVAL_S, True
            if err == "expired_token":
                return None, "xai device code expired", interval, False
            if err == "access_denied":
                return None, "xai device authorization denied", interval, False
            desc = (data.get("error_description") or "").strip()
            msg = f"xai device token error: {err}" + (f": {desc}" if desc else "")
            return None, msg, interval, False

        if resp.status_code != 200:
            return (
                None,
                f"xai device token request failed: {resp.status_code}",
                interval,
                False,
            )

        access = (data.get("access_token") or "").strip()
        if not access:
            return None, "xai device token response missing access_token", interval, False

        td = _build_token_data(
            access_token=access,
            refresh_token=data.get("refresh_token") or "",
            id_token=data.get("id_token") or "",
            token_type=data.get("token_type") or "Bearer",
            expires_in=int(data.get("expires_in") or 0),
        )
        return td, None, interval, False

    def refresh_tokens(
        self,
        refresh_token: str,
        token_endpoint: str = "",
    ) -> TokenData:
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            raise XAIAuthError("xai token refresh: refresh token is required")
        if not (token_endpoint or "").strip():
            token_endpoint = self.discover().token_endpoint
        form = urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            }
        )
        with self._client() as client:
            resp = client.post(
                token_endpoint.strip(),
                content=form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            body = resp.text
            if resp.status_code != 200:
                raise XAIAuthError(
                    f"xai token refresh failed: {resp.status_code} {body[:400]}"
                )
            data = resp.json()
        access = (data.get("access_token") or "").strip()
        if not access:
            raise XAIAuthError("xai token refresh response missing access_token")
        new_refresh = (data.get("refresh_token") or "").strip() or refresh_token
        return _build_token_data(
            access_token=access,
            refresh_token=new_refresh,
            id_token=data.get("id_token") or "",
            token_type=data.get("token_type") or "Bearer",
            expires_in=int(data.get("expires_in") or 0),
        )

    def create_token_storage(
        self,
        token: TokenData,
        *,
        token_endpoint: str = "",
        base_url: str = DEFAULT_API_BASE_URL,
        using_api: bool = False,
    ) -> TokenStorage:
        return TokenStorage(
            type="xai",
            auth_kind="oauth",
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            id_token=token.id_token,
            token_type=token.token_type,
            expires_in=token.expires_in,
            expired=token.expire,
            last_refresh=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            email=token.email,
            sub=token.subject,
            base_url=base_url or DEFAULT_API_BASE_URL,
            token_endpoint=token_endpoint,
            using_api=using_api,
        )


def save_token(storage: TokenStorage, auths_dir: Optional[Path] = None) -> Path:
    storage.type = "xai"
    auths_dir = Path(auths_dir or default_auths_dir())
    auths_dir.mkdir(parents=True, exist_ok=True)
    name = credential_filename(storage.email, storage.sub)
    # never write non-xai filenames
    if not name.startswith("xai-"):
        name = f"xai-{name}"
    path = auths_dir / name
    path.write_text(
        json.dumps(storage.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except Exception:
        pass
    current = auths_dir / "xai-current.json"
    current.write_text(
        json.dumps(
            {
                "path": str(path),
                "email": storage.email,
                "sub": storage.sub,
                "provider": PROVIDER,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_token(
    path: Optional[Path] = None,
    auths_dir: Optional[Path] = None,
) -> Optional[TokenStorage]:
    """Load current Grok/xAI credential; skip/reject non-xai files."""
    auths_dir = Path(auths_dir or default_auths_dir())
    if path is None:
        current = auths_dir / "xai-current.json"
        if current.is_file():
            try:
                meta = json.loads(current.read_text(encoding="utf-8"))
                p = Path(meta.get("path") or "")
                if p.is_file():
                    path = p
            except Exception:
                path = None
        if path is None:
            files = sorted(
                auths_dir.glob("xai-*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            files = [f for f in files if f.name != "xai-current.json"]
            path = files[0] if files else None
    if path is None or not Path(path).is_file():
        return None
    try:
        reject_foreign_filename(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert_xai_only(data, source_hint=str(path))
        return TokenStorage.from_dict(data)
    except XAIAuthError as e:
        logger.warning("load_token rejected non-xai credential %s: %s", path, e)
        return None
    except Exception as e:
        logger.warning("load_token failed for %s: %s", path, e)
        return None


# xAI rotates the refresh_token on every /oauth2/token call; two concurrent
# refreshes race and the loser gets 400 invalid_grant. Serialize with a
# module-level lock (CPA does the equivalent with singleflight.Group). Same
# process only — multi-worker deployments would need a file lock.
_refresh_lock = threading.Lock()


def ensure_fresh_token(
    storage: TokenStorage,
    *,
    auth: Optional[XAIAuth] = None,
    auths_dir: Optional[Path] = None,
    save: bool = True,
) -> TokenStorage:
    """Refresh access_token if near expiry; return updated storage.

    Thread-safe: serialized via ``_refresh_lock`` with double-checked expiry
    so waiters returning after another thread already refreshed skip the
    redundant HTTP round-trip and reload the freshly written credential.
    """
    td = storage.to_token_data()
    if not td.expired_or_near() and storage.access_token:
        return storage
    if not storage.refresh_token:
        raise XAIAuthError("token expired and no refresh_token")

    with _refresh_lock:
        # Re-check after acquiring: a peer may have refreshed while we waited.
        # Reload from disk so we pick up the rotated refresh_token they wrote.
        latest = load_token(auths_dir=auths_dir)
        if latest and latest.access_token and not latest.to_token_data().expired_or_near():
            return latest
        current = latest or storage
        if not current.refresh_token:
            raise XAIAuthError("token expired and no refresh_token")

        auth = auth or XAIAuth()
        new_td = auth.refresh_tokens(current.refresh_token, current.token_endpoint)
        current.access_token = new_td.access_token
        if new_td.refresh_token:
            current.refresh_token = new_td.refresh_token
        current.id_token = new_td.id_token or current.id_token
        current.expires_in = new_td.expires_in
        current.expired = new_td.expire
        current.last_refresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if new_td.email:
            current.email = new_td.email
        if new_td.subject:
            current.sub = new_td.subject
        current.type = "xai"
        if save:
            save_token(current, auths_dir)
        return current


def resolve_chat_base_url(storage: TokenStorage) -> str:
    """CPA xaiChatBaseURL logic."""
    base = (storage.base_url or "").strip().rstrip("/")
    if storage.using_api:
        return base or DEFAULT_API_BASE_URL
    default = DEFAULT_API_BASE_URL.rstrip("/")
    if base and base != default:
        return base
    return CLI_CHAT_PROXY_BASE_URL


def _new_traceparent() -> str:
    """W3C traceparent: ``00-<32-hex trace-id>-<16-hex span-id>-01`` (sampled)."""
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def oauth_request_headers(
    storage: TokenStorage,
    *,
    endpoint: str = "responses",
    session_id: str = "",
) -> Dict[str, str]:
    """Headers for xAI calls with OAuth token, per observed Grok Build traffic.

    ``endpoint`` picks the header signature (Build sends a different set per
    endpoint kind on ``cli-chat-proxy.grok.com``):

      - ``responses``  — POST /responses (SSE): full tracing + identifier
      - ``models``     — GET /models: user-identity binding (x-userid/x-email)
      - ``embeddings`` — POST /embeddings: CLI identity only, nothing else

    Wrong-shape headers still work (upstream ignores extras) but exact
    parity improves cache affinity and rate-bucket routing.
    """
    is_sse = endpoint == "responses"
    headers = {
        "Authorization": f"Bearer {storage.access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_sse else "application/json",
        "Connection": "Keep-Alive",
    }
    if session_id:
        headers["x-grok-conv-id"] = session_id
    if storage.using_api:
        return headers
    base = resolve_chat_base_url(storage)
    if base.rstrip("/") != CLI_CHAT_PROXY_BASE_URL.rstrip("/"):
        return headers

    # CLI-proxy identity — shared across all endpoints on cli-chat-proxy.
    headers[XAI_TOKEN_AUTH_HEADER] = XAI_TOKEN_AUTH_VALUE
    headers[XAI_CLIENT_VERSION_HEADER] = XAI_CLIENT_VERSION_VALUE
    headers["User-Agent"] = XAI_USER_AGENT

    if endpoint == "responses":
        # POST /responses — Build sends full tracing + identifier set.
        headers[XAI_AUTHENTICATE_RESPONSE_HEADER] = XAI_AUTHENTICATE_RESPONSE_VALUE
        headers[XAI_CLIENT_IDENTIFIER_HEADER] = XAI_CLIENT_IDENTIFIER_VALUE
        headers["x-grok-req-id"] = str(uuid.uuid4())
        headers["traceparent"] = _new_traceparent()
    elif endpoint == "models":
        # GET /models — Build binds user identity here.
        if storage.sub:
            headers["x-userid"] = storage.sub
        if storage.email:
            headers["x-email"] = storage.email
    # endpoint == "embeddings" — stop at CLI identity; Build sends nothing else.
    return headers


def interactive_login(
    *,
    auths_dir: Optional[Path] = None,
    open_browser: bool = True,
    using_api: bool = False,
) -> Path:
    """Full device-code login; blocks until user authorizes. Returns credential path."""
    auth = XAIAuth()
    print("Starting xAI / Grok OAuth (device code)...")
    print(f"  provider: {PROVIDER_LABEL} only")
    print(f"  discovery: {DISCOVERY_URL}")
    print(f"  client_id: {CLIENT_ID}")

    disc = auth.discover()
    print(f"  token_endpoint: {disc.token_endpoint}")

    device = auth.request_device_code(
        disc.device_authorization_endpoint,
        disc.token_endpoint,
    )
    url = device.open_url
    print()
    print("To authenticate, open:")
    print(f"  {url}")
    if device.user_code:
        print(f"Then enter code: {device.user_code}")
    print()

    if open_browser and url:
        try:
            import webbrowser

            webbrowser.open(url)
            print("Browser open attempted.")
        except Exception as e:
            print(f"(could not open browser: {e})")

    print("Waiting for authorization...")
    if device.expires_in:
        print(f"(timeout in ~{device.expires_in}s if not authorized)")

    token = auth.wait_for_authorization(device)
    storage = auth.create_token_storage(
        token,
        token_endpoint=disc.token_endpoint,
        base_url=DEFAULT_API_BASE_URL,
        using_api=using_api,
    )
    path = save_token(storage, auths_dir)
    label = storage.email or storage.sub or path.name
    print()
    print(f"xAI authentication successful: {label}")
    print(f"Saved to: {path}")
    print(f"using_api={storage.using_api}  chat_base={resolve_chat_base_url(storage)}")
    return path
