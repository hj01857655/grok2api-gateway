"""End-to-end mock test for xAI Device Code OAuth (RFC 8628).

Runs the full flow against an in-process ``httpx.MockTransport``, so it works
offline and does not touch real xAI servers:

  1. Discovery                (well-known / openid-configuration)
  2. Device code request      (POST device_authorization_endpoint)
  3. Token poll               (authorization_pending → success)
  4. Save credential to disk  (xai-*.json + xai-current.json)
  5. Reload credential
  6. Refresh access token     (refresh_token grant)
  7. Foreign-type rejection   (gemini / anthropic credentials refused)
  8. Import xai-*.json file   (CPA-compatible)

Usage:
  .\.venv\Scripts\python.exe scripts\test_oauth_flow.py
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

import httpx

# Make ``app.*`` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.oauth import xai  # noqa: E402
from app.oauth.xai import (  # noqa: E402
    CLIENT_ID,
    DISCOVERY_URL,
    XAIAuth,
    XAIAuthError,
    import_credential,
    load_token,
    parse_xai_credential,
    save_token,
)

DEVICE_AUTHORIZATION_ENDPOINT = "https://auth.x.ai/oauth2/device_authorization"
TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"


def _make_id_token(email: str, sub: str) -> str:
    """Craft a minimal unsigned JWT — only the payload is read by _parse_jwt_identity."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"email": email, "sub": sub}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


class MockOAuthServer:
    """Scripted responses for discovery, device authorize, poll, refresh."""

    def __init__(self) -> None:
        self.token_polls = 0
        self.refresh_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)

        if url == DISCOVERY_URL:
            return httpx.Response(
                200,
                json={
                    "device_authorization_endpoint": DEVICE_AUTHORIZATION_ENDPOINT,
                    "token_endpoint": TOKEN_ENDPOINT,
                },
            )

        body = request.content.decode() if request.content else ""

        if url == DEVICE_AUTHORIZATION_ENDPOINT:
            assert f"client_id={CLIENT_ID}" in body, "device authorize missing client_id"
            assert "scope=openid" in body, "device authorize missing scope"
            return httpx.Response(
                200,
                json={
                    "device_code": "dev-code-xyz",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.x.ai/device",
                    "verification_uri_complete": "https://auth.x.ai/device?code=ABCD-1234",
                    "expires_in": 600,
                    "interval": 1,
                },
            )

        if url == TOKEN_ENDPOINT:
            if "grant_type=refresh_token" in body:
                self.refresh_calls += 1
                return httpx.Response(
                    200,
                    json={
                        "access_token": f"access-refreshed-{self.refresh_calls}",
                        "refresh_token": "refresh-2",
                        "id_token": _make_id_token("user@example.com", "sub-42"),
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                )
            # device_code grant (grant_type=urn%3A...%3Adevice_code)
            self.token_polls += 1
            if self.token_polls == 1:
                return httpx.Response(200, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={
                    "access_token": "access-initial",
                    "refresh_token": "refresh-1",
                    "id_token": _make_id_token("user@example.com", "sub-42"),
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )

        return httpx.Response(404, json={"error": "not_found", "url": url})


def _install_mock(server: MockOAuthServer) -> None:
    """Route XAIAuth's internal httpx.Client through MockTransport."""
    transport = httpx.MockTransport(server.handler)

    def _client(self: XAIAuth) -> httpx.Client:
        return httpx.Client(transport=transport, timeout=5.0)

    XAIAuth._client = _client  # type: ignore[method-assign]

    # Avoid real sleeps between poll attempts.
    xai.time.sleep = lambda *_a, **_kw: None  # type: ignore[attr-defined]


def _line(title: str) -> None:
    print(f"\n---- {title} ----")


def main() -> int:
    server = MockOAuthServer()
    _install_mock(server)
    auth = XAIAuth(timeout=5.0)

    _line("1. Discovery")
    disc = auth.discover()
    print(f"  device_authorization_endpoint = {disc.device_authorization_endpoint}")
    print(f"  token_endpoint                = {disc.token_endpoint}")
    assert disc.device_authorization_endpoint == DEVICE_AUTHORIZATION_ENDPOINT
    assert disc.token_endpoint == TOKEN_ENDPOINT

    _line("2. Request device code")
    device = auth.request_device_code(
        disc.device_authorization_endpoint, disc.token_endpoint
    )
    print(f"  user_code  = {device.user_code}")
    print(f"  open_url   = {device.open_url}")
    print(f"  interval   = {device.interval}s   expires_in = {device.expires_in}s")
    assert device.device_code == "dev-code-xyz"

    _line("3. Poll for token (authorization_pending -> success)")
    token = auth.poll_for_token(device)
    print(f"  polls            = {server.token_polls}   (1 pending + 1 success)")
    print(f"  access_token     = {token.access_token}")
    print(f"  refresh_token    = {token.refresh_token}")
    print(f"  email (from JWT) = {token.email}")
    print(f"  sub   (from JWT) = {token.subject}")
    print(f"  expire (ISO)     = {token.expire}")
    assert token.access_token == "access-initial"
    assert token.email == "user@example.com"
    assert token.subject == "sub-42"
    assert server.token_polls == 2

    with tempfile.TemporaryDirectory() as td:
        auths_dir = Path(td)

        _line("4. Save credential to disk")
        storage = auth.create_token_storage(token, token_endpoint=disc.token_endpoint)
        path = save_token(storage, auths_dir)
        print(f"  saved       = {path.name}")
        print(f"  current.json = {(auths_dir / 'xai-current.json').is_file()}")
        assert path.name.startswith("xai-") and path.name.endswith(".json")
        assert (auths_dir / "xai-current.json").is_file()

        _line("5. Load credential back via xai-current.json pointer")
        loaded = load_token(auths_dir=auths_dir)
        assert loaded is not None
        print(f"  email        = {loaded.email}")
        print(f"  has_refresh  = {bool(loaded.refresh_token)}")
        print(f"  using_api    = {loaded.using_api}   (False -> cli-chat-proxy)")
        assert loaded.access_token == token.access_token
        assert loaded.refresh_token == token.refresh_token

        _line("6. Refresh access token")
        new_td = auth.refresh_tokens(
            loaded.refresh_token, token_endpoint=disc.token_endpoint
        )
        print(f"  refresh calls    = {server.refresh_calls}")
        print(f"  new access_token = {new_td.access_token}")
        print(f"  new refresh      = {new_td.refresh_token}")
        assert new_td.access_token == "access-refreshed-1"
        assert new_td.refresh_token == "refresh-2"

        _line("7. Foreign-type rejection")
        for foreign in ("gemini", "anthropic", "openai"):
            try:
                parse_xai_credential({"type": foreign, "access_token": "x"})
            except XAIAuthError as e:
                print(f"  rejected type={foreign!r}: {e}")
            else:
                raise AssertionError(f"{foreign!r} credential must be rejected")

        _line("8. Import an xai-*.json credential file")
        with tempfile.TemporaryDirectory() as src_dir:
            src = Path(src_dir) / "xai-import.json"
            src.write_text(
                json.dumps(
                    {
                        "type": "xai",
                        "access_token": "imported-access",
                        "refresh_token": "imported-refresh",
                        "email": "imported@example.com",
                        "expired": "2099-01-01T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            imported = import_credential(src, auths_dir=auths_dir)
            print(f"  imported -> {imported.name}")
            current = load_token(auths_dir=auths_dir)
            assert current is not None
            print(f"  current email = {current.email}")
            assert current.email == "imported@example.com"

    print("\nOK: OAuth flow mock test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
