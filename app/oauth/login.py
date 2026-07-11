"""CLI: python -m app.oauth.login

Grok / xAI only:
  - Device Code OAuth (same as CLIProxyAPI -xai-login)
  - Import CPA-compatible xai-*.json credentials

Does NOT touch custom-model / iamhc settings.
Does NOT import Gemini / OpenAI / Claude / other providers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .xai import (
    DEFAULT_API_BASE_URL,
    XAIAuthError,
    default_auths_dir,
    import_credentials,
    interactive_login,
    load_token,
    resolve_chat_base_url,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Grok/xAI OAuth: device-code login or import CPA xai-*.json credentials. "
            "Other providers (Gemini/OpenAI/…) are rejected."
        )
    )
    p.add_argument(
        "--auths-dir",
        type=Path,
        default=None,
        help=f"Where to save tokens (default: {default_auths_dir()})",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser automatically (login only)",
    )
    p.add_argument(
        "--using-api",
        action="store_true",
        help="Use official api.x.ai instead of cli-chat-proxy for chat",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print current saved credential and exit",
    )
    p.add_argument(
        "--import",
        dest="import_path",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Import Grok/xAI credential JSON (CPA xai-*.json) or a directory of them. "
            "Only type=xai is accepted."
        ),
    )
    args = p.parse_args(argv)

    if args.status:
        ts = load_token(auths_dir=args.auths_dir)
        if not ts:
            print("No saved xAI OAuth credential.")
            print(f"Auths dir: {args.auths_dir or default_auths_dir()}")
            return 1
        print("Saved xAI OAuth credential:")
        print(f"  email:     {ts.email or '(none)'}")
        print(f"  sub:       {ts.sub or '(none)'}")
        print(f"  expired:   {ts.expired or '(unknown)'}")
        print(f"  using_api: {ts.using_api}")
        print(f"  base_url:  {ts.base_url or DEFAULT_API_BASE_URL}")
        print(f"  chat_base: {resolve_chat_base_url(ts)}")
        print(f"  has_access:  {bool(ts.access_token)}")
        print(f"  has_refresh: {bool(ts.refresh_token)}")
        return 0

    if args.import_path is not None:
        try:
            paths = import_credentials(
                args.import_path,
                auths_dir=args.auths_dir,
                using_api=True if args.using_api else None,
                set_current=True,
            )
        except XAIAuthError as e:
            print(f"Import failed: {e}", file=sys.stderr)
            return 1
        print(f"Imported {len(paths)} Grok/xAI credential(s):")
        for path in paths:
            print(f"  {path}")
        ts = load_token(auths_dir=args.auths_dir)
        if ts:
            print(f"Current: {ts.email or ts.sub or '(unnamed)'}")
            print(f"  using_api={ts.using_api}  chat_base={resolve_chat_base_url(ts)}")
        print()
        print("Next: set UPSTREAM_MODE=credential in .env to use the imported account.")
        print("Custom model iamhc/voya is separate — import does not change it.")
        return 0

    path = interactive_login(
        auths_dir=args.auths_dir,
        open_browser=not args.no_browser,
        using_api=args.using_api,
    )
    print()
    print("Next: set UPSTREAM_MODE=oauth in .env to use this Device Code account.")
    print("Custom model iamhc/voya is separate — OAuth does not change it.")
    print(f"Credential: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
