"""Merge Grok2API custom models into ~/.grok/config.toml (optional helper).

Usage:
  python scripts/patch_grok_config.py
  python scripts/patch_grok_config.py --key sk-your-gateway-key

Does not overwrite existing [model.g2a-*] blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

INSERT_TEMPLATE = """
# --- Local Grok2API three protocols ---
# Start: cd <this-repo>; .\\start.ps1
# Client key must match gateway GROK2API_API_KEY

[model.g2a-chat]
model = "grok-4.5"
base_url = "http://127.0.0.1:8787/v1"
name = "Grok2API Chat"
description = "Local gateway Chat Completions"
api_key = "{api_key}"
api_backend = "chat_completions"
context_window = 200000

[model.g2a-responses]
model = "grok-4.5"
base_url = "http://127.0.0.1:8787/v1"
name = "Grok2API Responses"
description = "Local gateway Responses API"
api_key = "{api_key}"
api_backend = "responses"
context_window = 200000

[model.g2a-claude]
model = "grok-4.5"
base_url = "http://127.0.0.1:8787/v1"
name = "Grok2API as Claude"
description = "Local gateway Anthropic Messages"
api_key = "{api_key}"
api_backend = "messages"
context_window = 200000
extra_headers = {{ "anthropic-version" = "2023-06-01" }}

"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".grok" / "config.toml",
        help="Path to Grok Build config.toml",
    )
    p.add_argument(
        "--key",
        default="sk-change-me",
        help="GROK2API_API_KEY / client api_key for g2a-* models",
    )
    args = p.parse_args()
    dst: Path = args.config.expanduser()
    if not dst.is_file():
        raise SystemExit(f"config not found: {dst}")

    text = dst.read_text(encoding="utf-8")
    if "[model.g2a-chat]" in text:
        print("g2a models already present — skip insert")
    else:
        insert = INSERT_TEMPLATE.format(api_key=args.key)
        marker = "# --- Optional: second custom models (uncomment & fill) ---"
        if marker in text:
            text = text.replace(marker, insert + marker, 1)
        else:
            text = text.rstrip() + "\n" + insert
        dst.write_text(text, encoding="utf-8")
        print(f"inserted g2a models into {dst}")

    text = dst.read_text(encoding="utf-8")
    for k in (
        "[model.g2a-chat]",
        "[model.g2a-responses]",
        "[model.g2a-claude]",
    ):
        print(f"{k}: {'OK' if k in text else 'MISSING'}")


if __name__ == "__main__":
    main()
