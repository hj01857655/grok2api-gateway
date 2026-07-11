"""OAuth package — official Grok/xAI accounts only (not custom BYOK).

Single vendor: Grok / xAI. No Gemini / OpenAI / Claude account import.
"""

from .xai import (
    CLI_CHAT_PROXY_BASE_URL,
    CLIENT_ID,
    DEFAULT_API_BASE_URL,
    DISCOVERY_URL,
    PROVIDER,
    PROVIDER_LABEL,
    XAIAuth,
    XAIAuthError,
    assert_xai_only,
    ensure_fresh_token,
    import_credential,
    import_credentials,
    interactive_login,
    list_xai_credentials,
    load_token,
    oauth_request_headers,
    parse_xai_credential,
    reject_foreign_filename,
    resolve_chat_base_url,
    save_token,
)

__all__ = [
    "CLI_CHAT_PROXY_BASE_URL",
    "CLIENT_ID",
    "DEFAULT_API_BASE_URL",
    "DISCOVERY_URL",
    "PROVIDER",
    "PROVIDER_LABEL",
    "XAIAuth",
    "XAIAuthError",
    "assert_xai_only",
    "ensure_fresh_token",
    "import_credential",
    "import_credentials",
    "interactive_login",
    "list_xai_credentials",
    "load_token",
    "oauth_request_headers",
    "parse_xai_credential",
    "reject_foreign_filename",
    "resolve_chat_base_url",
    "save_token",
]
