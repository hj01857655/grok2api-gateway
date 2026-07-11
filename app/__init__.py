"""Grok2API — self-built three-protocol gateway.

Client protocols (what Grok Build custom models speak):
  - OpenAI Chat Completions   POST /v1/chat/completions
  - OpenAI Responses          POST /v1/responses
  - Anthropic Messages        POST /v1/messages

All convert to internal Chat Completions → upstream OpenAI-compatible
(xAI official / NewAPI / VoyA), then convert the response back.

Architecture inspired by chenyme/grok2api products layer; upstream is
API/BYOK only (no web reverse, no CLIProxyAPI).
"""

__version__ = "0.2.0"
