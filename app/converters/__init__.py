"""Converters package (official Grok /responses wire).

Anthropic ↔ Responses (direct, no Chat hop):
  anthropic_to_responses_request / responses_result_to_anthropic /
    stream_responses_to_anthropic

Chat ↔ Responses (Chat client bridge):
  chat_to_responses_request / responses_result_to_chat / stream_responses_to_chat
  prepare_official_responses_request               (native Responses client)
"""

from .anthropic import (
    anthropic_to_responses_request,
    responses_result_to_anthropic,
    stream_responses_to_anthropic,
)
from .responses import (
    chat_to_responses,
    chat_to_responses_request,
    collect_responses_completed,
    prepare_official_responses_request,
    responses_result_to_chat,
    responses_to_chat,
    stream_chat_to_responses,
    stream_responses_to_chat,
)

__all__ = [
    "anthropic_to_responses_request",
    "responses_result_to_anthropic",
    "stream_responses_to_anthropic",
    "responses_to_chat",
    "chat_to_responses",
    "stream_chat_to_responses",
    "chat_to_responses_request",
    "prepare_official_responses_request",
    "responses_result_to_chat",
    "collect_responses_completed",
    "stream_responses_to_chat",
]
