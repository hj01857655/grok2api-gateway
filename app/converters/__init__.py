"""Converters package — protocol ↔ Chat Completions.

Internal hub format is always OpenAI Chat Completions.

Official Grok OAuth token upstream speaks Responses (/responses), so
chat_to_responses_request / responses_result_to_chat / stream_responses_to_chat
bridge the hub to that path.
"""

from .anthropic import (
    anthropic_to_chat,
    chat_to_anthropic,
    stream_chat_to_anthropic,
)
from .responses import (
    chat_to_responses,
    chat_to_responses_request,
    collect_responses_completed,
    responses_result_to_chat,
    responses_to_chat,
    stream_chat_to_responses,
    stream_responses_to_chat,
)

__all__ = [
    "anthropic_to_chat",
    "chat_to_anthropic",
    "stream_chat_to_anthropic",
    "responses_to_chat",
    "chat_to_responses",
    "stream_chat_to_responses",
    "chat_to_responses_request",
    "responses_result_to_chat",
    "collect_responses_completed",
    "stream_responses_to_chat",
]
