"""Converters package — protocol ↔ Chat Completions.

Internal hub format is always OpenAI Chat Completions.
"""

from .anthropic import (
    anthropic_to_chat,
    chat_to_anthropic,
    stream_chat_to_anthropic,
)
from .responses import (
    chat_to_responses,
    responses_to_chat,
    stream_chat_to_responses,
)

__all__ = [
    "anthropic_to_chat",
    "chat_to_anthropic",
    "stream_chat_to_anthropic",
    "responses_to_chat",
    "chat_to_responses",
    "stream_chat_to_responses",
]
