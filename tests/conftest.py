"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app import channel_store
from app.config import reload_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Avoid cached Settings leaking between tests that touch env."""
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def client_key(monkeypatch, tmp_path):
    """Gateway door key + one admin-added mid-station channel (not from env)."""
    key = "sk-test-gateway-key"
    data = tmp_path / "g2a-data"
    monkeypatch.setenv("GROK2API_API_KEY", key)
    monkeypatch.setenv("UPSTREAM_MODE", "compat")
    monkeypatch.setenv("GROK2API_DATA_DIR", str(data))
    monkeypatch.setenv("DEFAULT_MODELS", "test-model")
    # Ensure legacy env keys cannot invent channels
    for name in (
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "VOYA_API_KEY",
        "OPENAI_COMPATIBILITY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    channel_store.add_provider(
        name="test-channel",
        base_url="https://example.test/v1",
        api_key="upstream-key",
        models="test-model",
        prefix="test",
        root=data,
    )
    reload_settings()
    return key
