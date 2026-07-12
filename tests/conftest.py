"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.config import reload_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Avoid cached Settings leaking between tests that touch env."""
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def client_key(monkeypatch, tmp_path):
    """Gateway door key + isolated data dir (official Grok only; no mid-station)."""
    key = "sk-test-gateway-key"
    data = tmp_path / "g2a-data"
    monkeypatch.setenv("GROK2API_API_KEY", key)
    monkeypatch.setenv("UPSTREAM_MODE", "oauth")
    monkeypatch.setenv("GROK2API_DATA_DIR", str(data))
    monkeypatch.setenv("DEFAULT_MODELS", "test-model")
    for name in (
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "VOYA_API_KEY",
        "OPENAI_COMPATIBILITY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    reload_settings()
    return key
