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


@pytest.fixture(autouse=True)
def _reset_upstream_singleton():
    """Drop the cached UpstreamClient so per-test `patch(...)` mocks apply.

    handlers._client() caches one UpstreamClient at module scope for perf; in
    tests each case patches app.handlers.UpstreamClient inside a `with` block
    and expects the next _client() call to build a fresh (mocked) instance.
    Without this reset the first test's real client leaks into later mocks.
    """
    from app.handlers import reset_upstream_client

    reset_upstream_client()
    yield
    reset_upstream_client()


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
