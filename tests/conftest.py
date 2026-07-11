"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

from app.config import reload_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Avoid cached Settings leaking between tests that touch env."""
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def client_key(monkeypatch):
    key = "sk-test-gateway-key"
    monkeypatch.setenv("GROK2API_API_KEY", key)
    monkeypatch.setenv("UPSTREAM_MODE", "compat")
    monkeypatch.setenv("XAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("XAI_API_KEY", "upstream-key")
    monkeypatch.setenv("DEFAULT_MODELS", "test-model")
    monkeypatch.delenv("OPENAI_COMPATIBILITY", raising=False)
    reload_settings()
    return key
