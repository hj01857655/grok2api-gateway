"""Credential pool unit tests — cooldown mapping + round-robin selection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.credential_pool import (
    COOLDOWN_AUTH_ERROR,
    COOLDOWN_GENERIC,
    COOLDOWN_QUOTA_EXHAUSTED,
    COOLDOWN_RATE_LIMIT_BASE,
    CredentialPool,
    _cooldown_for,
)


def _write_credential(auths: Path, email: str, refresh: str = "r") -> Path:
    """Write a minimal but schema-valid xai-*.json into ``auths``."""
    auths.mkdir(parents=True, exist_ok=True)
    path = auths / f"xai-{email}.json"
    path.write_text(
        json.dumps(
            {
                "type": "xai",
                "email": email,
                "access_token": f"at-{email}",
                "refresh_token": refresh,
                "expired": "2099-01-01T00:00:00Z",
                "sub": f"sub-{email}",
                "using_api": False,
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Cooldown reason → duration mapping
# ---------------------------------------------------------------------------

def test_cooldown_quota_exhausted_maps_to_24h():
    seconds, reason = _cooldown_for(
        429, "subscription:free-usage-exhausted", "You've used all…"
    )
    assert seconds == COOLDOWN_QUOTA_EXHAUSTED
    assert reason == "quota_exhausted"


def test_cooldown_rate_limit_maps_to_60s_base():
    seconds, reason = _cooldown_for(
        429, "resource-exhausted", "Too many requests per minute"
    )
    assert seconds == COOLDOWN_RATE_LIMIT_BASE
    assert reason == "rate_limit"


def test_cooldown_auth_error_maps_to_5min():
    seconds, reason = _cooldown_for(401, "", "invalid token")
    assert seconds == COOLDOWN_AUTH_ERROR
    assert reason == "auth_error"


def test_cooldown_generic_429_default():
    """Unknown 429 code — treat as short rate_limit (fixed 60s window)."""
    seconds, reason = _cooldown_for(429, "unknown-code", "")
    assert seconds == COOLDOWN_GENERIC
    assert reason == "rate_limit"


def test_cooldown_5xx_short():
    seconds, reason = _cooldown_for(502, "", "bad gateway")
    assert seconds == COOLDOWN_GENERIC
    assert reason == "server_error"


def test_cooldown_2xx_none():
    seconds, reason = _cooldown_for(200, "", "")
    assert seconds == 0
    assert reason == ""


# ---------------------------------------------------------------------------
# Round-robin + cooldown skip
# ---------------------------------------------------------------------------

def test_empty_dir_returns_no_slots(tmp_path):
    pool = CredentialPool(tmp_path)
    assert pool.size() == 0
    assert pool.pick() is None
    assert pool.pick_any() is None


def test_round_robin_cycles_across_slots(tmp_path):
    _write_credential(tmp_path, "a@x")
    _write_credential(tmp_path, "b@x")
    _write_credential(tmp_path, "c@x")
    pool = CredentialPool(tmp_path)
    assert pool.size() == 3

    seen = [pool.pick().email for _ in range(6)]
    # Every credential shows up twice across six picks
    assert sorted(seen) == sorted(["a@x", "b@x", "c@x"] * 2)


def test_cooldown_slot_is_skipped(tmp_path):
    _write_credential(tmp_path, "a@x")
    _write_credential(tmp_path, "b@x")
    pool = CredentialPool(tmp_path)

    first = pool.pick()
    pool.mark_cooldown(
        first, status=429, error_code="subscription:free-usage-exhausted"
    )

    # Next 5 picks must all be the OTHER slot
    other = "b@x" if first.email == "a@x" else "a@x"
    for _ in range(5):
        pick = pool.pick()
        assert pick.email == other


def test_all_cooldown_returns_pick_any_fallback(tmp_path):
    slot_a = _write_credential(tmp_path, "a@x")
    slot_b = _write_credential(tmp_path, "b@x")
    pool = CredentialPool(tmp_path)

    for slot in pool._slots:
        pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")

    # pick() honours cooldown — must return None
    assert pool.pick() is None
    # pick_any() ignores cooldown and returns the soonest-recovering slot
    fallback = pool.pick_any()
    assert fallback is not None
    assert fallback.email in ("a@x", "b@x")


def test_success_clears_cooldown_and_backoff(tmp_path):
    _write_credential(tmp_path, "a@x")
    pool = CredentialPool(tmp_path)
    slot = pool.pick()

    pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")
    assert slot.cooldown_until > time.time()
    assert slot.backoff_level == 1

    pool.note_success(slot)
    assert slot.cooldown_until == 0.0
    assert slot.backoff_level == 0


def test_rate_limit_backoff_doubles(tmp_path):
    _write_credential(tmp_path, "a@x")
    pool = CredentialPool(tmp_path)
    slot = pool._slots[0]

    now_before = time.time()
    pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")
    first_dur = slot.cooldown_until - now_before

    pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")
    second_dur = slot.cooldown_until - time.time()

    # Second cooldown is roughly 2× the first (60s → 120s)
    assert second_dur > first_dur * 1.5


# ---------------------------------------------------------------------------
# Reload preserves cooldown; introspection
# ---------------------------------------------------------------------------

def test_reload_preserves_cooldown_for_existing_slot(tmp_path):
    _write_credential(tmp_path, "a@x")
    pool = CredentialPool(tmp_path)
    slot = pool.pick()
    pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")
    cooldown_before = slot.cooldown_until
    assert cooldown_before > 0

    # Add a new credential and reload — old cooldown must survive
    _write_credential(tmp_path, "b@x")
    pool.reload()

    assert pool.size() == 2
    same_slot = pool.get_by_email("a@x")
    assert same_slot is not None
    assert same_slot.cooldown_until == cooldown_before


def test_reload_drops_removed_credentials(tmp_path):
    path_a = _write_credential(tmp_path, "a@x")
    _write_credential(tmp_path, "b@x")
    pool = CredentialPool(tmp_path)
    assert pool.size() == 2

    path_a.unlink()
    pool.reload()

    assert pool.size() == 1
    assert pool.get_by_email("a@x") is None
    assert pool.get_by_email("b@x") is not None


def test_status_snapshot_shape(tmp_path):
    _write_credential(tmp_path, "a@x")
    _write_credential(tmp_path, "b@x")
    pool = CredentialPool(tmp_path)
    slot = pool.pick()
    pool.mark_cooldown(slot, status=429, error_code="resource-exhausted")

    snap = pool.status()
    assert snap["total"] == 2
    assert snap["available"] == 1
    assert snap["in_cooldown"] == 1
    slots = {s["email"]: s for s in snap["slots"]}
    cooled = slots[slot.email]
    assert cooled["available"] is False
    assert cooled["cooldown_reason"] == "rate_limit"
    assert cooled["backoff_level"] == 1
    assert cooled["cooldown_seconds_remaining"] > 0


def test_current_pointer_orders_slot_first(tmp_path):
    """xai-current.json → its target slot is served first, even if older."""
    path_a = _write_credential(tmp_path, "older@x")
    time.sleep(0.02)
    _write_credential(tmp_path, "newer@x")
    (tmp_path / "xai-current.json").write_text(
        json.dumps({"path": str(path_a)}), encoding="utf-8"
    )
    pool = CredentialPool(tmp_path)
    primary = pool.primary()
    assert primary is not None
    assert primary.email == "older@x"


@pytest.mark.parametrize(
    "filename",
    [
        "gemini-token.json",
        "codex-account.json",
        "not-xai.json",
    ],
)
def test_foreign_credentials_are_skipped(tmp_path, filename):
    """Non-xAI files must be ignored, not raise."""
    (tmp_path / filename).write_text(
        json.dumps({"type": "gemini", "access_token": "x"}), encoding="utf-8"
    )
    _write_credential(tmp_path, "real@x")
    pool = CredentialPool(tmp_path)
    assert pool.size() == 1
    assert pool.get_by_email("real@x") is not None
