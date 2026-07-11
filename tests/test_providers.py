"""Multi-provider routing + managed channel store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import channel_store
from app.providers import CompatModel, CompatProvider, resolve_route


def test_channel_store_empty_by_default(tmp_path: Path):
    assert channel_store.load_providers(tmp_path) == []
    assert channel_store.list_public(tmp_path) == []


def test_channel_store_add_list_delete(tmp_path: Path):
    created = channel_store.add_provider(
        name="iamhc",
        base_url="https://api.example.com/v1",
        api_key="sk-secret-key-1234",
        models="m1, m2",
        prefix="iamhc",
        root=tmp_path,
    )
    assert created["name"] == "iamhc"
    assert created["key_configured"] is True

    pubs = channel_store.list_public(tmp_path)
    assert len(pubs) == 1
    assert pubs[0]["base_url"] == "https://api.example.com/v1"
    assert pubs[0]["key_configured"] is True
    # secret not fully exposed
    assert "sk-secret-key-1234" not in str(pubs[0].get("key_hint", ""))

    loaded = channel_store.load_providers(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].api_key == "sk-secret-key-1234"
    assert [m.client_id() for m in loaded[0].models] == ["m1", "m2"]

    pid = created["id"]
    assert channel_store.delete_provider(pid, root=tmp_path) is True
    assert channel_store.load_providers(tmp_path) == []


def test_channel_store_requires_fields(tmp_path: Path):
    with pytest.raises(ValueError, match="name"):
        channel_store.add_provider(
            name="", base_url="https://x/v1", api_key="k", root=tmp_path
        )
    with pytest.raises(ValueError, match="base_url"):
        channel_store.add_provider(
            name="n", base_url="", api_key="k", root=tmp_path
        )
    with pytest.raises(ValueError, match="api_key"):
        channel_store.add_provider(
            name="n", base_url="https://x/v1", api_key="", root=tmp_path
        )


def test_resolve_route_prefix_and_alias():
    providers = [
        CompatProvider(
            name="iamhc",
            prefix="iamhc",
            base_url="https://a/v1",
            api_key="k1",
            models=[
                CompatModel(name="DeepSeek-V4-Pro", alias="DeepSeek-V4-Pro"),
                CompatModel(name="grok-real", alias="grok-4.5"),
            ],
        ),
        CompatProvider(
            name="or",
            prefix="or",
            base_url="https://b/v1",
            api_key="k2",
            models=[CompatModel(name="gpt", alias="gpt")],
        ),
    ]
    r1 = resolve_route(
        providers,
        model="DeepSeek-V4-Pro",
        global_aliases={},
        default_models=["x"],
    )
    assert r1.provider == "iamhc"
    assert r1.model == "DeepSeek-V4-Pro"

    r2 = resolve_route(
        providers,
        model="iamhc/grok-4.5",
        global_aliases={},
        default_models=["x"],
    )
    assert r2.provider == "iamhc"
    assert r2.model == "grok-real"
    assert r2.client_model == "iamhc/grok-4.5"

    r3 = resolve_route(
        providers,
        model="gpt",
        global_aliases={},
        default_models=["x"],
    )
    assert r3.provider == "or"
    assert r3.model == "gpt"

    r4 = resolve_route(
        providers,
        model="alias-client",
        global_aliases={"alias-client": "gpt"},
        default_models=["x"],
    )
    assert r4.provider == "or"
    assert r4.model == "gpt"


def test_resolve_route_empty_raises():
    with pytest.raises(RuntimeError, match="No mid-station"):
        resolve_route([], model="x", global_aliases={}, default_models=["x"])
