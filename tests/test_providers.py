"""Multi-provider routing tests."""

from __future__ import annotations

import json
from pathlib import Path

from app.providers import (
    CompatModel,
    CompatProvider,
    build_provider_list,
    load_compat_providers,
    resolve_route,
)


def test_load_compat_providers_from_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-from-env")
    path = tmp_path / "compat.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "p1",
                    "prefix": "p1",
                    "base_url": "https://p1.example/v1",
                    "api_key": "${TEST_KEY}",
                    "models": [
                        {"name": "up-a", "alias": "a"},
                        "plain-b",
                    ],
                },
                {
                    "name": "disabled",
                    "base_url": "https://x",
                    "api_key": "k",
                    "disabled": True,
                    "models": ["m"],
                },
            ]
        ),
        encoding="utf-8",
    )
    providers = load_compat_providers(str(path))
    assert len(providers) == 1
    assert providers[0].api_key == "secret-from-env"
    assert providers[0].models[0].client_id() == "a"
    assert providers[0].models[1].upstream_id() == "plain-b"


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


def test_build_provider_list_legacy_default(monkeypatch):
    monkeypatch.delenv("VOYA_API_KEY", raising=False)
    providers = build_provider_list(
        compat_raw="",
        default_base_url="https://legacy/v1",
        default_api_key="legacy-key",
        default_models=["m1", "m2"],
        default_name="default",
    )
    assert len(providers) == 1
    assert providers[0].name == "default"
    assert providers[0].api_key == "legacy-key"
    assert [m.alias for m in providers[0].models] == ["m1", "m2"]
