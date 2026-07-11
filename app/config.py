from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .providers import (
    CompatProvider,
    UpstreamRoute,
    build_provider_list,
    providers_summary,
    resolve_route,
)

# compat     = OpenAI-compatible custom / mid-station (iamhc, OpenRouter, …)
# oauth      = official Grok via Device Code OAuth login
# credential = official Grok via imported credential files (xai-*.json)
UpstreamMode = Literal["compat", "oauth", "credential"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # --- Default / legacy single OpenAI-compat upstream (mid-station / BYOK) ---
    xai_api_key: str = Field(default="", validation_alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", validation_alias="XAI_BASE_URL")

    # Multi custom upstreams (CPA openai-compatibility style).
    openai_compatibility: str = Field(default="", validation_alias="OPENAI_COMPATIBILITY")

    # --- Upstream mode: compat | oauth | credential ---
    upstream_mode: str = Field(default="compat", validation_alias="UPSTREAM_MODE")
    oauth_auths_dir: str = Field(default="", validation_alias="OAUTH_AUTHS_DIR")
    oauth_token_file: str = Field(default="", validation_alias="OAUTH_TOKEN_FILE")

    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8787, validation_alias="PORT")
    grok2api_api_key: str = Field(default="", validation_alias="GROK2API_API_KEY")
    model_aliases: str = Field(default="", validation_alias="MODEL_ALIASES")
    default_models: str = Field(
        default="grok-3,grok-3-mini",
        validation_alias="DEFAULT_MODELS",
    )
    upstream_timeout: float = Field(default=600.0, validation_alias="UPSTREAM_TIMEOUT")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")

    @field_validator("xai_api_key", mode="before")
    @classmethod
    def fill_xai_from_voya(cls, v):
        if v and str(v).strip() and str(v).strip() not in ("xai-your-key-here", ""):
            return str(v).strip()
        for env_name in ("VOYA_API_KEY", "OPENAI_API_KEY", "NEWAPI_API_KEY"):
            alt = (os.environ.get(env_name) or "").strip()
            if alt:
                return alt
        return (str(v).strip() if v else "") or ""

    @field_validator("upstream_mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        m = (str(v or "compat")).strip().lower()
        if m in ("compat", "oauth", "credential"):
            return m
        raise ValueError(
            f"UPSTREAM_MODE must be 'compat', 'oauth', or 'credential', got {v!r}"
        )

    def is_compat_mode(self) -> bool:
        """OpenAI-compatible custom / mid-station upstream."""
        return self.upstream_mode == "compat"

    def is_oauth_mode(self) -> bool:
        """Official Grok via Device Code OAuth."""
        return self.upstream_mode == "oauth"

    def is_credential_mode(self) -> bool:
        """Official Grok via imported credential files."""
        return self.upstream_mode == "credential"

    def is_official_mode(self) -> bool:
        """Either oauth or credential — both use official Grok account tokens."""
        return self.upstream_mode in ("oauth", "credential")

    def auths_dir(self) -> Path:
        if self.oauth_auths_dir.strip():
            return Path(self.oauth_auths_dir.strip()).expanduser()
        return Path.home() / ".grok2api" / "auths"

    def oauth_token_path(self) -> Optional[Path]:
        if self.oauth_token_file.strip():
            return Path(self.oauth_token_file.strip()).expanduser()
        return None

    def alias_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        raw = (self.model_aliases or "").strip()
        if not raw:
            return out
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            client, upstream = part.split(":", 1)
            client, upstream = client.strip(), upstream.strip()
            if client and upstream:
                out[client] = upstream
        return out

    def default_model_list(self) -> List[str]:
        return [m.strip() for m in self.default_models.split(",") if m.strip()]

    def compat_providers(self) -> List[CompatProvider]:
        """All OpenAI-compatible custom / mid-station upstreams (not OAuth)."""
        return build_provider_list(
            compat_raw=self.openai_compatibility,
            default_base_url=self.xai_base_url,
            default_api_key=self.xai_api_key,
            default_models=self.default_model_list(),
            default_name="default",
        )

    def resolve_model(self, model: str | None) -> str:
        """Resolve client model → upstream model id."""
        if self.is_official_mode():
            if not model:
                models = self.default_model_list()
                return models[0] if models else "grok-3"
            return self.alias_map().get(model, model)
        return self.resolve_upstream(model).model

    def resolve_upstream(self, model: str | None) -> UpstreamRoute:
        """Full route: provider base/key + upstream model id (compat mode)."""
        return resolve_route(
            self.compat_providers(),
            model=model,
            global_aliases=self.alias_map(),
            default_models=self.default_model_list(),
        )

    def providers_public(self) -> List[dict]:
        return providers_summary(self.compat_providers())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
