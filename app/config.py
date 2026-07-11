from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .channel_store import load_providers, providers_path, summary_for_settings
from .providers import CompatProvider, UpstreamRoute, resolve_route

# auto       = use official Grok if credential present, else managed mid-stations
# compat     = managed mid-station channels only (admin-added)
# oauth      = official Grok via Device Code OAuth login
# credential = official Grok via imported credential files (xai-*.json)
UpstreamMode = Literal["auto", "compat", "oauth", "credential"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # Gateway process only — NOT upstream inventory.
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8787, validation_alias="PORT")
    grok2api_api_key: str = Field(default="", validation_alias="GROK2API_API_KEY")
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")
    upstream_timeout: float = Field(default=600.0, validation_alias="UPSTREAM_TIMEOUT")

    # Upstream selection. Inventory comes from admin stores, not env keys.
    upstream_mode: str = Field(default="auto", validation_alias="UPSTREAM_MODE")
    oauth_auths_dir: str = Field(default="", validation_alias="OAUTH_AUTHS_DIR")
    oauth_token_file: str = Field(default="", validation_alias="OAUTH_TOKEN_FILE")
    data_dir: str = Field(default="", validation_alias="GROK2API_DATA_DIR")

    model_aliases: str = Field(default="", validation_alias="MODEL_ALIASES")
    default_models: str = Field(
        default="grok-3,grok-3-mini",
        validation_alias="DEFAULT_MODELS",
    )

    @field_validator("upstream_mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        m = (str(v or "auto")).strip().lower()
        if m in ("auto", "compat", "oauth", "credential"):
            return m
        # legacy env values still accepted
        if m in ("mid", "midstation", "custom"):
            return "compat"
        raise ValueError(
            "UPSTREAM_MODE must be 'auto', 'compat', 'oauth', or 'credential', "
            f"got {v!r}"
        )

    def home_dir(self) -> Path:
        if self.data_dir.strip():
            return Path(self.data_dir.strip()).expanduser()
        return Path.home() / ".grok2api"

    def is_compat_mode(self) -> bool:
        return self.effective_upstream_mode() == "compat"

    def is_oauth_mode(self) -> bool:
        return self.effective_upstream_mode() == "oauth"

    def is_credential_mode(self) -> bool:
        return self.effective_upstream_mode() == "credential"

    def is_official_mode(self) -> bool:
        return self.effective_upstream_mode() in ("oauth", "credential")

    def has_official_credential(self) -> bool:
        from .oauth.xai import load_token

        ts = load_token(path=self.oauth_token_path(), auths_dir=self.auths_dir())
        return bool(ts and ts.access_token)

    def effective_upstream_mode(self) -> str:
        """Resolve auto → concrete mode from what admin has added."""
        mode = self.upstream_mode
        if mode != "auto":
            return mode
        if self.has_official_credential():
            # Prefer explicit oauth if credential file looks like device login;
            # credential vs oauth both use same token store.
            return "oauth"
        if self.compat_providers():
            return "compat"
        # Nothing configured yet — stay compat so errors talk about channels.
        return "compat"

    def auths_dir(self) -> Path:
        if self.oauth_auths_dir.strip():
            return Path(self.oauth_auths_dir.strip()).expanduser()
        return self.home_dir() / "auths"

    def providers_store_path(self) -> Path:
        return providers_path(self.home_dir())

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
        """Mid-station channels added via admin only."""
        return load_providers(self.home_dir())

    def resolve_model(self, model: str | None) -> str:
        if self.is_official_mode():
            if not model:
                models = self.default_model_list()
                return models[0] if models else "grok-3"
            return self.alias_map().get(model, model)
        return self.resolve_upstream(model).model

    def resolve_upstream(self, model: str | None) -> UpstreamRoute:
        providers = self.compat_providers()
        if not providers:
            raise RuntimeError(
                "No mid-station channels configured. "
                "Open /admin → add a channel (base URL + API key + models). "
                f"Store: {self.providers_store_path()}"
            )
        return resolve_route(
            providers,
            model=model,
            global_aliases=self.alias_map(),
            default_models=self.default_model_list(),
        )

    def providers_public(self) -> List[dict]:
        return summary_for_settings(self.home_dir())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
