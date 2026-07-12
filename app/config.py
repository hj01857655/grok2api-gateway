from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# auto       = official Grok if credential present, else error at request time
# oauth      = official Grok via Device Code OAuth login
# credential = official Grok via imported credential files (xai-*.json)
UpstreamMode = Literal["auto", "oauth", "credential"]


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

    # Official Grok only. Credentials from admin / Device Code / import.
    upstream_mode: str = Field(default="auto", validation_alias="UPSTREAM_MODE")
    oauth_auths_dir: str = Field(default="", validation_alias="OAUTH_AUTHS_DIR")
    oauth_token_file: str = Field(default="", validation_alias="OAUTH_TOKEN_FILE")
    data_dir: str = Field(default="", validation_alias="GROK2API_DATA_DIR")

    model_aliases: str = Field(default="", validation_alias="MODEL_ALIASES")
    default_models: str = Field(
        default="grok-3,grok-3-mini",
        validation_alias="DEFAULT_MODELS",
    )

    # Request log (admin console) — metadata JSONL under ~/.grok2api/logs
    request_log_enabled: bool = Field(default=True, validation_alias="REQUEST_LOG_ENABLED")
    request_log_keep_days: int = Field(default=7, validation_alias="REQUEST_LOG_KEEP_DAYS")
    request_log_max_mb: float = Field(default=50.0, validation_alias="REQUEST_LOG_MAX_MB")
    request_log_body_max: int = Field(default=0, validation_alias="REQUEST_LOG_BODY_MAX")

    # apply_patch: keep as function for xAI (unlike CPA which strips by name).
    # Local disk apply is OFF by default — clients (Codex) usually apply themselves.
    apply_patch_normalize: bool = Field(default=True, validation_alias="APPLY_PATCH_NORMALIZE")
    apply_patch_strip: bool = Field(default=False, validation_alias="APPLY_PATCH_STRIP")
    apply_patch_local: bool = Field(default=False, validation_alias="APPLY_PATCH_LOCAL")
    apply_patch_workspace: str = Field(default="", validation_alias="APPLY_PATCH_WORKSPACE")

    def apply_patch_root(self) -> Path:
        raw = (self.apply_patch_workspace or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return Path.cwd().resolve()

    @field_validator("upstream_mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        m = (str(v or "auto")).strip().lower()
        if m in ("auto", "oauth", "credential"):
            return m
        # legacy mid-station values → auto (official)
        if m in ("compat", "mid", "midstation", "custom"):
            return "auto"
        raise ValueError(
            "UPSTREAM_MODE must be 'auto', 'oauth', or 'credential', "
            f"got {v!r}"
        )

    def home_dir(self) -> Path:
        if self.data_dir.strip():
            return Path(self.data_dir.strip()).expanduser()
        return Path.home() / ".grok2api"

    def is_oauth_mode(self) -> bool:
        return self.effective_upstream_mode() == "oauth"

    def is_credential_mode(self) -> bool:
        return self.effective_upstream_mode() == "credential"

    def is_official_mode(self) -> bool:
        return True

    def has_official_credential(self) -> bool:
        from .oauth.xai import load_token

        ts = load_token(path=self.oauth_token_path(), auths_dir=self.auths_dir())
        return bool(ts and ts.access_token)

    def effective_upstream_mode(self) -> str:
        """Resolve auto → oauth when credential present; still oauth if missing."""
        mode = self.upstream_mode
        if mode != "auto":
            return mode
        return "oauth"

    def auths_dir(self) -> Path:
        if self.oauth_auths_dir.strip():
            return Path(self.oauth_auths_dir.strip()).expanduser()
        return self.home_dir() / "auths"

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

    def resolve_model(self, model: str | None) -> str:
        if not model:
            models = self.default_model_list()
            return models[0] if models else "grok-3"
        return self.alias_map().get(model, model)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
