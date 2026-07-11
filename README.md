# Grok2API — three-protocol gateway (custom API + optional official Grok)

**Not** CLIProxyAPI. **Not** chenyme web reverse.

Like CPA: **independent upstream kinds** — you do **not** need an official Grok account
to use custom OpenAI-compatible APIs.

```
Client (Chat / Responses / Anthropic Messages + count)
         │  GROK2API_API_KEY (gateway door)
         ▼
      grok2api :8787
         │
         ├─ UPSTREAM_MODE=compat (default)
         │    any OpenAI-compatible custom / mid-station API
         │    · XAI_BASE_URL + XAI_API_KEY / VOYA_API_KEY
         │    · and/or OPENAI_COMPATIBILITY (multi-provider JSON)
         │
         ├─ UPSTREAM_MODE=oauth
         │    official Grok via Device Code login
         │
         └─ UPSTREAM_MODE=credential
              official Grok via imported xai-*.json files
```

| Mode | What you need | Official Grok account? |
|------|---------------|------------------------|
| **compat** (default) | Custom base URL + API key (iamhc, NewAPI, OpenRouter, …) | **No** |
| **oauth** | Device Code login (`python -m app.oauth.login` or `/admin`) | **Yes** |
| **credential** | Import `xai-*.json` (paste/upload on `/admin` or CLI `--import`) | **Yes** |

`oauth` and `credential` both use official account tokens under `~/.grok2api/auths`.
They differ only in how you obtain the credential (login vs import).

Admin `/admin` manages official Grok credentials (login + import). Custom APIs are in `.env` / JSON.

## Quick start (custom API — no Grok account)

```powershell
cd E:\VSCodeSpace\grok2api
# .env: UPSTREAM_MODE=compat, XAI_BASE_URL=https://api.iamhc.cn/v1
# key via XAI_API_KEY or VOYA_API_KEY
.\start.ps1
```

- Health: `http://127.0.0.1:8787/health` → shows `upstream_mode=compat` and `custom_providers`
- Client key: `GROK2API_API_KEY` (Bearer / `x-api-key`)

```powershell
$H = @{ Authorization = "Bearer <GROK2API_API_KEY>"; "Content-Type" = "application/json" }
Invoke-RestMethod -Uri http://127.0.0.1:8787/v1/chat/completions -Method POST -Headers $H `
  -Body '{"model":"DeepSeek-V4-Pro","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
```

## Multi custom providers (CPA `openai-compatibility` style)

`.env`:

```env
UPSTREAM_MODE=compat
OPENAI_COMPATIBILITY=examples/openai-compatibility.json
```

`examples/openai-compatibility.json`:

```json
[
  {
    "name": "iamhc",
    "prefix": "iamhc",
    "base_url": "https://api.iamhc.cn/v1",
    "api_key": "${VOYA_API_KEY}",
    "models": [
      { "name": "DeepSeek-V4-Pro", "alias": "DeepSeek-V4-Pro" },
      { "name": "grok-4.5", "alias": "grok-4.5" }
    ]
  }
]
```

- Model match → that provider’s base_url + key  
- Or pin: `iamhc/DeepSeek-V4-Pro`  
- Leave `OPENAI_COMPATIBILITY` empty → single legacy `XAI_BASE_URL` + key only  

## Official Grok

### oauth — Device Code

```powershell
python -m app.oauth.login
# then .env: UPSTREAM_MODE=oauth
```

### credential — import file

```powershell
python -m app.oauth.login --import path\to\xai-user@x.ai.json
# then .env: UPSTREAM_MODE=credential
```

Or open `http://127.0.0.1:8787/admin` (same key as `GROK2API_API_KEY`) — Device Code + paste/upload import.

## Protocols

| Endpoint | Role |
|----------|------|
| `POST /v1/chat/completions` | OpenAI Chat |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/responses/input_tokens` | Count (local estimate) |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/messages/count_tokens` | Count (local estimate) |
| `GET  /v1/models` | Model list |

## Tests

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Covers protocol converters, multi-provider routing, token estimates, and HTTP API
with a mocked upstream (no real mid-station required).

CI runs the same suite on push/PR via `.github/workflows/ci.yml` (Python 3.11–3.13).

## Grok Build

1. Client key = `GROK2API_API_KEY`
2. Merge `examples/grok-build-config.toml` into `~/.grok/config.toml`
3. Keep existing `[model.voya]` — independent of official Grok modes

## Layout

```
app/
  providers.py       multi custom OpenAI-compat routing
  upstream.py        compat | oauth | credential
  token_count.py     count endpoints
  admin_routes.py    /admin — official Grok credentials (login + import)
  oauth/             Device Code + import
  converters/        Responses / Anthropic ↔ Chat
tests/               unit + API smoke tests
examples/
  openai-compatibility.json
  grok-build-config.toml
```

`_refs/` is study-only.
