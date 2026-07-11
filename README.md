# Grok2API — three-protocol gateway (admin-managed upstreams)

**Not** CLIProxyAPI. **Not** chenyme web reverse.

Upstream inventory is **only** what you add in `/admin`:

- **Mid-station channels** (iamhc, NewAPI, OpenRouter, …) → `~/.grok2api/providers.json`
- **Official Grok accounts** (Device Code / import `xai-*.json`) → `~/.grok2api/auths`

`.env` holds **process settings only** (host, port, door key, mode). It does **not** invent channels or accounts.

```
Client (Chat / Responses / Anthropic Messages + count)
         │  GROK2API_API_KEY (gateway door)
         ▼
      grok2api :8787
         │
         ├─ managed channels  (added at /admin)
         │    OpenAI-compatible mid-station / custom API
         │
         └─ official Grok     (added at /admin or oauth CLI)
              Device Code login  or  import xai-*.json
```

| What | Where it lives | How to add |
|------|----------------|------------|
| Mid-station channel | `~/.grok2api/providers.json` | `/admin` → 添加渠道 |
| Official Grok account | `~/.grok2api/auths/xai-*.json` | `/admin` Device Code / 导入 |
| Door key, host, port | `.env` | edit `.env` |

`UPSTREAM_MODE`:

| Mode | Behavior |
|------|----------|
| **auto** (default) | Official Grok if credential present, else managed channels |
| **compat** | Managed mid-station channels only |
| **oauth** | Official Grok (Device Code credential) |
| **credential** | Official Grok (imported `xai-*.json`) |

## Quick start

```powershell
cd E:\VSCodeSpace\grok2api
copy .env.example .env
# edit GROK2API_API_KEY
.\start.ps1
# open http://127.0.0.1:8787/admin  → add a channel or Grok account
```

- Health: `http://127.0.0.1:8787/health` — empty channels until you add one
- Admin: same key as `GROK2API_API_KEY`

```powershell
$H = @{ Authorization = "Bearer <GROK2API_API_KEY>"; "Content-Type" = "application/json" }
# after adding a channel with model DeepSeek-V4-Pro:
Invoke-RestMethod -Uri http://127.0.0.1:8787/v1/chat/completions -Method POST -Headers $H `
  -Body '{"model":"DeepSeek-V4-Pro","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
```

## Admin API (channels)

```http
GET    /admin/api/channels
POST   /admin/api/channels   {"name","base_url","api_key","models","prefix?"}
DELETE /admin/api/channels/{id}
```

## Official Grok

### Device Code

```powershell
python -m app.oauth.login
# or /admin → Device Code
```

### Import file

```powershell
python -m app.oauth.login --import path\to\xai-user@x.ai.json
# or /admin paste/upload
```

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

CI: `.github/workflows/ci.yml` (Python 3.11–3.13).

## Grok Build

1. Client key = `GROK2API_API_KEY`
2. Merge `examples/grok-build-config.toml` into `~/.grok/config.toml`
3. Point custom models at this gateway after channels are added in `/admin`

## Layout

```
app/
  channel_store.py   managed mid-station providers.json
  providers.py       routing / model rewrite
  upstream.py        auto | compat | oauth | credential
  admin_routes.py    /admin — channels + Grok credentials
  oauth/             Device Code + import
  converters/        Responses / Anthropic ↔ Chat
tests/
examples/
  grok-build-config.toml
```

`_refs/` is study-only.
