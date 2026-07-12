# Grok2API — three-protocol gateway (official Grok only)

**Not** CLIProxyAPI. **Not** chenyme web reverse.

Upstream is **official Grok only** (Device Code OAuth or imported `xai-*.json`).
No mid-station / OpenAI-compat relay channels.

`.env` holds **process settings only** (host, port, door key, mode). Credentials live under `~/.grok2api/auths`.

```
Client (Chat / Responses / Anthropic Messages + count)
         │  GROK2API_API_KEY (gateway door)
         ▼
      grok2api :8787
         │
         └─ official Grok token
              wire: POST …/responses only
              convert when client ≠ Responses
```

| What | Where it lives | How to add |
|------|----------------|------------|
| Official Grok account | `~/.grok2api/auths/xai-*.json` | `/admin` Device Code / 导入 |
| Door key, host, port | `.env` | edit `.env` |

`UPSTREAM_MODE`:

| Mode | Behavior |
|------|----------|
| **auto** (default) | Prefer official credential when present; still official-only |
| **oauth** | Official Grok (Device Code credential) |
| **credential** | Official Grok (imported `xai-*.json`) |

## Quick start

```powershell
cd E:\VSCodeSpace\grok2api
copy .env.example .env
# edit GROK2API_API_KEY
.\start.ps1
# open http://127.0.0.1:8787/admin  → Device Code or import xai-*.json
```

- Health: `http://127.0.0.1:8787/health`
- Admin: multi-page SPA at `/admin` (same key as `GROK2API_API_KEY`)

### Admin UI (React)

Production build is served by FastAPI from `app/static/admin-dist`. Rebuild after UI changes:

```powershell
cd admin-ui
npm install
npm run build
# → app/static/admin-dist  (base path /admin/)
```

Dev (hot reload; proxy `/admin/api` → `:8787`):

```powershell
# terminal 1: gateway
.\start.ps1
# terminal 2
cd admin-ui
npm run dev
# open http://127.0.0.1:5173/admin/
```

If `admin-dist` is missing, `/admin` falls back to legacy `app/static/admin.html`.

```powershell
$H = @{ Authorization = "Bearer <GROK2API_API_KEY>"; "Content-Type" = "application/json" }
Invoke-RestMethod -Uri http://127.0.0.1:8787/v1/chat/completions -Method POST -Headers $H `
  -Body '{"model":"grok-3","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
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
| `POST /v1/chat/completions` | OpenAI Chat → convert Chat↔Responses |
| `POST /v1/responses` | OpenAI Responses → native `/responses` |
| `POST /v1/responses/input_tokens` | Count (local estimate) |
| `POST /v1/messages` | Anthropic → convert Anthropic↔Chat↔Responses |
| `POST /v1/messages/count_tokens` | Count (local estimate) |
| `GET  /v1/models` | Model list |

Official xAI wire speaks **only** `POST …/responses`. Chat and Anthropic clients convert once.

## Tests

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

CI: `.github/workflows/ci.yml` (Python 3.11–3.13).

## Grok Build

1. Client key = `GROK2API_API_KEY`
2. Merge `examples/grok-build-config.toml` into `~/.grok/config.toml`
3. Point custom models at this gateway after a Grok account is added in `/admin`

## Layout

```
admin-ui/            React + Vite admin SPA (source)
app/
  upstream.py        official /responses bridge
  admin_routes.py    /admin SPA + /admin/api/* (accounts, logs, models, OAuth)
  request_log.py     JSONL request metadata under ~/.grok2api/logs
  apply_patch.py     optional local apply_patch executor
  static/admin-dist/ production admin build (npm run build in admin-ui/)
  oauth/             Device Code + import
  converters/        Chat / Anthropic ↔ Responses
tests/
```

## Request logs

Enabled by default (`REQUEST_LOG_ENABLED`). Metadata only (path/status/duration/model); no full bodies unless `REQUEST_LOG_BODY_MAX` > 0.

| API | Notes |
|-----|--------|
| `GET /admin/api/logs` | `limit`, `offset`, `path_prefix`, `status_min`/`max`, `since` |
| `GET /admin/api/logs/summary` | last 1h / 24h counts, 4xx/5xx |
| `GET /admin/api/models` | admin-wrapped upstream model list |
