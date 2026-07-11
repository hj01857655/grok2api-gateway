@echo off
setlocal
cd /d "%~dp0"

REM Prefer .env via python app; only set env fallbacks here if missing.
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8787"

echo Starting grok2api (reads GROK2API_API_KEY from .env)
if exist ".client_key" (
  echo Client key file: .client_key
)

".venv\Scripts\python.exe" -m app
