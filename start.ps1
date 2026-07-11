$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

if (-not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    Write-Host "Created .env from example — fill XAI_API_KEY / XAI_BASE_URL"
}

Write-Host "Starting grok2api on port from .env (default 8787)..."
& .\.venv\Scripts\python.exe -m app
