$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

if (-not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    Write-Host "Created .env from example — set GROK2API_API_KEY, then open /admin to add a Grok account"
}

$dist = ".\app\static\admin-dist\index.html"
if (-not (Test-Path $dist)) {
    Write-Host "Admin SPA not built (app/static/admin-dist missing). Falling back to legacy admin.html."
    Write-Host "  Optional: cd admin-ui; npm install; npm run build"
}

Write-Host "Starting grok2api on port from .env (default 8787)..."
Write-Host "  Admin: http://127.0.0.1:8787/admin"
& .\.venv\Scripts\python.exe -m app
