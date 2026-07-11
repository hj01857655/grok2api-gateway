# Live smoke against running gateway (or starts one).
# Requires valid mid-station key in .env (XAI_API_KEY / OPENAI_COMPATIBILITY).
param(
  [string]$Base = "http://127.0.0.1:8787",
  [string]$Model = "DeepSeek-V4-Flash",
  [switch]$StartServer
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$key = ((Get-Content .env -ErrorAction Stop) | Where-Object { $_ -match '^GROK2API_API_KEY=' }) -replace '^GROK2API_API_KEY=',''
if (-not $key) { throw "GROK2API_API_KEY missing in .env" }

$proc = $null
if ($StartServer) {
  $proc = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","app" -PassThru -WindowStyle Hidden
  $ok = $false
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 300
    try { Invoke-RestMethod "$Base/health" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
  }
  if (-not $ok) { throw "server failed to start" }
}

try {
  $H = @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" }
  $health = Invoke-RestMethod "$Base/health"
  Write-Host "health mode=$($health.upstream_mode) key_configured=$($health.upstream_key_configured)"

  $chat = Invoke-RestMethod -Uri "$Base/v1/chat/completions" -Method POST -Headers $H -TimeoutSec 120 -Body (@{
    model = $Model
    messages = @(@{ role = "user"; content = "Reply with exactly: pong" })
    max_tokens = 16
    temperature = 0
  } | ConvertTo-Json -Depth 5)
  Write-Host "chat: $($chat.choices[0].message.content)"

  $resp = Invoke-RestMethod -Uri "$Base/v1/responses" -Method POST -Headers $H -TimeoutSec 120 -Body (@{
    model = $Model
    input = "Reply with exactly: pong"
    max_output_tokens = 16
  } | ConvertTo-Json)
  Write-Host "responses: $($resp.output_text)"

  $msg = Invoke-RestMethod -Uri "$Base/v1/messages" -Method POST -Headers $H -TimeoutSec 120 -Body (@{
    model = $Model
    max_tokens = 16
    messages = @(@{ role = "user"; content = "Reply with exactly: pong" })
  } | ConvertTo-Json -Depth 5)
  Write-Host "messages: $($msg.content[0].text)"
  Write-Host "E2E_OK"
}
finally {
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
