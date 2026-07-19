# Start the Brain Node WITH per-request access logging (writes logs\requests.log).
# This runs the transparent wrapper app.request_logging:app -- endpoints and server
# behaviour are IDENTICAL to run.ps1 (which stays as the no-logging launcher).
# GenieX must already be running (or use start-brain-node.ps1 to bring it up first).
#
#   .\scripts\run-logged.ps1              # real mode (default), 0.0.0.0:8080
#   .\scripts\run-logged.ps1 -Mode mock   # mock mode
param(
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8080,
  [string]$Mode = "real"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

$env:PYTHONPATH = $root
$env:BRUTUS_MODE = $Mode
$env:BRUTUS_HOST = $BindHost
$env:BRUTUS_PORT = "$Port"

Write-Host "Starting Brain Node (mode=$Mode) with request logging -> logs\requests.log"
& $py -m uvicorn app.request_logging:app --host $BindHost --port $Port --workers 1
