# Start the Brutus Brain Node (control-path FastAPI server).
# Single worker => models load once and stay resident. No --reload in production.
#
#   .\scripts\run.ps1                       # mock mode, 0.0.0.0:8080
#   .\scripts\run.ps1 -Mode real -Port 8080 # real (NPU) mode
#
# NOTE: $Host is a reserved PowerShell variable, so the bind address param is -BindHost.
param(
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8080,
  [string]$Mode = "mock"
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

Write-Host "Starting Brain Node (mode=$Mode) on ${BindHost}:${Port} using $py"
& $py -m uvicorn app.main:app --host $BindHost --port $Port --workers 1
