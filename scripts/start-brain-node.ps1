# One-command boot for Node 1: starts the GenieX NPU LLM server (if not already
# running) and then the Brain Node, wired to the locked-in Qwen3-4B NPU model.
#
#   .\scripts\start-brain-node.ps1
#   .\scripts\start-brain-node.ps1 -SkipServe        # GenieX already running elsewhere
#
# GenieX opens in its own window (so you can watch NPU logs); the Brain Node runs
# in this window. Ctrl+C stops the Brain Node; close the GenieX window to stop it.
param(
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8080,
  [switch]$SkipServe
)
$ErrorActionPreference = "Stop"

function Resolve-Geniex {
  $known = Join-Path $env:LOCALAPPDATA "GenieX CLI\geniex.exe"
  if (Test-Path $known) { return $known }
  $cmd = Get-Command geniex -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  throw "geniex.exe not found. Install the GenieX CLI: https://geniex.aihub.qualcomm.com/en/run/cli/install"
}

function Test-GenieX {
  try {
    Invoke-WebRequest -Uri "http://127.0.0.1:18181/v1/models" -TimeoutSec 3 -UseBasicParsing | Out-Null
    return $true
  } catch {
    return $false
  }
}

if (-not $SkipServe) {
  if (Test-GenieX) {
    Write-Host "GenieX server already running on :18181."
  } else {
    $geniex = Resolve-Geniex
    Write-Host "Starting GenieX server in a new window: $geniex serve"
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", "& `"$geniex`" serve"
    Write-Host "Waiting for GenieX to be ready on :18181 ..."
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
      Start-Sleep -Seconds 2
      if (Test-GenieX) { $ready = $true; break }
    }
    if (-not $ready) { throw "GenieX did not become ready on :18181 in time (check the GenieX window)." }
    Write-Host "GenieX is up."
  }
}

# Locked-in Node 1 LLM config (explicit, so leftover shell vars can't change it).
$env:BRUTUS_LLM_BACKEND = "geniex"
$env:BRUTUS_GENIEX_MODEL = "qualcomm/Qwen3-4B"
$env:BRUTUS_LLM_SYSTEM_SUFFIX = "/no_think"

Write-Host "Starting the Brain Node (real mode) on ${BindHost}:${Port} ..."
& (Join-Path $PSScriptRoot "run.ps1") -Mode real -BindHost $BindHost -Port $Port
