# One-time Windows setup for the Brain Node appliance:
#   1) keep the machine awake on AC power
#   2) add an inbound firewall rule for the API port (so Node 2 / other machines can connect)
# Auto-elevates via a UAC prompt if not already running as Administrator.
#
#   .\scripts\setup_windows.ps1              # will prompt for admin if needed
#   .\scripts\setup_windows.ps1 -Port 8080
param([int]$Port = 8080)
$ErrorActionPreference = "Stop"

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Self-elevate: relaunch this script as Administrator in a new window, then exit.
if (-not (Test-Admin)) {
  Write-Host "Not elevated - requesting Administrator (accept the UAC prompt)..."
  try {
    Start-Process pwsh -Verb RunAs -ArgumentList @(
      "-NoExit", "-NoProfile", "-File", "`"$PSCommandPath`"", "-Port", "$Port"
    )
    Write-Host "An elevated window was opened - the setup runs there."
  } catch {
    Write-Warning "Auto-elevation failed: $($_.Exception.Message)"
    Write-Warning "Open PowerShell as Administrator and run:  $PSCommandPath -Port $Port"
  }
  return
}

Write-Host "[admin] 1) Keeping the machine awake on AC power..."
try {
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  powercfg /change monitor-timeout-ac 0
  Write-Host "   ok"
} catch {
  Write-Warning "   power settings failed: $($_.Exception.Message)"
}

Write-Host "[admin] 2) Inbound firewall rule for TCP $Port..."
$ruleName = "Brutus Brain Node ($Port)"
try {
  if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
    Write-Host "   already exists: $ruleName"
  } else {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
      -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
    Write-Host "   ADDED: $ruleName"
  }
} catch {
  Write-Warning "   firewall rule FAILED: $($_.Exception.Message)"
}

Write-Host "[admin] 3) Static IP: prefer a DHCP reservation on the travel router (safer than a manual static IP)."
Write-Host ""
Write-Host "Done. Verify the rule:  Get-NetFirewallRule -DisplayName '$ruleName'"
Write-Host "This machine's LAN IP:   (Get-NetIPConfiguration | Where-Object IPv4DefaultGateway).IPv4Address.IPAddress"
