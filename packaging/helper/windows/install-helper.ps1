#Requires -Version 5.1
# Install Nomicous Inference Helper from the extracted Windows zip.
param(
  [string]$RegistryUrl = "https://api.nomicous.com/inference/v1/registry",
  [string]$CorsOrigins = "https://app.nomicous.com"
)

$ErrorActionPreference = "Stop"

$InstallRoot = Join-Path $env:LOCALAPPDATA "Nomicous\InferenceHelper"
$LogDir = Join-Path $env:USERPROFILE ".nomicous\logs"
$CacheDir = Join-Path $env:USERPROFILE ".nomicous\hf\cache"
$SourceDir = Join-Path $PSScriptRoot "nomicous-inference-helper"

New-Item -ItemType Directory -Force -Path $InstallRoot, $LogDir, $CacheDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $SourceDir "*") $InstallRoot
[Environment]::SetEnvironmentVariable("HELPER_REGISTRY_URL", $RegistryUrl, "User")
$env:HELPER_REGISTRY_URL = $RegistryUrl
[Environment]::SetEnvironmentVariable("HELPER_CORS_ORIGINS", $CorsOrigins, "User")
$env:HELPER_CORS_ORIGINS = $CorsOrigins

$Exe = Join-Path $InstallRoot "nomicous-inference-helper.exe"
$Action = New-ScheduledTaskAction -Execute $Exe -WorkingDirectory $InstallRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "NomicousInferenceHelper" -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName "NomicousInferenceHelper"

Write-Host "Installed Nomicous Inference Helper. Probe http://127.0.0.1:8001/health"
