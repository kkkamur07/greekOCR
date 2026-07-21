#Requires -Version 5.1
# Install Nomicous Inference Helper from the extracted Windows zip.
param(
  [string]$RegistryUrl = "https://api.nomicous.com/inference/v1/registry"
)

$ErrorActionPreference = "Stop"

$InstallRoot = Join-Path $env:LOCALAPPDATA "Nomicous\InferenceHelper"
$LogDir = Join-Path $env:USERPROFILE ".nomicous\logs"
$CacheDir = Join-Path $env:USERPROFILE ".nomicous\hf\cache"
$SourceDir = Join-Path $PSScriptRoot "nomicous-inference-helper"
$StageRoot = "$InstallRoot.staging"
$BackupRoot = "$InstallRoot.previous"

if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
  throw "Extract the ZIP first and run this script beside the 'nomicous-inference-helper' directory."
}

function Stop-HelperTaskAndWait {
  $task = Get-ScheduledTask -TaskName "NomicousInferenceHelper" -ErrorAction SilentlyContinue
  if (-not $task) { return }
  Stop-ScheduledTask -TaskName "NomicousInferenceHelper" -ErrorAction SilentlyContinue
  for ($attempt = 0; $attempt -lt 50; $attempt++) {
    $task = Get-ScheduledTask -TaskName "NomicousInferenceHelper" -ErrorAction SilentlyContinue
    if (-not $task -or $task.State -ne "Running") { return }
    Start-Sleep -Milliseconds 200
  }
  throw "Timed out waiting for the existing helper task to stop."
}

function Wait-InstallUnlocked([string]$Root) {
  $executable = Join-Path $Root "nomicous-inference-helper.exe"
  if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { return }
  for ($attempt = 0; $attempt -lt 50; $attempt++) {
    try {
      $stream = [System.IO.File]::Open(
        $executable,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::None
      )
      $stream.Dispose()
      return
    } catch {
      Start-Sleep -Milliseconds 200
    }
  }
  throw "Timed out waiting for the existing helper executable to be released."
}

Remove-Item -LiteralPath $StageRoot, $BackupRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $StageRoot, $LogDir, $CacheDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $SourceDir "*") $StageRoot

$InstalledNewBundle = $false
$PreviousTaskExisted = $false
$EnvironmentChanged = $false
$EnvironmentNames = @(
  "HELPER_REGISTRY_URL",
  "HF_CACHE_ROOT",
  "INFERENCE_MAX_REQUEST_BODY_BYTES",
  "INFERENCE_MAX_ENCODED_IMAGE_BYTES",
  "INFERENCE_MAX_DECODED_IMAGE_BYTES",
  "INFERENCE_MAX_IMAGE_PIXELS",
  "INFERENCE_MAX_TRANSCRIBE_LINES"
)
$PreviousUserEnvironment = @{}
$PreviousProcessEnvironment = @{}
foreach ($name in $EnvironmentNames) {
  $PreviousUserEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "User")
  $PreviousProcessEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
  $ExistingTask = Get-ScheduledTask -TaskName "NomicousInferenceHelper" -ErrorAction SilentlyContinue
  if ($ExistingTask) {
    $PreviousTaskExisted = $true
    Stop-HelperTaskAndWait
    Wait-InstallUnlocked $InstallRoot
    Unregister-ScheduledTask -TaskName "NomicousInferenceHelper" -Confirm:$false
  }

  if (Test-Path -LiteralPath $InstallRoot) {
    Move-Item -LiteralPath $InstallRoot -Destination $BackupRoot
  }
  Move-Item -LiteralPath $StageRoot -Destination $InstallRoot
  $InstalledNewBundle = $true

  $EnvironmentChanged = $true
  [Environment]::SetEnvironmentVariable("HELPER_REGISTRY_URL", $RegistryUrl, "User")
  $env:HELPER_REGISTRY_URL = $RegistryUrl
  $InferenceLimits = @{
    "HF_CACHE_ROOT" = $CacheDir
    "INFERENCE_MAX_REQUEST_BODY_BYTES" = "167772160"
    "INFERENCE_MAX_ENCODED_IMAGE_BYTES" = "167772160"
    "INFERENCE_MAX_DECODED_IMAGE_BYTES" = "104857600"
    "INFERENCE_MAX_IMAGE_PIXELS" = "200000000"
    "INFERENCE_MAX_TRANSCRIBE_LINES" = "10000"
  }
  foreach ($entry in $InferenceLimits.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "User")
    Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
  }

  $Exe = Join-Path $InstallRoot "nomicous-inference-helper.exe"
  $Action = New-ScheduledTaskAction -Execute $Exe -WorkingDirectory $InstallRoot
  $Trigger = New-ScheduledTaskTrigger -AtLogOn
  $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName "NomicousInferenceHelper" -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null
  Start-ScheduledTask -TaskName "NomicousInferenceHelper"

  $ready = $false
  for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
      $response = Invoke-WebRequest `
        -Uri "http://127.0.0.1:8001/health" `
        -UseBasicParsing `
        -TimeoutSec 2
      if ($response.StatusCode -eq 200) {
        $ready = $true
        break
      }
    } catch {
      # The scheduled task may need a moment to start.
    }
    Start-Sleep -Seconds 1
  }

  if (-not $ready) {
    throw "The helper did not become ready on http://127.0.0.1:8001. See $LogDir\inference-helper.log."
  }

  Remove-Item -LiteralPath $BackupRoot -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "Installed Nomicous Inference Helper."
} catch {
  if ($InstalledNewBundle) {
    Stop-HelperTaskAndWait
    Wait-InstallUnlocked $InstallRoot
    Unregister-ScheduledTask -TaskName "NomicousInferenceHelper" -Confirm:$false -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $InstallRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path -LiteralPath $BackupRoot) {
    Move-Item -LiteralPath $BackupRoot -Destination $InstallRoot
  }
  if ($EnvironmentChanged) {
    foreach ($name in $EnvironmentNames) {
      [Environment]::SetEnvironmentVariable($name, $PreviousUserEnvironment[$name], "User")
      [Environment]::SetEnvironmentVariable($name, $PreviousProcessEnvironment[$name], "Process")
    }
  }
  if ($PreviousTaskExisted -and (Test-Path -LiteralPath $InstallRoot)) {
    $PreviousExe = Join-Path $InstallRoot "nomicous-inference-helper.exe"
    $PreviousAction = New-ScheduledTaskAction -Execute $PreviousExe -WorkingDirectory $InstallRoot
    $PreviousTrigger = New-ScheduledTaskTrigger -AtLogOn
    $PreviousSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName "NomicousInferenceHelper" -Action $PreviousAction -Trigger $PreviousTrigger -Settings $PreviousSettings -Force | Out-Null
    Start-ScheduledTask -TaskName "NomicousInferenceHelper"
  }
  throw
} finally {
  Remove-Item -LiteralPath $StageRoot -Recurse -Force -ErrorAction SilentlyContinue
}
