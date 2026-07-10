#Requires -Version 5.1
param(
  # Authenticode signing (all optional - omit for an unsigned build).
  [string]$SigningThumbprint = $env:WINDOWS_SIGNING_THUMBPRINT,   # cert in the local store
  [string]$SigningCertPath = $env:WINDOWS_SIGNING_CERT,           # or path to a .pfx
  [string]$SigningCertPassword = $env:WINDOWS_SIGNING_CERT_PASSWORD,
  [string]$TimestampUrl = "http://timestamp.digicert.com",
  [switch]$SignAllBinaries
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$Root = Resolve-Path (Join-Path $ScriptDir "../../..")
$HelperDir = Join-Path $Root "packaging/helper"
$DistDir = Join-Path $HelperDir "dist"
$BundleDir = Join-Path $DistDir "nomicous-inference-helper"
$BuildScript = Join-Path $HelperDir "scripts/build-pyinstaller.sh"

function Resolve-SignTool {
  $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $found = Get-ChildItem -Path "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "x64" } | Sort-Object FullName -Descending | Select-Object -First 1
  if ($found) { return $found.FullName }
  throw "signtool.exe not found. Install the Windows SDK or add signtool to PATH."
}

function Invoke-AuthenticodeSign([string[]]$Paths) {
  if (-not $SigningThumbprint -and -not $SigningCertPath) {
    Write-Warning "No signing cert provided (WINDOWS_SIGNING_THUMBPRINT / WINDOWS_SIGNING_CERT) - building UNSIGNED. SmartScreen will warn on install."
    return
  }
  $signtool = Resolve-SignTool
  $common = @("sign", "/fd", "SHA256", "/tr", $TimestampUrl, "/td", "SHA256")
  if ($SigningThumbprint) {
    $common += @("/sha1", $SigningThumbprint)
  } else {
    $common += @("/f", $SigningCertPath)
    if ($SigningCertPassword) { $common += @("/p", $SigningCertPassword) }
  }
  foreach ($p in $Paths) {
    Write-Host "Signing $p"
    & $signtool @common $p
    if ($LASTEXITCODE -ne 0) { throw "signtool failed for $p (exit $LASTEXITCODE)" }
  }
}

Push-Location $HelperDir
try {
  & bash $BuildScript

  $InstallDir = Join-Path $DistDir "windows-installer"
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Copy-Item -Recurse -Force $BundleDir (Join-Path $InstallDir "nomicous-inference-helper")
  Copy-Item -Force (Join-Path $ScriptDir "install-helper.ps1") (Join-Path $InstallDir "install-helper.ps1")

  # Authenticode-sign before zipping so the shipped binaries are trusted.
  $HelperExe = Join-Path $InstallDir "nomicous-inference-helper/nomicous-inference-helper.exe"
  if ($SignAllBinaries) {
    $targets = Get-ChildItem -Path (Join-Path $InstallDir "nomicous-inference-helper") -Recurse -Include *.exe, *.dll, *.pyd |
      Select-Object -ExpandProperty FullName
    Invoke-AuthenticodeSign -Paths $targets
  } else {
    Invoke-AuthenticodeSign -Paths @($HelperExe)
  }

  if (Get-Command Compress-Archive -ErrorAction SilentlyContinue) {
    $ZipPath = Join-Path $DistDir "nomicous-inference-helper-windows.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath }
    Compress-Archive -Path (Join-Path $InstallDir "*") -DestinationPath $ZipPath
    Write-Host "Built $ZipPath"
  }
}
finally {
  Pop-Location
}
