param(
    [Parameter(Mandatory=$true)][string]$Preset,
    [string]$OutDir = "projects/Kelly/assets",
    [switch]$OpenFolder
)

$ErrorActionPreference = "Stop"

Write-Host "=== Kelly Asset Generator ===" -ForegroundColor Cyan
Write-Host "Preset: $Preset" -ForegroundColor Yellow

if (-not (Test-Path $Preset)) {
    Write-Error "Preset not found: $Preset"
}

# Ensure Python is available
$python = "python"
try {
    & $python --version | Out-Null
} catch {
    Write-Error "Python not found in PATH. Install Python 3.9+ and retry."
}

# Run generator
& $python tools/kelly_asset_generator.py $Preset --outdir $OutDir

if ($LASTEXITCODE -ne 0) {
    Write-Error "Generation failed."
}

if ($OpenFolder) {
    Start-Process $OutDir
}

Write-Host "Done." -ForegroundColor Green


