param(
    [string]$ProjectId,
    [string]$ServiceAccountJson = "",
    [string]$VertexLocation = "us-central1",
    [string]$RealEsrganExe = ""
)

$ErrorActionPreference = "Stop"
if (-not $ProjectId) { throw "-ProjectId is required" }

$env:GOOGLE_CLOUD_PROJECT = $ProjectId
$env:VERTEX_LOCATION = $VertexLocation
if ($ServiceAccountJson) {
    if (-not (Test-Path $ServiceAccountJson)) { throw "Service account file not found: $ServiceAccountJson" }
    $env:GOOGLE_APPLICATION_CREDENTIALS = (Resolve-Path $ServiceAccountJson).Path
}
if ($RealEsrganExe) {
    if (-not (Test-Path $RealEsrganExe)) { throw "REAL-ESRGAN exe not found: $RealEsrganExe" }
    $env:REAL_ESRGAN_EXE = (Resolve-Path $RealEsrganExe).Path
}

Write-Host "Environment configured:" -ForegroundColor Green
Write-Host "  GOOGLE_CLOUD_PROJECT = $env:GOOGLE_CLOUD_PROJECT"
Write-Host "  VERTEX_LOCATION      = $env:VERTEX_LOCATION"
if ($env:GOOGLE_APPLICATION_CREDENTIALS) { Write-Host "  GOOGLE_APPLICATION_CREDENTIALS = $env:GOOGLE_APPLICATION_CREDENTIALS" }
if ($env:REAL_ESRGAN_EXE) { Write-Host "  REAL_ESRGAN_EXE = $env:REAL_ESRGAN_EXE" }


