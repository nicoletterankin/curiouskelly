param(
    [string]$RendersDir = "projects/Kelly/assets/renders",
    [string]$OutPath = "projects/Kelly/assets/cc5_identity_set.json",
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$python = "python"
& $python --version | Out-Null

& $python tools/cc5_identity_helper.py --renders $RendersDir --out $OutPath
if ($LASTEXITCODE -ne 0) { throw "Identity set JSON failed" }
Write-Host "CC5 identity set written to $OutPath" -ForegroundColor Green
if ($Open) { Start-Process $OutPath }


