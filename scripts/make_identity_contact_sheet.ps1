param(
    [string]$RendersDir = "projects/Kelly/assets/renders",
    [string]$OutPath = "projects/Kelly/assets/identity_contact_sheet.png"
)

$ErrorActionPreference = "Stop"
$python = "python"
& $python --version | Out-Null

& $python tools/identity_contact_sheet.py --renders $RendersDir --out $OutPath
if ($LASTEXITCODE -ne 0) { throw "Contact sheet failed" }
Write-Host "Contact sheet written to $OutPath" -ForegroundColor Green


