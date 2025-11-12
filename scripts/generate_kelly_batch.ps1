param(
    [string]$PresetsFolder = "presets",
    [string]$OutDir = "projects/Kelly/assets"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Kelly Batch Generator ===" -ForegroundColor Cyan
Write-Host "Folder: $PresetsFolder" -ForegroundColor Yellow

$python = "python"
& $python --version | Out-Null

$presets = Get-ChildItem -Path $PresetsFolder -Filter *.yaml | Sort-Object Name
$pass = 0
$fail = 0
$rows = @()
foreach ($p in $presets) {
    Write-Host ("\n-- {0}" -f $p.Name) -ForegroundColor Green
$env:PYTHONWARNINGS = "ignore"
$out = & $python -W ignore tools/kelly_asset_generator.py $p.FullName --outdir $OutDir | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed: $($p.Name)"
        $fail++
        continue
    }

    try {
        $json = $out | ConvertFrom-Json
        $manifestPath = $json.manifest
        $m = Get-Content $manifestPath -Raw | ConvertFrom-Json
        $verdict = $m.qc.verdict
        if ($verdict.pass -eq $true) {
            Write-Host ("   PASS  {0}" -f $manifestPath) -ForegroundColor Cyan
            $pass++
            $rows += [pscustomobject]@{ preset=$p.Name; status='PASS'; reasons='' }
        } else {
            $reason = ($verdict.reasons -join ',')
            Write-Host ("   FAIL  {0}  [{1}]" -f $manifestPath, $reason) -ForegroundColor Red
            $fail++
            $rows += [pscustomobject]@{ preset=$p.Name; status='FAIL'; reasons=$reason }
        }
    } catch {
        Write-Warning "   Could not parse manifest for $($p.Name)"
        $fail++
    }
}

Write-Host "\n=== Kelly Batch Summary ===" -ForegroundColor Cyan
Write-Host ("PASS: {0}   FAIL: {1}" -f $pass, $fail) -ForegroundColor Yellow
if ($rows.Count -gt 0) {
    $rows | Format-Table -AutoSize | Out-String | Write-Host
}


