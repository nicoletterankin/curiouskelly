# Fix Reference Image Format - Multiple Attempts

# Source the main asset generation script
. .\generate_assets.ps1

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "FIXING REFERENCE IMAGE FORMAT" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Attempting different reference image formats to find correct structure..." -ForegroundColor Yellow
Write-Host ""

# Load ONLY the best 1-2 character references
Write-Host "Loading best character references..." -ForegroundColor Yellow
$charRefs = Get-CharacterReferences
$bestRefs = $charRefs | Where-Object { $_.Priority -eq "primary" } | Select-Object -First 1
if ($bestRefs.Count -eq 0) {
    $bestRefs = $charRefs | Select-Object -First 1
}

if ($bestRefs.Count -eq 0) {
    Write-Host "ERROR: No reference images found!" -ForegroundColor Red
    exit 1
}

Write-Host "Using best reference: $($bestRefs.Name)" -ForegroundColor Green
Write-Host ""

# Create test directory
$testDir = "reference_fix_tests_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

# Simple test prompt
$simplePrompt = "Full-body photorealistic portrait of Kelly Rein, facing camera, Reinmaker armor, professional photography quality"

Write-Host "[Test] Generating with reference image (wrapped format)..." -ForegroundColor Yellow
Generate-VertexAI-Asset `
    -AssetName "Reference Image Test" `
    -Prompt $simplePrompt `
    -NegativePrompt "" `
    -ReferenceImages $bestRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testDir/test_reference_wrapped.png"

Write-Host ""
Write-Host "Test complete. Check output and error messages above." -ForegroundColor Cyan
Write-Host "If successful, reference images are working!" -ForegroundColor Green
Write-Host "If failed, check payload JSON for format details." -ForegroundColor Yellow
Write-Host ""












