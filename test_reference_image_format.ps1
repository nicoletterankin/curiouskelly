# Test Reference Image Format - Vertex AI API

# Source the main asset generation script
. .\generate_assets.ps1

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "TESTING REFERENCE IMAGE FORMATS" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Load character references
Write-Host "Loading character references..." -ForegroundColor Yellow
$charRefs = Get-CharacterReferences
Write-Host "Loaded $($charRefs.Count) character reference(s)" -ForegroundColor Green

if ($charRefs.Count -eq 0) {
    Write-Host "ERROR: No reference images found!" -ForegroundColor Red
    Write-Host "Please ensure reference images are in: $referenceImagePath" -ForegroundColor Yellow
    exit 1
}

# Use only the best 1-2 references
$bestRefs = $charRefs | Where-Object { $_.Priority -eq "primary" } | Select-Object -First 2
if ($bestRefs.Count -eq 0) {
    $bestRefs = $charRefs | Select-Object -First 2
}

Write-Host ""
Write-Host "Testing with $($bestRefs.Count) reference image(s):" -ForegroundColor Yellow
foreach ($ref in $bestRefs) {
    Write-Host "  • $($ref.Name)" -ForegroundColor Gray
}
Write-Host ""

# Create test directory
$testDir = "reference_format_tests_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null
Write-Host "Test directory: $testDir" -ForegroundColor Gray
Write-Host ""

# Test 1: Format in parameters (new format)
Write-Host "[Test 1] Reference images in parameters section..." -ForegroundColor Yellow
$test1Prompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic portrait of Kelly, facing camera" `
    -WardrobeVariant "Reinmaker" `
    -Pose "standing, confident pose" `
    -ReferenceImages $bestRefs

Generate-VertexAI-Asset `
    -AssetName "Test 1: Reference in Parameters" `
    -Prompt $test1Prompt.Prompt `
    -NegativePrompt $test1Prompt.Negative `
    -ReferenceImages $bestRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testDir/test1_params_format.png"

Write-Host ""

# If Test 1 fails, try alternative formats
Write-Host "If Test 1 fails, check the payload JSON files for format details." -ForegroundColor Yellow
Write-Host "Testing complete. Check results in: $testDir" -ForegroundColor Cyan
Write-Host ""











