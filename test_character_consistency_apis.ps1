# Test Script for Character Consistency API Comparison
# This script tests both Vertex AI and Google AI Studio APIs with and without reference images

# Load the main script functions
. .\generate_assets.ps1

Write-Host "="*80 -ForegroundColor Green
Write-Host "Character Consistency API Comparison Test" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

# Load reference images
Write-Host "Loading reference images..." -ForegroundColor Cyan
$refImages = Get-ReferenceImages
Write-Host ""

# Create test output directory
$testOutputDir = "test_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $testOutputDir -Force | Out-Null
Write-Host "Test output directory: $testOutputDir" -ForegroundColor Yellow
Write-Host ""

# Test asset: A1 Player sprite
$testPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in neutral running pose, facing right, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, dynamic movement, readable silhouette, action-ready stance" `
    -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" `
    -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering" `
    -ReferenceImages $refImages

$testPromptText = $testPrompt.Prompt
$testNegativePrompt = $testPrompt.Negative

Write-Host "Test Asset: A1 Player Sprite (1024x1280)" -ForegroundColor Yellow
Write-Host ""

# Test 1: Vertex AI with reference images
Write-Host "[TEST 1/4] Vertex AI WITH reference images..." -ForegroundColor Cyan
$outputFile1 = Join-Path $testOutputDir "test_vertex_ai_with_ref.png"
try {
    Generate-VertexAI-Asset `
        -AssetName "Test: Vertex AI (with ref)" `
        -Prompt $testPromptText `
        -NegativePrompt $testNegativePrompt `
        -ReferenceImages $refImages `
        -MasterWidth 1024 `
        -MasterHeight 1280 `
        -OutputFile $outputFile1
    Write-Host "[SUCCESS] Test 1 complete: $outputFile1" -ForegroundColor Green
} catch {
    Write-Host "[FAILED] Test 1 failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 2: Vertex AI without reference images
Write-Host "[TEST 2/4] Vertex AI WITHOUT reference images..." -ForegroundColor Cyan
$outputFile2 = Join-Path $testOutputDir "test_vertex_ai_without_ref.png"
try {
    Generate-VertexAI-Asset `
        -AssetName "Test: Vertex AI (without ref)" `
        -Prompt $testPromptText `
        -NegativePrompt $testNegativePrompt `
        -ReferenceImages @() `
        -MasterWidth 1024 `
        -MasterHeight 1280 `
        -OutputFile $outputFile2
    Write-Host "[SUCCESS] Test 2 complete: $outputFile2" -ForegroundColor Green
} catch {
    Write-Host "[FAILED] Test 2 failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 3: Google AI Studio with reference images
Write-Host "[TEST 3/4] Google AI Studio WITH reference images..." -ForegroundColor Cyan
$outputFile3 = Join-Path $testOutputDir "test_google_ai_studio_with_ref.png"
try {
    Generate-Google-Asset `
        -AssetName "Test: Google AI Studio (with ref)" `
        -Prompt $testPromptText `
        -NegativePrompt $testNegativePrompt `
        -ReferenceImages $refImages `
        -MasterWidth 1024 `
        -MasterHeight 1280 `
        -OutputFile $outputFile3
    Write-Host "[SUCCESS] Test 3 complete: $outputFile3" -ForegroundColor Green
} catch {
    Write-Host "[FAILED] Test 3 failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Note: Google AI Studio may not support reference images" -ForegroundColor Yellow
}
Write-Host ""

# Test 4: Google AI Studio without reference images
Write-Host "[TEST 4/4] Google AI Studio WITHOUT reference images..." -ForegroundColor Cyan
$outputFile4 = Join-Path $testOutputDir "test_google_ai_studio_without_ref.png"
try {
    Generate-Google-Asset `
        -AssetName "Test: Google AI Studio (without ref)" `
        -Prompt $testPromptText `
        -NegativePrompt $testNegativePrompt `
        -ReferenceImages @() `
        -MasterWidth 1024 `
        -MasterHeight 1280 `
        -OutputFile $outputFile4
    Write-Host "[SUCCESS] Test 4 complete: $outputFile4" -ForegroundColor Green
} catch {
    Write-Host "[FAILED] Test 4 failed: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Summary
Write-Host "="*80 -ForegroundColor Green
Write-Host "Test Summary" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "All test outputs saved to: $testOutputDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Please review the generated images and compare:" -ForegroundColor Cyan
Write-Host "  1. Character consistency between with/without reference images" -ForegroundColor Gray
Write-Host "  2. Quality differences between Vertex AI and Google AI Studio" -ForegroundColor Gray
Write-Host "  3. Which API produces better results" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review images in $testOutputDir" -ForegroundColor Gray
Write-Host "  2. Document findings in compare_results.md" -ForegroundColor Gray
Write-Host "  3. Select best API approach for production regeneration" -ForegroundColor Gray
Write-Host ""












