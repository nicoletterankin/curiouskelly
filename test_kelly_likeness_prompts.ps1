# Test Kelly Likeness - Prompt Variations

# Source the main asset generation script
. .\generate_assets.ps1

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "KELLY LIKENESS TEST - PROMPT VARIATIONS" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Testing different prompt strengths for:" -ForegroundColor Yellow
Write-Host "  1. Face shape (oval, soft contours)" -ForegroundColor White
Write-Host "  2. Hair length (long, extending past shoulders)" -ForegroundColor White
Write-Host ""

# Create test directory
$testDir = "likeness_tests_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null
Write-Host "Test directory: $testDir" -ForegroundColor Gray
Write-Host ""

# Test 1: Enhanced Face Shape & Hair Length
Write-Host "[Test 1] Enhanced descriptions (current updated prompts)..." -ForegroundColor Yellow
$test1Prompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic portrait of Kelly, facing camera, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "standing, confident pose, facing camera" `
    -Lighting "professional photography quality lighting, soft key light at 45 degrees" `
    -AdditionalNegatives "" `
    -ReferenceImages @()

Generate-VertexAI-Asset `
    -AssetName "Test 1: Enhanced Descriptions" `
    -Prompt $test1Prompt.Prompt `
    -NegativePrompt $test1Prompt.Negative `
    -ReferenceImages @() `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testDir/test1_enhanced_descriptions.png"

Write-Host ""

# Test 2: Explicit Exclusions
Write-Host "[Test 2] Explicit exclusions (NOT angular, NOT short hair)..." -ForegroundColor Yellow
$test2Prompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic portrait of Kelly, facing camera, NOT angular face, NOT short hair, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "standing, confident pose, facing camera" `
    -Lighting "professional photography quality lighting, soft key light at 45 degrees" `
    -AdditionalNegatives "angular face, short hair, shoulder-length hair, bob cut" `
    -ReferenceImages @()

Generate-VertexAI-Asset `
    -AssetName "Test 2: Explicit Exclusions" `
    -Prompt $test2Prompt.Prompt `
    -NegativePrompt $test2Prompt.Negative `
    -ReferenceImages @() `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testDir/test2_explicit_exclusions.png"

Write-Host ""

# Test 3: Maximum Emphasis
Write-Host "[Test 3] Maximum emphasis on face shape and hair length..." -ForegroundColor Yellow
$test3Scene = "Full-body photorealistic portrait of Kelly, facing camera. CRITICAL: Oval face shape with soft rounded contours - NO angular features, NO sharp jawline. CRITICAL: Long hair extending well past shoulders, reaching mid-back area - NO short hair, NO shoulder-length hair."
$test3Prompt = Build-KellyPrompt `
    -SceneDescription $test3Scene `
    -WardrobeVariant "Reinmaker" `
    -Pose "standing, confident pose, facing camera" `
    -Lighting "professional photography quality lighting, soft key light at 45 degrees" `
    -AdditionalNegatives "angular face, square face, sharp jawline, short hair, shoulder-length hair, bob cut, pixie cut" `
    -ReferenceImages @()

Generate-VertexAI-Asset `
    -AssetName "Test 3: Maximum Emphasis" `
    -Prompt $test3Prompt.Prompt `
    -NegativePrompt $test3Prompt.Negative `
    -ReferenceImages @() `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testDir/test3_maximum_emphasis.png"

Write-Host ""

Write-Host "="*80 -ForegroundColor Green
Write-Host "TEST GENERATION COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Generated 3 test images:" -ForegroundColor Cyan
Write-Host "  1. Enhanced descriptions (updated prompts)" -ForegroundColor White
Write-Host "  2. Explicit exclusions (NOT angular, NOT short)" -ForegroundColor White
Write-Host "  3. Maximum emphasis (CRITICAL keywords)" -ForegroundColor White
Write-Host ""
Write-Host "Compare results in: $testDir" -ForegroundColor Yellow
Write-Host "Compare against reference images to determine best approach." -ForegroundColor Gray
Write-Host ""











