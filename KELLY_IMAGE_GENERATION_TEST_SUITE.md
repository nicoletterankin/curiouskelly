# Kelly Image Generation - Test Suite

**Purpose:** Systematic validation of Kelly image generation system

**Standard:** Steve Jobs "Insanely Great" - Every test must pass

---

## Test Execution Script

This script runs comprehensive tests on Kelly image generation.

```powershell
# Kelly Image Generation Test Suite
# Run this to validate system quality

# Test Configuration
$testResults = @()
$testOutputDir = "test_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $testOutputDir -Force | Out-Null

# Load main script
. .\generate_assets.ps1

# Test Scoring Function
function Score-Asset {
    param (
        [string]$AssetPath,
        [string]$TestName,
        [array]$ReferenceImages = @()
    )
    
    $score = @{
        TestName = $TestName
        AssetPath = $AssetPath
        CharacterConsistency = 0
        BrandAlignment = 0
        TechnicalQuality = 0
        OverallScore = 0
        Issues = @()
        Pass = $false
    }
    
    # TODO: Implement actual scoring logic
    # For now, return placeholder
    return $score
}

# Test Set 1: Character Consistency
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Test Set 1: Character Consistency" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Test 1.1: Front View Test
Write-Host "`nTest 1.1: Front View Test" -ForegroundColor Yellow
$refImages = Get-ReferenceImages
$prompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly facing camera, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "standing, facing camera, neutral expression" `
    -ReferenceImages $refImages

# Generate test asset
$testResult = Generate-VertexAI-Asset `
    -AssetName "Test 1.1: Front View" `
    -Prompt $prompt.Prompt `
    -NegativePrompt $prompt.Negative `
    -ReferenceImages $refImages `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "$testOutputDir/test_1_1_front_view.png"

# Score and document
$score = Score-Asset -AssetPath $testResult -TestName "Front View Test" -ReferenceImages $refImages
$testResults += $score

# Repeat for all test cases...
# (Full implementation would include all tests from the plan)

# Generate Report
Write-Host "`n"*2 -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host "Test Results Summary" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green

$overallScore = ($testResults | Measure-Object -Property OverallScore -Average).Average
Write-Host "Overall Average Score: $([math]::Round($overallScore, 2))/5" -ForegroundColor Cyan
Write-Host "Total Tests: $($testResults.Count)" -ForegroundColor Cyan
Write-Host "Passed: $(($testResults | Where-Object { $_.Pass }).Count)" -ForegroundColor Green
Write-Host "Failed: $(($testResults | Where-Object { -not $_.Pass }).Count)" -ForegroundColor Red

# Export results
$testResults | ConvertTo-Json -Depth 10 | Out-File "$testOutputDir/test_results.json"
Write-Host "`nTest results saved to: $testOutputDir" -ForegroundColor Yellow
```

---

## Manual Test Checklist

Use this checklist for manual validation:

### Character Consistency
- [ ] Face shape matches reference (oval)
- [ ] Skin tone matches reference (warm light-medium)
- [ ] Eye color matches reference (warm brown)
- [ ] Hair color matches reference (medium brown with caramel highlights)
- [ ] Hair texture matches reference (wavy to slightly curly)
- [ ] Facial features match reference (nose, lips, eyebrows)
- [ ] Expression matches reference (genuine warm smile)
- [ ] Overall appearance matches reference

### Brand Alignment
- [ ] Wardrobe matches variant reference
- [ ] Colors match brand restrictions
- [ ] Style is photorealistic (not cartoon/stylized)
- [ ] Quality is professional photography level
- [ ] No brand violations (bright colors, wrong style, etc.)

### Technical Quality
- [ ] Resolution matches requested dimensions
- [ ] Aspect ratio is correct
- [ ] No compression artifacts
- [ ] No blur or pixelation
- [ ] Natural colors (not oversaturated)
- [ ] No watermarks or text overlays

**Score each category 1-5, then calculate overall average.**

---

## Quality Gate Criteria

**Minimum Acceptable:** ≥4/5 average score  
**Good:** ≥4.5/5 average score  
**Great:** ≥4.8/5 average score  
**Insanely Great:** ≥4.9/5 average score + 100% brand compliance

**Action:** Regenerate if score < 4/5











