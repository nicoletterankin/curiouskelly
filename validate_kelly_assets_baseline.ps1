# Kelly Asset Baseline Validation Script
# Phase 1: Establish baseline quality scores

# Load main script functions
. .\generate_assets.ps1

$validationResults = @()
$validationOutputDir = "validation_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $validationOutputDir -Force | Out-Null

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "KELLY ASSET BASELINE VALIDATION" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Output Directory: $validationOutputDir" -ForegroundColor Yellow
Write-Host ""

# Load reference images for comparison
Write-Host "Loading reference images..." -ForegroundColor Cyan
$refImages = Get-ReferenceImages
Write-Host "Loaded $($refImages.Count) reference image(s)" -ForegroundColor Green
Write-Host ""

# Validation Function
function Validate-Asset {
    param (
        [string]$AssetPath,
        [string]$AssetName,
        [string]$ExpectedVariant = "Reinmaker",
        [array]$ReferenceImages = @()
    )
    
    $result = @{
        AssetName = $AssetName
        AssetPath = $AssetPath
        ExpectedVariant = $ExpectedVariant
        ValidationDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Scores = @{
            CharacterConsistency = @{
                FaceShape = 0
                SkinTone = 0
                EyeColor = 0
                HairColor = 0
                HairTexture = 0
                Expression = 0
                Overall = 0
            }
            BrandAlignment = @{
                Wardrobe = 0
                Colors = 0
                Style = 0
                Quality = 0
                Overall = 0
            }
            TechnicalQuality = @{
                Resolution = 0
                Artifacts = 0
                Blur = 0
                Colors = 0
                Watermarks = 0
                Overall = 0
            }
        }
        Issues = @()
        Recommendations = @()
        OverallScore = 0
        QualityLevel = ""
    }
    
    # Check if file exists
    if (-not (Test-Path $AssetPath)) {
        $result.Issues += "Asset file not found: $AssetPath"
        $result.OverallScore = 0
        $result.QualityLevel = "Unacceptable"
        return $result
    }
    
    # Get file info
    $fileInfo = Get-Item $AssetPath
    $result.FileSize = $fileInfo.Length
    $result.FileModified = $fileInfo.LastWriteTime
    
    # Manual validation checklist
    # Note: This requires manual review - scores should be entered manually
    # For now, we'll create a validation template
    
    Write-Host "="*60 -ForegroundColor Yellow
    Write-Host "Validating: $AssetName" -ForegroundColor Yellow
    Write-Host "Path: $AssetPath" -ForegroundColor Gray
    Write-Host "Expected Variant: $ExpectedVariant" -ForegroundColor Gray
    Write-Host "="*60 -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "CHARACTER CONSISTENCY CHECKLIST:" -ForegroundColor Cyan
    Write-Host "  Score each item 1-5 (5 = perfect match):" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [ ] Face Shape: Oval (not round, not square)" -ForegroundColor White
    Write-Host "  [ ] Skin Tone: Warm light-medium (not pale, not dark, not cool-toned)" -ForegroundColor White
    Write-Host "  [ ] Eye Color: Warm brown, almond-shaped (not blue, not round)" -ForegroundColor White
    Write-Host "  [ ] Hair Color: Medium brown with caramel/honey-blonde highlights" -ForegroundColor White
    Write-Host "  [ ] Hair Texture: Soft wavy to slightly curly (not straight, not tight curls)" -ForegroundColor White
    Write-Host "  [ ] Expression: Genuine warm smile with straight white teeth" -ForegroundColor White
    Write-Host ""
    
    Write-Host "BRAND ALIGNMENT CHECKLIST:" -ForegroundColor Cyan
    Write-Host "  Score each item 1-5 (5 = perfect match):" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [ ] Wardrobe: Matches $ExpectedVariant variant reference" -ForegroundColor White
    Write-Host "  [ ] Colors: Matches brand restrictions" -ForegroundColor White
    Write-Host "  [ ] Style: Photorealistic (not cartoon, not stylized)" -ForegroundColor White
    Write-Host "  [ ] Quality: Professional photography quality" -ForegroundColor White
    Write-Host ""
    
    Write-Host "TECHNICAL QUALITY CHECKLIST:" -ForegroundColor Cyan
    Write-Host "  Score each item 1-5 (5 = perfect):" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  [ ] Resolution: Matches requested dimensions" -ForegroundColor White
    Write-Host "  [ ] Artifacts: No compression artifacts" -ForegroundColor White
    Write-Host "  [ ] Blur: No blur or pixelation" -ForegroundColor White
    Write-Host "  [ ] Colors: Natural, not oversaturated" -ForegroundColor White
    Write-Host "  [ ] Watermarks: No watermarks or text overlays" -ForegroundColor White
    Write-Host ""
    
    Write-Host "Manual scoring required. Please review asset and enter scores." -ForegroundColor Yellow
    Write-Host ""
    
    return $result
}

# Assets to validate
$assetsToValidate = @(
    @{
        Name = "A1. Player: Kelly (Runner)"
        Path = "assets\player.png"
        Variant = "Reinmaker"
    },
    @{
        Name = "E1. Opening Splash"
        Path = "marketing\splash_intro.png"
        Variant = "Reinmaker"
    },
    @{
        Name = "F1. Itch.io Banner"
        Path = "marketing\itch-banner-1920x480.png"
        Variant = "Reinmaker"
    }
)

Write-Host "Validating $($assetsToValidate.Count) Kelly assets..." -ForegroundColor Cyan
Write-Host ""

foreach ($asset in $assetsToValidate) {
    $validation = Validate-Asset `
        -AssetPath $asset.Path `
        -AssetName $asset.Name `
        -ExpectedVariant $asset.Variant `
        -ReferenceImages $refImages
    
    $validationResults += $validation
    
    Write-Host ""
    Write-Host "Validation complete for: $($asset.Name)" -ForegroundColor Green
    Write-Host ""
}

# Generate validation report
Write-Host "="*80 -ForegroundColor Green
Write-Host "BASELINE VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "Assets Validated: $($validationResults.Count)" -ForegroundColor Cyan
Write-Host "Results saved to: $validationOutputDir" -ForegroundColor Yellow
Write-Host ""

# Export results
$validationResults | ConvertTo-Json -Depth 10 | Out-File "$validationOutputDir\baseline_validation.json"
Write-Host "Validation results exported to: $validationOutputDir\baseline_validation.json" -ForegroundColor Green
Write-Host ""

Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Review each asset manually" -ForegroundColor White
Write-Host "2. Score each checklist item 1-5" -ForegroundColor White
Write-Host "3. Document specific issues" -ForegroundColor White
Write-Host "4. Calculate overall scores" -ForegroundColor White
Write-Host "5. Generate baseline quality report" -ForegroundColor White
Write-Host ""












