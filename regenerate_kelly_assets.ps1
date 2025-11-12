# Regenerate Kelly Assets with Perfect Character Consistency
# This script regenerates all Kelly assets using Vertex AI with reference images

# Load the main script functions
. .\generate_assets.ps1

Write-Host "="*80 -ForegroundColor Green
Write-Host "Perfect Character Consistency Regeneration" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

# Step 1: Load CHARACTER reference images (face, hair, skin tone - NOT wardrobe)
Write-Host "Step 1: Loading CHARACTER reference images..." -ForegroundColor Cyan
Write-Host "Using character references for face, hair, and skin tone consistency..." -ForegroundColor Gray
$charRefs = Get-CharacterReferences
Write-Host "Loaded $($charRefs.Count) character reference(s) for character consistency" -ForegroundColor Green
Write-Host ""

# Step 2: Backup existing assets
Write-Host "Step 2: Backing up existing assets..." -ForegroundColor Cyan
$backupDir = "assets\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "Backup directory: $backupDir" -ForegroundColor Gray

# Backup each asset
$playerSrc = "assets\player.png"
$splashSrc = "marketing\splash_intro.png"
$bannerSrc = "marketing\itch-banner-1920x480.png"

if (Test-Path $playerSrc) {
    Copy-Item $playerSrc (Join-Path $backupDir "player_old.png") -Force
    Write-Host "  Backed up: $playerSrc" -ForegroundColor Green
}
if (Test-Path $splashSrc) {
    Copy-Item $splashSrc (Join-Path $backupDir "splash_intro_old.png") -Force
    Write-Host "  Backed up: $splashSrc" -ForegroundColor Green
}
if (Test-Path $bannerSrc) {
    Copy-Item $bannerSrc (Join-Path $backupDir "itch-banner-1920x480_old.png") -Force
    Write-Host "  Backed up: $bannerSrc" -ForegroundColor Green
}
Write-Host ""

# Step 3: Regenerate Kelly assets with enhanced prompts and CHARACTER reference images
Write-Host "Step 3: Regenerating Kelly assets with perfect character consistency..." -ForegroundColor Cyan
Write-Host "IMPORTANT: Using CHARACTER references (face, hair, skin tone) - NOT wardrobe references" -ForegroundColor Yellow
if ($charRefs.Count -gt 0) {
    Write-Host "Using: Vertex AI with $($charRefs.Count) CHARACTER reference image(s) for face/hair/skin consistency" -ForegroundColor Yellow
    Write-Host "Primary references: $($charRefs | Where-Object { $_.Priority -eq 'primary' } | Select-Object -First 3 | ForEach-Object { $_.Name } | Join-String -Separator ', ')" -ForegroundColor Gray
} else {
    Write-Host "Using: Vertex AI with enhanced prompts only" -ForegroundColor Yellow
}
Write-Host ""

# A1. Player: Kelly (Runner)
Write-Host "[1/3] Regenerating A1. Player: Kelly (Runner)" -ForegroundColor Cyan
$playerPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in neutral running pose, facing right, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, dynamic movement, readable silhouette, action-ready stance" `
    -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" `
    -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "A1. Player: Kelly (Runner)" `
    -Prompt $playerPrompt.Prompt `
    -NegativePrompt $playerPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "assets/player.png"

Write-Host ""

# E1. Opening Splash
Write-Host "[2/3] Regenerating E1. Opening Splash" -ForegroundColor Cyan
$splashPrompt = Build-KellyPrompt `
    -SceneDescription "Cinematic splash art: Kelly stepping out from a grand dark forge into the light, on the dark side old leather reins hang like chains, on the light side the air is clean, she carries a small glowing circuit-rein token that illuminates her determined face, high contrast between shadow and light, dusk palette, the glowing token is the focal point" `
    -WardrobeVariant "Reinmaker" `
    -Pose "stepping forward, determined expression, holding glowing circuit-rein token" `
    -Lighting "cinematic lighting, high contrast between dark forge interior and bright exterior, token provides warm key light on face" `
    -AdditionalNegatives "painterly, vector style, stylized, illustration, non-photorealistic" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "E1. Opening Splash" `
    -Prompt $splashPrompt.Prompt `
    -NegativePrompt $splashPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1280 `
    -MasterHeight 720 `
    -OutputFile "marketing/splash_intro.png"

Write-Host ""

# F1. Itch.io Banner
Write-Host "[3/3] Regenerating F1. Itch.io Banner" -ForegroundColor Cyan
$bannerDesc = "Wide marketing banner montage: on the right, photorealistic Kelly in full running pose; on the left, the game's bold logo (The Rein Maker's Daughter); in the background is the parallax skyline of the Hall of the Seven Tribes at dusk"
$bannerPrompt = Build-KellyPrompt `
    -SceneDescription $bannerDesc `
    -WardrobeVariant "Reinmaker" `
    -Pose "full running pose, dynamic action, facing right" `
    -Lighting "dynamic composition lighting, dusk palette, cinematic banner aesthetic" `
    -AdditionalNegatives "stylized sprite, illustration, non-photorealistic character, cartoon, game asset style" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "F1. Itch.io Banner" `
    -Prompt $bannerPrompt.Prompt `
    -NegativePrompt $bannerPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1920 `
    -MasterHeight 480 `
    -OutputFile "marketing/itch-banner-1920x480.png"

Write-Host ""

# Summary
Write-Host "="*80 -ForegroundColor Green
Write-Host "Regeneration Complete!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Regenerated Assets:" -ForegroundColor Cyan
Write-Host "  assets/player.png" -ForegroundColor Green
Write-Host "  marketing/splash_intro.png" -ForegroundColor Green
Write-Host "  marketing/itch-banner-1920x480.png" -ForegroundColor Green
Write-Host ""
Write-Host "Backup Location: $backupDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "All assets regenerated with:" -ForegroundColor Cyan
Write-Host "  - Enhanced character consistency prompts" -ForegroundColor Gray
Write-Host "  - CHARACTER reference images: $($charRefs.Count) (face, hair, skin tone)" -ForegroundColor Gray
Write-Host "  - Primary references: $($charRefs | Where-Object { $_.Priority -eq 'primary' } | Measure-Object).Count" -ForegroundColor Gray
Write-Host "  - Vertex AI API with OAuth2 authentication" -ForegroundColor Gray
Write-Host "  - Photorealistic style enforcement" -ForegroundColor Gray
Write-Host ""
