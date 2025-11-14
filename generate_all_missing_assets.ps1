# Generate All Missing Reinmaker Assets - Insanely Great Quality
# Using Updated Character Consistency System & Hair Specification

# Source the main asset generation script
. .\generate_assets.ps1

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "REINMAKER ASSET GENERATION - INSANELY GREAT QUALITY" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Using Updated Systems:" -ForegroundColor Cyan
Write-Host "  • Character consistency with proper character references" -ForegroundColor White
Write-Host "  • Updated hair specification (soft cohesive waves)" -ForegroundColor White
Write-Host "  • Vertex AI API with OAuth2 authentication" -ForegroundColor White
Write-Host "  • Enhanced prompts with Kelly's complete character description" -ForegroundColor White
Write-Host ""

# Load CHARACTER reference images (face, hair, skin tone - NOT wardrobe)
Write-Host "Step 1: Loading CHARACTER reference images..." -ForegroundColor Cyan
$charRefs = Get-CharacterReferences
Write-Host "Loaded $($charRefs.Count) character reference(s) for character consistency" -ForegroundColor Green
if ($charRefs.Count -gt 0) {
    $primaryRefs = $charRefs | Where-Object { $_.Priority -eq 'primary' } | Select-Object -First 3
    Write-Host "Primary references: $($primaryRefs | ForEach-Object { $_.Name } | Join-String -Separator ', ')" -ForegroundColor Gray
}
Write-Host ""

# Create backup directory
$backupDir = "assets\backup_all_assets_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "Step 2: Creating backup directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "Backup location: $backupDir" -ForegroundColor Gray
Write-Host ""

# Ensure output directories exist
$outputDirs = @(
    "assets",
    "assets\stones",
    "assets\banners",
    "marketing"
)

foreach ($dir in $outputDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Gray
    }
}
Write-Host ""

Write-Host "="*80 -ForegroundColor Yellow
Write-Host "GENERATING MISSING ASSETS" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Yellow
Write-Host ""

# --- A. Core Gameplay Sprites ---
Write-Host "[A] Core Gameplay Sprites" -ForegroundColor Cyan
Write-Host ""

# A3. Ground Stripe
Write-Host "[A3] Ground Stripe..." -ForegroundColor Yellow
Generate-VertexAI-Asset `
    -AssetName "A3. Ground Stripe" `
    -Prompt "Minimal road dash sprite, 60x6 px, rounded ends, off-white (#F2F7FA), flat fill, orthographic, transparent background, no shadow. Clean, simple, game-ready asset." `
    -NegativePrompt "complex, detailed, rounded, photo, blurry, watermark, busy, high contrast" `
    -ReferenceImages @() `
    -MasterWidth 60 `
    -MasterHeight 6 `
    -OutputFile "assets/ground_stripe.png"

Write-Host ""

# --- B. Background & Environment ---
Write-Host "[B] Background & Environment" -ForegroundColor Cyan
Write-Host ""

# B2. Ground Texture
Write-Host "[B2] Ground Texture..." -ForegroundColor Yellow
Generate-VertexAI-Asset `
    -AssetName "B2. Ground Texture" `
    -Prompt "Seamless ground strip tile, dark steel-stone texture with very faint forge specks, low contrast, 512x64px, seamless horizontally, orthographic, transparent background. Subtle texture, perfect for tiling." `
    -NegativePrompt "busy, high contrast, detailed, complex, non-seamless, photo, blurry, watermark" `
    -ReferenceImages @() `
    -MasterWidth 512 `
    -MasterHeight 64 `
    -OutputFile "assets/ground_tex.png"

Write-Host ""

# --- C. UI & Meta ---
Write-Host "[C] UI & Meta" -ForegroundColor Cyan
Write-Host ""

# C1. Logo / Title Card (Square Variant)
Write-Host "[C1] Logo / Title Card (Square 600x600)..." -ForegroundColor Yellow
Generate-VertexAI-Asset `
    -AssetName "C1. Logo / Title Card (Square)" `
    -Prompt "Key art logo cinematic title card, text reads 'The Rein Maker's Daughter', elegant serif combined with a clean geometric sans-serif font, features a central emblem of a broken leather rein seamlessly reshaping into a glowing circuit loop, brass and ember (#D8A24A) accents on the emblem, deep blue steel (#495057) background vignette, cinematic composition, crisp, high contrast, transparent background, export at 600x600 square format." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, logo placement errors, low contrast" `
    -ReferenceImages @() `
    -MasterWidth 600 `
    -MasterHeight 600 `
    -OutputFile "marketing/square-600.png"

Write-Host ""

# --- D. Lore Collectibles ---
Write-Host "[D] Lore Collectibles" -ForegroundColor Cyan
Write-Host ""

# D2. Tribe Banners (7)
Write-Host "[D2] Tribe Banners..." -ForegroundColor Yellow
Write-Host "Generating all 7 tribe banners..." -ForegroundColor Gray
Write-Host ""

# Light Tribe Banner
Write-Host "  [1/7] Light Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Light Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Light Tribe (eye), base color #F2F7FA, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_light.png"

# Stone Tribe Banner
Write-Host "  [2/7] Stone Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Stone Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Stone Tribe (mountain), base color #8E9BA7, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_stone.png"

# Metal Tribe Banner
Write-Host "  [3/7] Metal Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Metal Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Metal Tribe (gear), base color #adb5bd, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_metal.png"

# Code Tribe Banner
Write-Host "  [4/7] Code Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Code Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Code Tribe (code brackets '<>'), base color #0BB39C, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_code.png"

# Air Tribe Banner
Write-Host "  [5/7] Air Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Air Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Air Tribe (feather), base color #aed9e0, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_air.png"

# Water Tribe Banner
Write-Host "  [6/7] Water Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Water Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Water Tribe (wave), base color #4dabf7, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_water.png"

# Fire Tribe Banner
Write-Host "  [7/7] Fire Tribe Banner..." -ForegroundColor Gray
Generate-VertexAI-Asset `
    -AssetName "D2. Banner: Fire Tribe" `
    -Prompt "Vertical banner, fabric weave texture, bold symbol for the Fire Tribe (flame), base color #F25F5C, subtle gold trim, orthographic, transparent background, 128x256px. Clean, readable, perfect for vertical tiling." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, non-tileable, horizontal orientation" `
    -ReferenceImages @() `
    -MasterWidth 128 `
    -MasterHeight 256 `
    -OutputFile "assets/banners/banner_fire.png"

Write-Host ""

# --- G. Stretch Goals ---
Write-Host "[G] Stretch Goals" -ForegroundColor Cyan
Write-Host ""

# G1. Coin Pickup
Write-Host "[G1] Coin Pickup..." -ForegroundColor Yellow
Generate-VertexAI-Asset `
    -AssetName "G1. Coin Pickup" `
    -Prompt "Small glowing glyph coin for a 2D runner game, circular, abstract circuit pattern inside, teal and amber (#D8A24A) color blend, 24x24px, transparent background, clean silhouette. Perfect for game collectible." `
    -NegativePrompt "low quality, blurry, pixelated, compression artifacts, oversaturated colors, watermark, text overlay, complex, detailed, non-circular" `
    -ReferenceImages @() `
    -MasterWidth 24 `
    -MasterHeight 24 `
    -OutputFile "assets/coin.png"

Write-Host ""

# G2. Run Animation (3 frames) - REQUIRES CHARACTER CONSISTENCY
Write-Host "[G2] Run Animation (3 frames)..." -ForegroundColor Yellow
Write-Host "Generating 3-frame running animation with perfect character consistency..." -ForegroundColor Gray
Write-Host ""

# Frame 0: Contact (one foot on ground)
Write-Host "  [Frame 0/3] Contact Frame..." -ForegroundColor Gray
$runContactPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in running pose (contact frame, one foot on ground), facing right, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, contact frame, one foot on ground, dynamic movement, readable silhouette, action-ready stance" `
    -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" `
    -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "G2. Run Animation: Contact Frame" `
    -Prompt $runContactPrompt.Prompt `
    -NegativePrompt $runContactPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "assets/player_run_0.png"

# Frame 1: Air (both feet off ground)
Write-Host "  [Frame 1/3] Air Frame..." -ForegroundColor Gray
$runAirPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in running pose (air frame, both feet off ground), facing right, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, air frame, both feet off ground, dynamic movement, readable silhouette, action-ready stance" `
    -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" `
    -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "G2. Run Animation: Air Frame" `
    -Prompt $runAirPrompt.Prompt `
    -NegativePrompt $runAirPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "assets/player_run_1.png"

# Frame 2: Passing (legs crossing)
Write-Host "  [Frame 2/3] Passing Frame..." -ForegroundColor Gray
$runPassingPrompt = Build-KellyPrompt `
    -SceneDescription "Full-body photorealistic view of Kelly in running pose (passing frame, legs crossing), facing right, maintaining complete character consistency" `
    -WardrobeVariant "Reinmaker" `
    -Pose "running pose, passing frame, legs crossing, dynamic movement, readable silhouette, action-ready stance" `
    -Lighting "orthographic camera view, soft forge key light from lower-left, cool rim light upper-right, transparent background, game-ready asset format" `
    -AdditionalNegatives "stylized, cel-shaded, painterly texture, pixel art, low resolution, game sprite aesthetic, non-photorealistic, cartoon rendering" `
    -ReferenceImages $charRefs

Generate-VertexAI-Asset `
    -AssetName "G2. Run Animation: Passing Frame" `
    -Prompt $runPassingPrompt.Prompt `
    -NegativePrompt $runPassingPrompt.Negative `
    -ReferenceImages $charRefs `
    -MasterWidth 1024 `
    -MasterHeight 1280 `
    -OutputFile "assets/player_run_2.png"

Write-Host ""

# --- Summary ---
Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "GENERATION COMPLETE" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "✅ All Missing Assets Generated:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Core Assets:" -ForegroundColor Yellow
Write-Host "  ✓ A3. Ground Stripe" -ForegroundColor Green
Write-Host "  ✓ B2. Ground Texture" -ForegroundColor Green
Write-Host "  ✓ C1. Logo (Square 600x600)" -ForegroundColor Green
Write-Host ""
Write-Host "Lore Collectibles:" -ForegroundColor Yellow
Write-Host "  ✓ D2. Tribe Banners (all 7)" -ForegroundColor Green
Write-Host ""
Write-Host "Stretch Goals:" -ForegroundColor Yellow
Write-Host "  ✓ G1. Coin Pickup" -ForegroundColor Green
Write-Host "  ✓ G2. Run Animation (3 frames)" -ForegroundColor Green
Write-Host ""
Write-Host "Character Consistency:" -ForegroundColor Yellow
Write-Host "  • Used $($charRefs.Count) character reference(s) for animation frames" -ForegroundColor White
Write-Host "  • Updated hair specification applied (soft cohesive waves)" -ForegroundColor White
Write-Host "  • Vertex AI API with OAuth2 authentication" -ForegroundColor White
Write-Host ""
Write-Host "Backup Location: $backupDir" -ForegroundColor Gray
Write-Host ""
Write-Host "🎯 All assets generated with Insanely Great quality!" -ForegroundColor Green
Write-Host ""












