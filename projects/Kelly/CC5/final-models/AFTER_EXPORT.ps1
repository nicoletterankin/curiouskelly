# AFTER EXPORT - Run this after kelly_intro_full.fbx is exported
# This copies everything to the right places

Write-Host "Kelly Intro Deployment Script" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

$sourceDir = "c:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models"
$unityAssets = "c:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\Assets"
$streamingAssets = "$unityAssets\StreamingAssets\kelly-intro"

# Check if FBX exists
$fbxPath = "$sourceDir\kelly_intro_full.fbx"
if (Test-Path $fbxPath) {
    Write-Host "[OK] Found: kelly_intro_full.fbx" -ForegroundColor Green
    
    # Copy FBX to Unity Assets
    Write-Host "Copying FBX to Unity Assets..." -ForegroundColor Yellow
    Copy-Item $fbxPath "$unityAssets\kelly_intro_full.fbx" -Force
    Write-Host "[OK] FBX copied to Unity Assets" -ForegroundColor Green
    
    # Check for .fbm folder (textures)
    $fbmPath = "$sourceDir\kelly_intro_full.fbm"
    if (Test-Path $fbmPath) {
        Write-Host "Copying textures folder..." -ForegroundColor Yellow
        Copy-Item $fbmPath "$unityAssets\kelly_intro_full.fbm" -Recurse -Force
        Write-Host "[OK] Textures copied" -ForegroundColor Green
    }
} else {
    Write-Host "[WAITING] kelly_intro_full.fbx not found yet" -ForegroundColor Yellow
    Write-Host "Export from iClone first, then run this script again" -ForegroundColor Yellow
}

# Check audio
$audioPath = "$streamingAssets\kelly_intro_audio.mp3"
if (Test-Path $audioPath) {
    Write-Host "[OK] Audio already in place: kelly_intro_audio.mp3" -ForegroundColor Green
} else {
    Write-Host "[COPYING] Audio file..." -ForegroundColor Yellow
    Copy-Item "$sourceDir\kelly_intro_audio.mp3" $audioPath -Force
    Write-Host "[OK] Audio copied" -ForegroundColor Green
}

Write-Host ""
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Open Unity project: digital-kelly\engines\Kelly_Engine_V2\onlykelly" -ForegroundColor White
Write-Host "2. Wait for import (may take a few minutes)" -ForegroundColor White
Write-Host "3. If CCIC popup appears, choose 'High Quality (URP)'" -ForegroundColor White
Write-Host "4. Find kelly_intro_full in Assets" -ForegroundColor White
Write-Host "5. Drag to scene or update existing prefab" -ForegroundColor White
Write-Host "==============================" -ForegroundColor Cyan
