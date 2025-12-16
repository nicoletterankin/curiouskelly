# ============================================
# DEPLOY KELLY CHAIR TO WEBSITE
# ============================================
# Run this after Unity WebGL build completes
# ============================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   DEPLOYING KELLY CHAIR BUILD" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$sourcePath = "C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\Builds\WebGL\kelly-chair"
$destPath = "C:\Users\user\UI-TARS-desktop\public\unity\kelly-chair"
$dailyMarketingPath = "C:\Users\user\UI-TARS-desktop\daily-lesson-marketing\public\unity\kelly-chair"

# Check if source exists
if (-not (Test-Path $sourcePath)) {
    Write-Host "[ERROR] Build folder not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected location:" -ForegroundColor Yellow
    Write-Host $sourcePath -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Did you complete the Unity build?" -ForegroundColor Yellow
    Write-Host "Build output should be in: Builds/WebGL/kelly-chair" -ForegroundColor Yellow
    exit 1
}

# Check for index.html
if (-not (Test-Path "$sourcePath\index.html")) {
    Write-Host "[ERROR] index.html not found in build!" -ForegroundColor Red
    Write-Host "The build may have failed. Check Unity console for errors." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Found Unity build at:" -ForegroundColor Green
Write-Host "   $sourcePath" -ForegroundColor White
Write-Host ""

# Copy kbridge.js from existing build
$kbridgeSource = "C:\Users\user\UI-TARS-desktop\public\unity\kelly-v1\kbridge.js"
if (Test-Path $kbridgeSource) {
    Write-Host "Copying kbridge.js to new build..." -ForegroundColor Cyan
    Copy-Item $kbridgeSource "$sourcePath\kbridge.js" -Force
    Write-Host "   [OK] kbridge.js copied" -ForegroundColor Green
}

# Deploy to public/unity/kelly-chair
Write-Host ""
Write-Host "Deploying to public/unity/kelly-chair..." -ForegroundColor Cyan
if (Test-Path $destPath) {
    Remove-Item $destPath -Recurse -Force
}
Copy-Item $sourcePath $destPath -Recurse
Write-Host "   [OK] Deployed to public folder" -ForegroundColor Green

# Also deploy to daily-lesson-marketing if it exists
if (Test-Path (Split-Path $dailyMarketingPath -Parent)) {
    Write-Host ""
    Write-Host "Deploying to daily-lesson-marketing..." -ForegroundColor Cyan
    if (Test-Path $dailyMarketingPath) {
        Remove-Item $dailyMarketingPath -Recurse -Force
    }
    Copy-Item $sourcePath $dailyMarketingPath -Recurse
    Write-Host "   [OK] Deployed to daily-lesson-marketing" -ForegroundColor Green
}

# Success!
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Kelly in her chair is now available at:" -ForegroundColor White
Write-Host ""
Write-Host "   /unity/kelly-chair/index.html" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use in your lesson player, update the iframe src:" -ForegroundColor White
Write-Host ""
Write-Host "   <iframe src='/unity/kelly-chair/index.html'></iframe>" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or use the progressive loader:" -ForegroundColor White
Write-Host ""
Write-Host "   <script src='/unity/kelly-loader.js'></script>" -ForegroundColor Yellow
Write-Host "   <script>KellyLoader.load('#kelly-container');</script>" -ForegroundColor Yellow
Write-Host ""
Write-Host "Kelly is in the building!" -ForegroundColor Magenta
Write-Host ""
