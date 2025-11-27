# KELLY V2 - VERCEL DEPLOYMENT SCRIPT
# December 17, 2025 Launch

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "KELLY V2 DEPLOYMENT TO VERCEL" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ProjectRoot = $PSScriptRoot
$BuildPath = Join-Path $ProjectRoot "Builds\WebGL"
$VercelConfig = Join-Path $ProjectRoot "vercel.json"

# Step 1: Verify Build Exists
Write-Host "[Step 1] Verifying build..." -ForegroundColor Yellow

if (-not (Test-Path $BuildPath)) {
    Write-Host "ERROR: Build folder not found!" -ForegroundColor Red
    Write-Host "   Expected: $BuildPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Run Unity build first:" -ForegroundColor Yellow
    Write-Host "   Kelly > Build > Build WebGL (Production)" -ForegroundColor Yellow
    exit 1
}

$IndexFile = Join-Path $BuildPath "index.html"
if (-not (Test-Path $IndexFile)) {
    Write-Host "ERROR: index.html not found in build!" -ForegroundColor Red
    Write-Host "   The build may be incomplete." -ForegroundColor Red
    exit 1
}

$BuildFolder = Join-Path $BuildPath "Build"
$BuildFiles = Get-ChildItem $BuildFolder -ErrorAction SilentlyContinue
if ($BuildFiles.Count -eq 0) {
    Write-Host "ERROR: Build/ subfolder is empty!" -ForegroundColor Red
    exit 1
}

Write-Host "OK Build verified: $BuildPath" -ForegroundColor Green
Write-Host "   Files found: $($BuildFiles.Count) in Build/" -ForegroundColor Gray
Write-Host ""

# Step 2: Verify Vercel Config
Write-Host "[Step 2] Checking Vercel configuration..." -ForegroundColor Yellow

if (-not (Test-Path $VercelConfig)) {
    Write-Host "ERROR: vercel.json not found!" -ForegroundColor Red
    exit 1
}

Write-Host "OK vercel.json found" -ForegroundColor Green
Write-Host ""

# Step 3: Check Vercel CLI
Write-Host "[Step 3] Checking Vercel CLI..." -ForegroundColor Yellow

$VercelInstalled = Get-Command vercel -ErrorAction SilentlyContinue
if (-not $VercelInstalled) {
    Write-Host "Vercel CLI not found. Installing..." -ForegroundColor Yellow
    npm install -g vercel
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install Vercel CLI" -ForegroundColor Red
        Write-Host "   Run manually: npm install -g vercel" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "OK Vercel CLI ready" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy
Write-Host "[Step 4] Deploying to Vercel..." -ForegroundColor Yellow
Write-Host ""

Set-Location $ProjectRoot

# Run Vercel deploy
Write-Host "Running: vercel --prod" -ForegroundColor Gray
Write-Host ""

vercel --prod

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "    DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Kelly is now live! Check the URL above." -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "DEPLOYMENT FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Run vercel login if not authenticated" -ForegroundColor White
    Write-Host "2. Check vercel.json for errors" -ForegroundColor White
    Write-Host "3. Ensure build files are not too large" -ForegroundColor White
    Write-Host ""
    exit 1
}
