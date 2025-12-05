# ============================================================================
# Upload Unity Build to Cloudflare R2 (PowerShell)
# Uploads Brotli-compressed Unity WebGL files to R2 bucket
# ============================================================================

param(
    [string]$BuildDir = "digital-kelly\engines\Kelly_Engine_V2\onlykelly\Builds\WebGL\Build",
    [string]$BuildName = "Kelly_Web_Build",
    [string]$R2Bucket = "curious-kelly-unity",
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

if (-not $Version) {
    $Version = Get-Date -Format "yyyyMMdd-HHmmss"
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Unity R2 Upload" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Build Name: $BuildName"
Write-Host "Build Dir:  $BuildDir"
Write-Host "R2 Bucket:  $R2Bucket"
Write-Host "Version:    $Version"
Write-Host ""

# Check if wrangler is installed
$wranglerCmd = Get-Command wrangler -ErrorAction SilentlyContinue
if (-not $wranglerCmd) {
    Write-Host "ERROR: Wrangler CLI is not installed." -ForegroundColor Red
    Write-Host "   Install with: npm install -g wrangler" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
try {
    wrangler whoami 2>$null | Out-Null
} catch {
    Write-Host "ERROR: Not authenticated with Cloudflare." -ForegroundColor Red
    Write-Host "   Run: wrangler login" -ForegroundColor Yellow
    exit 1
}

# Verify build directory
if (-not (Test-Path $BuildDir)) {
    Write-Host "ERROR: Build directory not found: $BuildDir" -ForegroundColor Red
    exit 1
}

# Required files
$requiredFiles = @(
    "$BuildName.loader.js",
    "$BuildName.data.br",
    "$BuildName.framework.js.br",
    "$BuildName.wasm.br"
)

Write-Host "=== Verifying files ===" -ForegroundColor Yellow
$missing = 0
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $BuildDir $file
    if (Test-Path $fullPath) {
        Write-Host "  OK $file" -ForegroundColor Green
    } else {
        Write-Host "  MISSING $file" -ForegroundColor Red
        $missing++
    }
}

if ($missing -gt 0) {
    Write-Host ""
    Write-Host "ERROR: $missing required files are missing." -ForegroundColor Red
    Write-Host "   Run compress-unity-build.ps1 first to generate Brotli files." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== Uploading to R2 (root) ===" -ForegroundColor Yellow

# Upload to root path (current/latest)
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $BuildDir $file
    Write-Host "  Uploading $file..."
    wrangler r2 object put "$R2Bucket/$file" --file="$fullPath"
}

Write-Host ""
Write-Host "=== Uploading to R2 (versioned: $Version) ===" -ForegroundColor Yellow

# Upload to versioned path
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $BuildDir $file
    Write-Host "  Uploading $Version/$file..."
    wrangler r2 object put "$R2Bucket/$Version/$file" --file="$fullPath"
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "    Upload Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "CDN URL: https://unity-cdn.nicoletterankin.workers.dev" -ForegroundColor Cyan
Write-Host ""
Write-Host "Latest files:"
Write-Host "  https://unity-cdn.nicoletterankin.workers.dev/$BuildName.loader.js"
Write-Host ""
Write-Host "Versioned files:"
Write-Host "  https://unity-cdn.nicoletterankin.workers.dev/$Version/$BuildName.loader.js"
Write-Host ""
Write-Host "Test with:" -ForegroundColor Yellow
Write-Host "  curl -I https://unity-cdn.nicoletterankin.workers.dev/$BuildName.loader.js"


