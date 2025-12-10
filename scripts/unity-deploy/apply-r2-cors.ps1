# ═══════════════════════════════════════════════════════════════════
# Apply CORS Policy to R2 Bucket for Unity WebGL Assets
# ═══════════════════════════════════════════════════════════════════
#
# This script applies CORS configuration to the curious-kelly-unity R2 bucket
# to allow Unity WebGL builds to load from curiouskelly.com
#
# Prerequisites:
# - Wrangler CLI installed (npm install -g wrangler)
# - Logged into Cloudflare (wrangler login)
#
# Usage: .\scripts\unity-deploy\apply-r2-cors.ps1
# ═══════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# Configuration
$BUCKET_NAME = "curious-kelly-unity"
$CORS_FILE = "infrastructure/cloudflare/r2-cors-policy.json"
$ACCOUNT_ID = "47ebb2a1adc311cb106acc89720e352c"

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Applying CORS Policy to R2 Bucket: $BUCKET_NAME" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if wrangler is installed
try {
    $wranglerVersion = wrangler --version
    Write-Host "✅ Wrangler installed: $wranglerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Wrangler not found. Install with: npm install -g wrangler" -ForegroundColor Red
    exit 1
}

# Check if logged in
Write-Host ""
Write-Host "Checking Cloudflare authentication..." -ForegroundColor Yellow
try {
    wrangler whoami 2>$null
    Write-Host "✅ Logged into Cloudflare" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Not logged into Cloudflare. Running wrangler login..." -ForegroundColor Yellow
    wrangler login
}

# Check if CORS file exists
if (-not (Test-Path $CORS_FILE)) {
    Write-Host "❌ CORS policy file not found: $CORS_FILE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "CORS Policy to apply:" -ForegroundColor Yellow
Get-Content $CORS_FILE | Write-Host

Write-Host ""
Write-Host "Applying CORS policy to bucket '$BUCKET_NAME'..." -ForegroundColor Yellow

# Apply CORS using wrangler
# Note: wrangler r2 bucket cors put requires the policy via stdin or --file
try {
    # Using wrangler to apply CORS
    Get-Content $CORS_FILE | wrangler r2 bucket cors put $BUCKET_NAME --account-id $ACCOUNT_ID
    Write-Host ""
    Write-Host "✅ CORS policy applied successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to apply CORS policy: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Apply manually via Cloudflare Dashboard:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://dash.cloudflare.com/$ACCOUNT_ID/r2/default/buckets/$BUCKET_NAME" -ForegroundColor White
    Write-Host "2. Click 'Settings' tab" -ForegroundColor White
    Write-Host "3. Scroll to 'CORS Policy'" -ForegroundColor White
    Write-Host "4. Click 'Edit CORS Policy'" -ForegroundColor White
    Write-Host "5. Paste the contents of $CORS_FILE" -ForegroundColor White
    Write-Host "6. Click 'Save'" -ForegroundColor White
    exit 1
}

# Verify CORS was applied
Write-Host ""
Write-Host "Verifying CORS policy..." -ForegroundColor Yellow
try {
    wrangler r2 bucket cors list $BUCKET_NAME --account-id $ACCOUNT_ID
    Write-Host ""
    Write-Host "✅ CORS verification complete!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not verify CORS policy. Please check manually in the dashboard." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Next Steps:" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "1. Test locally: npm run dev, navigate to /learn.html, click 3D button" -ForegroundColor White
Write-Host "2. Check browser console for CORS errors" -ForegroundColor White
Write-Host "3. If working, deploy to production" -ForegroundColor White
Write-Host ""




