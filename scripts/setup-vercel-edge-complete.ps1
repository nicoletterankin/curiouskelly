# Complete Vercel Edge Setup Script
# This script guides you through Dashboard setup and verifies configuration

Write-Host "🚀 Vercel Edge Optimization Setup" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if logged in
Write-Host "Step 1: Verifying Vercel login..." -ForegroundColor Yellow
try {
    $whoami = vercel whoami 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Logged in as: $whoami" -ForegroundColor Green
    } else {
        Write-Host "❌ Not logged in. Run: vercel login" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error checking login" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Dashboard Configuration Required" -ForegroundColor Yellow
Write-Host ""
Write-Host "Please complete these steps in Vercel Dashboard:" -ForegroundColor White
Write-Host ""
Write-Host "1. Open: https://vercel.com/dashboard" -ForegroundColor Cyan
Write-Host "2. Select project: curiouskelly" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Create Edge Config:" -ForegroundColor Yellow
Write-Host "   - Go to: Storage → Edge Config" -ForegroundColor Gray
Write-Host "   - Click: 'Create Edge Config'" -ForegroundColor Gray
Write-Host "   - Name: curious-kelly-lessons" -ForegroundColor Gray
Write-Host "   - Click: 'Create'" -ForegroundColor Gray
Write-Host "   - COPY the Connection String" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Create Blob Storage Buckets:" -ForegroundColor Yellow
Write-Host "   - Go to: Storage → Blob" -ForegroundColor Gray
Write-Host "   - Click: 'Create Bucket' (3 times)" -ForegroundColor Gray
Write-Host "   - Names: curious-kelly-videos, curious-kelly-audio, curious-kelly-visuals" -ForegroundColor Gray
Write-Host "   - Set all to Public: Yes" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Set Environment Variables:" -ForegroundColor Yellow
Write-Host "   - Go to: Settings → Environment Variables" -ForegroundColor Gray
Write-Host "   - Add EDGE_CONFIG (paste connection string from step 3)" -ForegroundColor Gray
Write-Host "   - Add EDGE_CONFIG_SYNC_SECRET (generate below)" -ForegroundColor Gray
Write-Host ""

# Generate secret
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$secret = [Convert]::ToHexString($bytes).ToLower()

Write-Host "Generated Sync Secret: $secret" -ForegroundColor Green
Write-Host ""
Write-Host "Press Enter after completing Dashboard setup..." -ForegroundColor Yellow
Read-Host

# Step 3: Verify configuration
Write-Host ""
Write-Host "Step 3: Verifying configuration..." -ForegroundColor Yellow

# Check environment variables
try {
    $envs = vercel env ls 2>&1
    if ($envs -match "EDGE_CONFIG") {
        Write-Host "✅ EDGE_CONFIG found" -ForegroundColor Green
    } else {
        Write-Host "⚠️  EDGE_CONFIG not found" -ForegroundColor Yellow
    }
    
    if ($envs -match "EDGE_CONFIG_SYNC_SECRET") {
        Write-Host "✅ EDGE_CONFIG_SYNC_SECRET found" -ForegroundColor Green
    } else {
        Write-Host "⚠️  EDGE_CONFIG_SYNC_SECRET not found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not verify environment variables" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run initial sync: npm run sync-edge-config" -ForegroundColor White
Write-Host "  2. Migrate assets: npx tsx scripts/migrate-to-blob.ts --dry-run" -ForegroundColor White

