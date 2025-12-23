# Setup Vercel Edge Config and Blob Storage
# PowerShell version for Windows

Write-Host "🚀 Setting up Vercel Edge Optimization..." -ForegroundColor Cyan

# Check if logged in
try {
    $whoami = vercel whoami 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Not logged into Vercel. Please run: vercel login" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Logged into Vercel" -ForegroundColor Green
} catch {
    Write-Host "❌ Error checking Vercel login" -ForegroundColor Red
    exit 1
}

# Generate a random secret for Edge Config sync
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$EDGE_CONFIG_SYNC_SECRET = [Convert]::ToHexString($bytes).ToLower()
Write-Host "🔐 Generated Edge Config sync secret: $EDGE_CONFIG_SYNC_SECRET" -ForegroundColor Yellow

Write-Host ""
Write-Host "⚠️  Edge Config and Blob Storage must be created via Vercel Dashboard" -ForegroundColor Yellow
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to: https://vercel.com/dashboard" -ForegroundColor White
Write-Host "2. Select your project: curiouskelly" -ForegroundColor White
Write-Host "3. Go to Storage → Edge Config → Create" -ForegroundColor White
Write-Host "   - Name: curious-kelly-lessons" -ForegroundColor Gray
Write-Host "   - Copy the connection string" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Go to Storage → Blob → Create buckets:" -ForegroundColor White
Write-Host "   - curious-kelly-videos" -ForegroundColor Gray
Write-Host "   - curious-kelly-audio" -ForegroundColor Gray
Write-Host "   - curious-kelly-visuals" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Add environment variables in Settings → Environment Variables:" -ForegroundColor White
Write-Host "   - EDGE_CONFIG=<connection-string-from-step-3>" -ForegroundColor Gray
Write-Host "   - EDGE_CONFIG_SYNC_SECRET=$EDGE_CONFIG_SYNC_SECRET" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ Setup script complete!" -ForegroundColor Green
Write-Host ""
Write-Host "After completing Dashboard setup, run:" -ForegroundColor Cyan
Write-Host "  npm run sync-edge-config" -ForegroundColor White

