# Deployment script for The Rein Maker's Daughter Runner Game
# This script builds the game and prepares it for deployment

Write-Host "🎮 Building The Rein Maker's Daughter..." -ForegroundColor Cyan
Write-Host ""

# Build the game
Write-Host "📦 Building game..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host ""
    
    # Create deployment package
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $zipName = "reinmaker-runner-game-$timestamp.zip"
    
    Write-Host "📦 Creating deployment package..." -ForegroundColor Yellow
    Compress-Archive -Path "dist\*" -DestinationPath $zipName -Force
    
    Write-Host "✅ Package created: $zipName" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "🚀 Ready to deploy to Itch.io!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "1. Go to https://itch.io/game/new" -ForegroundColor Gray
    Write-Host "2. Upload $zipName" -ForegroundColor Gray
    Write-Host "3. Set 'Kind of project' to HTML" -ForegroundColor Gray
    Write-Host "4. Check 'This file will be played in the browser'" -ForegroundColor Gray
    Write-Host "5. Set viewport to 800 x 600" -ForegroundColor Gray
    Write-Host "6. Save & Publish!" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📊 Build stats:" -ForegroundColor White
    $distSize = (Get-ChildItem -Path "dist" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "   Total size: $([math]::Round($distSize, 2)) MB" -ForegroundColor Gray
    
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}








