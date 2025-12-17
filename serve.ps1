# Curious Kelly Local Development Server

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Curious Kelly Local Development Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting local server at " -NoNewline
Write-Host "http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "URL Parameters for testing:" -ForegroundColor Yellow
Write-Host "  ?autoplay=false  - Disable auto-advance"
Write-Host "  ?phase=N         - Start at phase N (0-6)"
Write-Host "  ?debug=true      - Enable debug logging"
Write-Host "  ?day=N           - Load specific day lesson"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host "============================================"
Write-Host ""

Set-Location -Path $PSScriptRoot\public
npx serve -l 3000





