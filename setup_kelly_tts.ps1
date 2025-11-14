# Kelly TTS Setup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Kelly TTS Setup - ElevenLabs Integration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$envPath = "curious-kellly\backend\.env"
$envExamplePath = "curious-kellly\backend\.env.example"

if (Test-Path $envPath) {
    Write-Host ".env file exists" -ForegroundColor Green
    $envContent = Get-Content $envPath -Raw
    if ($envContent -match "ELEVENLABS_API_KEY\s*=\s*[^\s]+" -and $envContent -notmatch "ELEVENLABS_API_KEY\s*=\s*your_") {
        Write-Host "ELEVENLABS_API_KEY is configured" -ForegroundColor Green
    } else {
        Write-Host "ELEVENLABS_API_KEY needs to be set in .env file" -ForegroundColor Yellow
        Write-Host "Open: $envPath" -ForegroundColor Gray
    }
} else {
    Write-Host ".env file not found" -ForegroundColor Red
    if (Test-Path $envExamplePath) {
        Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item $envExamplePath $envPath
        Write-Host "Created .env file - please add your API keys" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env and add your ELEVENLABS_API_KEY" -ForegroundColor White
Write-Host "2. Run: .\start_backend.ps1" -ForegroundColor White
Write-Host "3. In Unity: Add KellyTTSClient component to kelly_character" -ForegroundColor White
Write-Host ""











