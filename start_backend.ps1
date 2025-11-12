# Start Backend Server for Kelly TTS
Write-Host "Starting Kelly TTS Backend Server..." -ForegroundColor Cyan
Write-Host ""

$backendPath = "curious-kellly\backend"

if (-not (Test-Path $backendPath)) {
    Write-Host "Backend directory not found: $backendPath" -ForegroundColor Red
    exit 1
}

$envPath = Join-Path $backendPath ".env"
if (-not (Test-Path $envPath)) {
    Write-Host ".env file not found!" -ForegroundColor Yellow
    Write-Host "Run: .\setup_kelly_tts.ps1 first" -ForegroundColor Gray
    exit 1
}

$nodeModulesPath = Join-Path $backendPath "node_modules"
if (-not (Test-Path $nodeModulesPath)) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    Set-Location $backendPath
    npm install
    Set-Location "..\..\"
}

Write-Host "Starting backend server..." -ForegroundColor Green
Write-Host "Server will run on: http://localhost:3000" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

Set-Location $backendPath
npm run dev










