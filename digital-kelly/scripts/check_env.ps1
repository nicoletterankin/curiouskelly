# Kelly OS - Environment Check (PowerShell)

Write-Host "🔍 Kelly OS - Environment Check" -ForegroundColor Cyan
Write-Host "================================"
Write-Host ""

if (-not (Test-Path .env)) {
    Write-Host "⚠️  .env file not found" -ForegroundColor Yellow
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host "✅ Created .env from .env.example" -ForegroundColor Green
        Write-Host "⚠️  Please edit .env and fill in ELEVENLABS_API_KEY" -ForegroundColor Yellow
        exit 1
    } else {
        Write-Host "❌ .env.example not found" -ForegroundColor Red
        exit 1
    }
}

Get-Content .env | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]
        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$key = $env:ELEVENLABS_API_KEY

if ([string]::IsNullOrWhiteSpace($key) -or $key -eq "__PUT_KEY_IN_YOUR_LOCAL_.env__") {
    Write-Host "❌ ELEVENLABS_API_KEY not set in .env" -ForegroundColor Red
    Write-Host "   Edit .env and add your ElevenLabs API key"
    exit 1
}

if ($key.Length -lt 20) {
    Write-Host "⚠️  ELEVENLABS_API_KEY looks suspiciously short" -ForegroundColor Yellow
    Write-Host "   Verify it's correct"
    exit 1
}

Write-Host "✅ .env file exists" -ForegroundColor Green
Write-Host "✅ ELEVENLABS_API_KEY is set (length: $($key.Length))" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Environment check passed!" -ForegroundColor Green


















