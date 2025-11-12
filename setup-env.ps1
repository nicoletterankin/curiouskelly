#!/usr/bin/env pwsh
# Environment Setup Script for Curious Kellly
# Creates .env files from examples if they don't exist

Write-Host "🔧 Curious Kellly - Environment Setup" -ForegroundColor Cyan
Write-Host "=" * 50

# Backend .env
$backendEnv = "curious-kellly/backend/.env"
$backendExample = "curious-kellly/backend/.env.example"

if (-not (Test-Path $backendEnv)) {
    if (Test-Path $backendExample) {
        Copy-Item $backendExample $backendEnv
        Write-Host "✅ Created $backendEnv from example" -ForegroundColor Green
        Write-Host "⚠️  Please edit $backendEnv and add your API keys!" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  .env.example not found for backend" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Backend .env already exists" -ForegroundColor Green
}

# Mobile .env
$mobileEnv = "curious-kellly/mobile/.env"
$mobileExample = "curious-kellly/mobile/.env.example"

if (-not (Test-Path $mobileEnv)) {
    if (Test-Path $mobileExample) {
        Copy-Item $mobileExample $mobileEnv
        Write-Host "✅ Created $mobileEnv from example" -ForegroundColor Green
        Write-Host "⚠️  Please edit $mobileEnv and set API_BASE_URL!" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  .env.example not found for mobile" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Mobile .env already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Edit backend/.env - Add OPENAI_API_KEY" -ForegroundColor White
Write-Host "2. Edit mobile/.env - Set API_BASE_URL to your backend URL" -ForegroundColor White
Write-Host "3. Run: cd curious-kellly/backend && node scripts/verify-env.js" -ForegroundColor White
Write-Host ""







