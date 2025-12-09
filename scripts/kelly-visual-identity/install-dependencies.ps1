# Kelly Visual Identity Pipeline - Dependency Installation
# Run this script to install all required dependencies

Write-Host "🎨 Kelly Visual Identity Pipeline - Installing Dependencies" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if Node.js is installed
Write-Host "`n📦 Checking Node.js..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Node.js $nodeVersion installed" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js not found. Please install from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Check if npm is installed
Write-Host "`n📦 Checking npm..." -ForegroundColor Yellow
$npmVersion = npm --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ npm $npmVersion installed" -ForegroundColor Green
} else {
    Write-Host "❌ npm not found" -ForegroundColor Red
    exit 1
}

# Install global dependencies
Write-Host "`n📦 Installing global dependencies..." -ForegroundColor Yellow
npm install -g tsx
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ tsx installed globally" -ForegroundColor Green
} else {
    Write-Host "⚠️  tsx installation failed, will use local version" -ForegroundColor Yellow
}

# Install project dependencies
Write-Host "`n📦 Installing project dependencies..." -ForegroundColor Yellow
npm install @google/generative-ai @aws-sdk/client-s3 @supabase/supabase-js dotenv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Project dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Install dev dependencies
Write-Host "`n📦 Installing dev dependencies..." -ForegroundColor Yellow
npm install --save-dev @types/node tsx typescript
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dev dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dev dependencies" -ForegroundColor Red
    exit 1
}

# Check for .env.local
Write-Host "`n🔐 Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env.local") {
    Write-Host "✅ .env.local found" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env.local not found" -ForegroundColor Yellow
    Write-Host "   Copy scripts/kelly-visual-identity/env-template.txt to .env.local" -ForegroundColor Yellow
    Write-Host "   and fill in your credentials" -ForegroundColor Yellow
}

# Summary
Write-Host "`n" + ("=" * 60)
Write-Host "📊 INSTALLATION SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host "✅ All dependencies installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Copy env-template.txt to .env.local and fill in credentials"
Write-Host "2. Run: npx tsx scripts/kelly-visual-identity/prepare-lora-dataset.ts"
Write-Host "3. Follow KELLY_VISUAL_IDENTITY_EXECUTION_CHECKLIST.md"
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   - scripts/kelly-visual-identity/README.md"
Write-Host "   - KELLY_VISUAL_IDENTITY_EXECUTION_CHECKLIST.md"
Write-Host ""








