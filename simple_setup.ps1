# Simple Creative Pipeline Setup
Write-Host "=== UI-Tars Creative Pipeline Setup ===" -ForegroundColor Cyan

$root = "D:\iLearnStudio"
Write-Host "Setting up pipeline at: $root" -ForegroundColor Green

# Create root directory
if (-not (Test-Path $root)) {
    New-Item -ItemType Directory -Path $root -Force | Out-Null
    Write-Host "✓ Created root directory: $root" -ForegroundColor Green
} else {
    Write-Host "✓ Root directory already exists: $root" -ForegroundColor Green
}

# Copy scripts directory
if (Test-Path "scripts") {
    Copy-Item -Path "scripts" -Destination "$root\scripts" -Recurse -Force
    Write-Host "✓ Copied scripts directory" -ForegroundColor Green
} else {
    Write-Host "⚠ Scripts directory not found" -ForegroundColor Yellow
}

# Copy tools directory
if (Test-Path "tools") {
    Copy-Item -Path "tools" -Destination "$root\tools" -Recurse -Force
    Write-Host "✓ Copied tools directory" -ForegroundColor Green
} else {
    Write-Host "⚠ Tools directory not found" -ForegroundColor Yellow
}

# Copy config directory
if (Test-Path "config") {
    Copy-Item -Path "config" -Destination "$root\config" -Recurse -Force
    Write-Host "✓ Copied config directory" -ForegroundColor Green
} else {
    Write-Host "⚠ Config directory not found" -ForegroundColor Yellow
}

# Copy docs directory
if (Test-Path "docs") {
    Copy-Item -Path "docs" -Destination "$root\docs" -Recurse -Force
    Write-Host "✓ Copied docs directory" -ForegroundColor Green
} else {
    Write-Host "⚠ Docs directory not found" -ForegroundColor Yellow
}

# Copy README
if (Test-Path "README.md") {
    Copy-Item -Path "README.md" -Destination "$root\README.md" -Force
    Write-Host "✓ Copied README.md" -ForegroundColor Green
}

# Run bootstrap script
Write-Host "`nRunning bootstrap script..." -ForegroundColor Yellow
try {
    & "$root\scripts\00_bootstrap.ps1"
    Write-Host "✓ Bootstrap completed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Bootstrap had issues: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "✅ Creative pipeline setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Navigate to: $root" -ForegroundColor White
Write-Host "2. Run: .\scripts\run_pipeline.ps1" -ForegroundColor White
Write-Host "3. Install Reallusion software" -ForegroundColor White
Write-Host "4. Add your audio files" -ForegroundColor White

Write-Host "`nQuick start:" -ForegroundColor Cyan
Write-Host "cd `"$root`"" -ForegroundColor Gray
Write-Host ".\scripts\run_pipeline.ps1" -ForegroundColor Gray
