Write-Host "=== UI-Tars Creative Pipeline Setup ===" -ForegroundColor Cyan

$root = "C:\iLearnStudio"
Write-Host "Setting up pipeline at: $root" -ForegroundColor Green

# Create root directory
New-Item -ItemType Directory -Path $root -Force | Out-Null
Write-Host "✓ Created root directory: $root" -ForegroundColor Green

# Copy all directories
Copy-Item -Path "scripts" -Destination "$root\scripts" -Recurse -Force
Write-Host "✓ Copied scripts directory" -ForegroundColor Green

Copy-Item -Path "tools" -Destination "$root\tools" -Recurse -Force
Write-Host "✓ Copied tools directory" -ForegroundColor Green

Copy-Item -Path "config" -Destination "$root\config" -Recurse -Force
Write-Host "✓ Copied config directory" -ForegroundColor Green

Copy-Item -Path "docs" -Destination "$root\docs" -Recurse -Force
Write-Host "✓ Copied docs directory" -ForegroundColor Green

# Copy README
Copy-Item -Path "README.md" -Destination "$root\README.md" -Force
Write-Host "✓ Copied README.md" -ForegroundColor Green

# Update bootstrap script for C: drive
$bootstrapScript = "$root\scripts\00_bootstrap.ps1"
$content = Get-Content $bootstrapScript -Raw
$content = $content -replace 'D:\\iLearnStudio', 'C:\iLearnStudio'
Set-Content -Path $bootstrapScript -Value $content
Write-Host "✓ Updated bootstrap script for C: drive" -ForegroundColor Green

# Run bootstrap
Write-Host "`nRunning bootstrap script..." -ForegroundColor Yellow
& "$root\scripts\00_bootstrap.ps1"
Write-Host "✓ Bootstrap completed" -ForegroundColor Green

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "✅ Creative pipeline setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Navigate to: $root" -ForegroundColor White
Write-Host "2. Run: .\scripts\run_pipeline.ps1" -ForegroundColor White
Write-Host "3. Install Reallusion software" -ForegroundColor White
Write-Host "4. Add your audio files" -ForegroundColor White

Write-Host "`nQuick start:" -ForegroundColor Cyan
Write-Host "cd $root" -ForegroundColor Gray
Write-Host ".\scripts\run_pipeline.ps1" -ForegroundColor Gray
