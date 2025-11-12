# UI-Tars Creative Pipeline Setup Script
# This script sets up the complete CC5/iClone creative pipeline

param(
    [string]$TargetDrive = "D:",
    [switch]$DryRun,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
$root = "$TargetDrive\iLearnStudio"
$currentDir = Get-Location

Write-Host "=== UI-Tars Creative Pipeline Setup ===" -ForegroundColor Cyan
Write-Host "Setting up pipeline at: $root" -ForegroundColor Green
Write-Host "Current workspace: $currentDir" -ForegroundColor Gray

# Check if we're running from the correct location
$requiredFiles = @("scripts\00_bootstrap.ps1", "tools\analyze_audio.py", "config\characters.yml")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "❌ Missing required files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "   • $file" -ForegroundColor Red
    }
    Write-Host "Please run this script from the UI-TARS-desktop directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ All required files found" -ForegroundColor Green

# Create the target directory structure
Write-Host "`n1. Creating target directory structure..." -ForegroundColor Yellow

if ($DryRun) {
    Write-Host "   [DRY RUN] Would create: $root" -ForegroundColor Gray
} else {
    # Create root directory
    if (-not (Test-Path $root)) {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        Write-Host "   ✓ Created root directory: $root" -ForegroundColor Green
    } else {
        Write-Host "   ✓ Root directory already exists: $root" -ForegroundColor Green
    }
}

# Copy all files to target location
Write-Host "`n2. Copying pipeline files..." -ForegroundColor Yellow

$copyOperations = @(
    @{Source="scripts"; Target="$root\scripts"; Description="PowerShell scripts"},
    @{Source="tools"; Target="$root\tools"; Description="Python tools"},
    @{Source="config"; Target="$root\config"; Description="Configuration files"},
    @{Source="docs"; Target="$root\docs"; Description="Documentation"}
)

foreach ($op in $copyOperations) {
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would copy: $($op.Source) → $($op.Target)" -ForegroundColor Gray
    } else {
        if (Test-Path $op.Source) {
            Copy-Item -Path $op.Source -Destination $op.Target -Recurse -Force
            Write-Host "   ✓ Copied $($op.Description)" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ Source not found: $($op.Source)" -ForegroundColor Yellow
        }
    }
}

# Copy individual files
$individualFiles = @(
    @{Source="README.md"; Target="$root\README.md"; Description="Main README"},
    @{Source="setup_creative_pipeline.ps1"; Target="$root\setup_creative_pipeline.ps1"; Description="Setup script"}
)

foreach ($file in $individualFiles) {
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would copy: $($file.Source) → $($file.Target)" -ForegroundColor Gray
    } else {
        if (Test-Path $file.Source) {
            Copy-Item -Path $file.Source -Destination $file.Target -Force
            Write-Host "   ✓ Copied $($file.Description)" -ForegroundColor Green
        }
    }
}

# Run the bootstrap script
Write-Host "`n3. Running bootstrap script..." -ForegroundColor Yellow

if ($DryRun) {
    Write-Host "   [DRY RUN] Would run: $root\scripts\00_bootstrap.ps1" -ForegroundColor Gray
} else {
    try {
        & "$root\scripts\00_bootstrap.ps1"
        Write-Host "   ✓ Bootstrap completed" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Bootstrap failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Run validation if requested
if (-not $SkipValidation) {
    Write-Host "`n4. Validating setup..." -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run validation" -ForegroundColor Gray
    } else {
        try {
            & "$root\scripts\validate_setup.ps1"
            Write-Host "   ✓ Validation completed" -ForegroundColor Green
        } catch {
            Write-Host "   ⚠ Validation had issues: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Final instructions
Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "This was a dry run. Use without -DryRun to execute." -ForegroundColor Yellow
} else {
    Write-Host "✅ Creative pipeline setup complete!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "1. Navigate to: $root" -ForegroundColor White
    Write-Host "2. Run: .\scripts\run_pipeline.ps1" -ForegroundColor White
    Write-Host "3. Install Reallusion software (CC5, iClone 8, Headshot 2)" -ForegroundColor White
    Write-Host "4. Add your audio files to projects\Kelly\Audio\" -ForegroundColor White
    Write-Host "5. Follow the setup guide: docs\SETUP_GUIDE.md" -ForegroundColor White
    
    Write-Host "`nQuick start commands:" -ForegroundColor Cyan
    Write-Host "cd `"$root`"" -ForegroundColor Gray
    Write-Host ".\scripts\run_pipeline.ps1" -ForegroundColor Gray
    Write-Host ".\scripts\validate_setup.ps1" -ForegroundColor Gray
}

Write-Host "`nSetup script completed." -ForegroundColor Cyan
