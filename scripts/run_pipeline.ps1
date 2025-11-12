# UI-Tars CC5/iClone Pipeline Master Execution Script
# This script orchestrates the complete creative pipeline setup

param(
    [switch]$SkipBootstrap,
    [switch]$SkipDeps,
    [switch]$SkipReallusion,
    [switch]$SkipAudio,
    [switch]$SkipAnalytics,
    [switch]$SkipCharacters,
    [switch]$SkipTasks,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = "D:\iLearnStudio"

Write-Host "=== UI-Tars CC5/iClone Creative Pipeline ===" -ForegroundColor Cyan
Write-Host "Starting pipeline execution..." -ForegroundColor Green

# Step 1: Bootstrap workspace
if (-not $SkipBootstrap) {
    Write-Host "`n1. Bootstrapping workspace..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run: scripts\00_bootstrap.ps1" -ForegroundColor Gray
    } else {
        & "$root\scripts\00_bootstrap.ps1"
        Write-Host "   ✓ Workspace bootstrapped" -ForegroundColor Green
    }
} else {
    Write-Host "`n1. Skipping bootstrap (--SkipBootstrap)" -ForegroundColor Gray
}

# Step 2: Install dependencies
if (-not $SkipDeps) {
    Write-Host "`n2. Installing system dependencies..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run: scripts\01_install_deps.ps1" -ForegroundColor Gray
    } else {
        & "$root\scripts\01_install_deps.ps1"
        Write-Host "   ✓ Dependencies installed" -ForegroundColor Green
    }
} else {
    Write-Host "`n2. Skipping dependencies (--SkipDeps)" -ForegroundColor Gray
}

# Step 3: Detect Reallusion software
if (-not $SkipReallusion) {
    Write-Host "`n3. Detecting Reallusion installations..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run: scripts\02_detect_reallusion.ps1" -ForegroundColor Gray
    } else {
        & "$root\scripts\02_detect_reallusion.ps1"
        Write-Host "   ✓ Reallusion detection complete" -ForegroundColor Green
    }
} else {
    Write-Host "`n3. Skipping Reallusion detection (--SkipReallusion)" -ForegroundColor Gray
}

# Step 4: Audio analysis (requires audio file)
if (-not $SkipAudio) {
    Write-Host "`n4. Analyzing audio..." -ForegroundColor Yellow
    $audioFile = "$root\projects\Kelly\Audio\kelly25_audio.wav"
    if (Test-Path $audioFile) {
        if ($DryRun) {
            Write-Host "   [DRY RUN] Would run: scripts\10_audio_analyze.ps1" -ForegroundColor Gray
        } else {
            & "$root\scripts\10_audio_analyze.ps1"
            Write-Host "   ✓ Audio analysis complete" -ForegroundColor Green
        }
    } else {
        Write-Host "   ⚠ Audio file not found: $audioFile" -ForegroundColor Yellow
        Write-Host "   Please place kelly25_audio.wav in the Audio folder" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n4. Skipping audio analysis (--SkipAudio)" -ForegroundColor Gray
}

# Step 5: Generate analytics (requires rendered video)
if (-not $SkipAnalytics) {
    Write-Host "`n5. Generating analytics..." -ForegroundColor Yellow
    $videoFile = "$root\renders\Kelly\kelly_test_talk_v1.mp4"
    if (Test-Path $videoFile) {
        if ($DryRun) {
            Write-Host "   [DRY RUN] Would run contact sheet and frame metrics scripts" -ForegroundColor Gray
        } else {
            & "$root\scripts\20_contact_sheet.ps1"
            & "$root\scripts\21_frame_metrics.ps1"
            Write-Host "   ✓ Analytics generated" -ForegroundColor Green
        }
    } else {
        Write-Host "   ⚠ Rendered video not found: $videoFile" -ForegroundColor Yellow
        Write-Host "   Please render Kelly test video first" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n5. Skipping analytics (--SkipAnalytics)" -ForegroundColor Gray
}

# Step 6: Scaffold characters
if (-not $SkipCharacters) {
    Write-Host "`n6. Scaffolding character projects..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run: scripts\30_new_character.ps1" -ForegroundColor Gray
    } else {
        & "$root\scripts\30_new_character.ps1"
        Write-Host "   ✓ Character projects scaffolded" -ForegroundColor Green
    }
} else {
    Write-Host "`n6. Skipping character scaffolding (--SkipCharacters)" -ForegroundColor Gray
}

# Step 7: Create VS Code tasks
if (-not $SkipTasks) {
    Write-Host "`n7. Creating VS Code/Cursor tasks..." -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "   [DRY RUN] Would run: scripts\40_write_tasksjson.ps1" -ForegroundColor Gray
    } else {
        & "$root\scripts\40_write_tasksjson.ps1"
        Write-Host "   ✓ VS Code tasks created" -ForegroundColor Green
    }
} else {
    Write-Host "`n7. Skipping VS Code tasks (--SkipTasks)" -ForegroundColor Gray
}

# Final status
Write-Host "`n=== Pipeline Execution Complete ===" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "This was a dry run. Use without -DryRun to execute." -ForegroundColor Yellow
} else {
    Write-Host "Check metrics\install_status.json for detailed status" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Install Reallusion software (CC5, iClone 8, Headshot 2)" -ForegroundColor White
    Write-Host "  2. Place kelly25_audio.wav in projects\Kelly\Audio\" -ForegroundColor White
    Write-Host "  3. Build Kelly character in CC5/iClone" -ForegroundColor White
    Write-Host "  4. Render test video to renders\Kelly\" -ForegroundColor White
    Write-Host "  5. Run analytics scripts" -ForegroundColor White
}
