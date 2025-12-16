# ============================================
# PRE-FLIGHT CHECK: Kelly Chair Integration
# ============================================
# Run this BEFORE starting the integration
# to make sure everything is in place
# ============================================

$ErrorActionPreference = "SilentlyContinue"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   PRE-FLIGHT CHECK" -ForegroundColor Cyan
Write-Host "   Kelly Chair Integration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true
$warnings = @()

# Check 1: CC5 Project File
Write-Host "1. Checking CC5 project file..." -ForegroundColor White
$cc5File = "C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\CC5 Cloth update 8.1.ccProject"
if (Test-Path $cc5File) {
    $size = [math]::Round((Get-Item $cc5File).Length / 1MB, 1)
    Write-Host "   [OK] Found: CC5 Cloth update 8.1.ccProject ($size MB)" -ForegroundColor Green
} else {
    Write-Host "   [FAIL] NOT FOUND: CC5 Cloth update 8.1.ccProject" -ForegroundColor Red
    $allGood = $false
}

# Check 2: Character Creator 5
Write-Host ""
Write-Host "2. Checking for Character Creator 5..." -ForegroundColor White
$cc5Paths = @(
    "C:\Program Files\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe",
    "C:\Program Files (x86)\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe",
    "$env:LOCALAPPDATA\Programs\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe"
)
$cc5Found = $false
foreach ($path in $cc5Paths) {
    if (Test-Path $path) {
        Write-Host "   [OK] Found Character Creator 5" -ForegroundColor Green
        $cc5Found = $true
        break
    }
}
if (-not $cc5Found) {
    Write-Host "   [WARN] Character Creator 5 not found in standard locations" -ForegroundColor Yellow
    $warnings += "CC5 path not detected (may still work if installed elsewhere)"
}

# Check 3: iClone 8
Write-Host ""
Write-Host "3. Checking for iClone 8..." -ForegroundColor White
$iclonePaths = @(
    "C:\Program Files\Reallusion\iClone 8\Bin64\iClone.exe",
    "C:\Program Files (x86)\Reallusion\iClone 8\Bin64\iClone.exe"
)
$icloneFound = $false
foreach ($path in $iclonePaths) {
    if (Test-Path $path) {
        Write-Host "   [OK] Found iClone 8" -ForegroundColor Green
        $icloneFound = $true
        break
    }
}
if (-not $icloneFound) {
    Write-Host "   [WARN] iClone 8 not found in standard locations" -ForegroundColor Yellow
    $warnings += "iClone 8 path not detected (may still work if installed elsewhere)"
}

# Check 4: Unity Project
Write-Host ""
Write-Host "4. Checking Unity project..." -ForegroundColor White
$unityProject = "C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly"
if (Test-Path "$unityProject\Assets") {
    Write-Host "   [OK] Found Unity project: onlykelly" -ForegroundColor Green
    
    # Check for CCIC tools
    if (Test-Path "$unityProject\Assets\Reallusion") {
        Write-Host "   [OK] CCIC Unity Tools installed" -ForegroundColor Green
    } else {
        Write-Host "   [WARN] CCIC Unity Tools may not be installed" -ForegroundColor Yellow
        $warnings += "CCIC Unity Tools might need installation"
    }
} else {
    Write-Host "   [FAIL] Unity project not found" -ForegroundColor Red
    $allGood = $false
}

# Check 5: Unity Hub
Write-Host ""
Write-Host "5. Checking for Unity Hub..." -ForegroundColor White
$unityHubPaths = @(
    "$env:LOCALAPPDATA\UnityHub\Unity Hub.exe",
    "C:\Program Files\Unity Hub\Unity Hub.exe"
)
$unityHubFound = $false
foreach ($path in $unityHubPaths) {
    if (Test-Path $path) {
        Write-Host "   [OK] Found Unity Hub" -ForegroundColor Green
        $unityHubFound = $true
        break
    }
}
if (-not $unityHubFound) {
    Write-Host "   [WARN] Unity Hub not found in standard locations" -ForegroundColor Yellow
    $warnings += "Unity Hub path not detected"
}

# Check 6: Target directories
Write-Host ""
Write-Host "6. Checking target directories..." -ForegroundColor White
$targetDirs = @(
    "C:\Users\user\UI-TARS-desktop\public\unity",
    "C:\Users\user\UI-TARS-desktop\daily-lesson-marketing\public\unity"
)
foreach ($dir in $targetDirs) {
    if (Test-Path $dir) {
        Write-Host "   [OK] $dir" -ForegroundColor Green
    } else {
        Write-Host "   [WARN] $dir (will be created)" -ForegroundColor Yellow
    }
}

# Check 7: Scripts
Write-Host ""
Write-Host "7. Checking helper scripts..." -ForegroundColor White
$scripts = @(
    @{ Path = "C:\Users\user\UI-TARS-desktop\deploy-kelly-chair.ps1"; Name = "Deploy script" },
    @{ Path = "C:\Users\user\UI-TARS-desktop\public\unity\kelly-loader.js"; Name = "Progressive loader" }
)
foreach ($script in $scripts) {
    if (Test-Path $script.Path) {
        Write-Host "   [OK] $($script.Name)" -ForegroundColor Green
    } else {
        Write-Host "   [WARN] $($script.Name) not found" -ForegroundColor Yellow
    }
}

# Check 8: Disk Space
Write-Host ""
Write-Host "8. Checking disk space..." -ForegroundColor White
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -gt 5) {
    Write-Host "   [OK] $freeGB GB free (need about 2GB for build)" -ForegroundColor Green
} else {
    Write-Host "   [WARN] Only $freeGB GB free - may need more space" -ForegroundColor Yellow
    $warnings += "Low disk space"
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allGood -and $warnings.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You are ready to start the integration." -ForegroundColor White
    Write-Host ""
    Write-Host "Next step: Open this guide and follow it step by step:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   projects\Kelly\CC5\final-models\DO_THIS_NOW.md" -ForegroundColor Yellow
    Write-Host ""
} elseif ($allGood) {
    Write-Host "READY TO GO (with minor warnings)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Warnings:" -ForegroundColor Yellow
    foreach ($w in $warnings) {
        Write-Host "   - $w" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "These warnings are usually fine. Proceed with:" -ForegroundColor White
    Write-Host ""
    Write-Host "   projects\Kelly\CC5\final-models\DO_THIS_NOW.md" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "SOME CHECKS FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please resolve the issues marked with [FAIL] before proceeding." -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "Guide location:" -ForegroundColor Cyan
Write-Host "   C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\DO_THIS_NOW.md" -ForegroundColor White
Write-Host ""
