# Software Verification Script
# Run this to verify your testing environment is ready

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Kelly Avatar Testing Environment Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check folder structure
Write-Host "Checking folder structure..." -ForegroundColor Yellow
$folders = @("original", "testing", "screenshots", "feedback")
foreach ($folder in $folders) {
    if (Test-Path $folder) {
        Write-Host "  [OK] $folder exists" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $folder MISSING" -ForegroundColor Red
        $allGood = $false
    }
}
Write-Host ""

# Check testing log
Write-Host "Checking testing log..." -ForegroundColor Yellow
if (Test-Path "TESTING_LOG.md") {
    Write-Host "  [OK] TESTING_LOG.md exists" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] TESTING_LOG.md MISSING" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check Unity scripts
Write-Host "Checking Unity Week 3 scripts..." -ForegroundColor Yellow
$scripts = @(
    "OptimizedBlendshapeDriver.cs",
    "GazeController.cs",
    "VisemeMapper.cs",
    "ExpressionCueDriver.cs",
    "AudioSyncCalibrator.cs",
    "FPSCounter.cs",
    "PerformanceMonitor.cs"
)

$scriptsPath = Join-Path $PSScriptRoot "..\..\..\digital-kelly\engines\kelly_unity_player\Assets\Kelly\Scripts"
$scriptsPath = Resolve-Path $scriptsPath -ErrorAction SilentlyContinue
if (-not $scriptsPath) {
    $scriptsPath = "c:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\Assets\Kelly\Scripts"
}
if (Test-Path $scriptsPath) {
    foreach ($script in $scripts) {
        $fullPath = Join-Path $scriptsPath $script
        if (Test-Path $fullPath) {
            Write-Host "  [OK] $script" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] $script MISSING" -ForegroundColor Red
            $allGood = $false
        }
    }
} else {
    Write-Host "  [FAIL] Scripts directory not found at: $scriptsPath" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check software (manual check needed)
Write-Host "Software checks (manual verification needed):" -ForegroundColor Yellow
Write-Host "  [ ] Character Creator 5 installed and licensed" -ForegroundColor White
Write-Host "  [ ] iClone 8 installed and licensed" -ForegroundColor White
Write-Host "  [ ] Unity 2022.3 LTS installed" -ForegroundColor White
Write-Host "  [ ] Unity project opens without errors" -ForegroundColor White
Write-Host "  [ ] Face Puppet accessible in iClone" -ForegroundColor White
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "[OK] Folder structure and files ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Verify software manually (CC5, iClone, Unity)" -ForegroundColor White
    Write-Host "2. Open Unity project and verify scripts compile" -ForegroundColor White
    Write-Host "3. Test FPS counter (F3) and Performance Monitor (F4)" -ForegroundColor White
    Write-Host "4. Fill out VERIFICATION_CHECKLIST.md" -ForegroundColor White
} else {
    Write-Host "[FAIL] Some items need attention!" -ForegroundColor Red
    Write-Host "Check the errors above and fix them." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

