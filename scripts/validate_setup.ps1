# UI-Tars Pipeline Validation Script
# Validates the complete setup and reports status

$ErrorActionPreference = "Continue"
$root = "C:\Users\user\UI-TARS-desktop"
$issues = @()
$warnings = @()

Write-Host "=== UI-Tars Pipeline Validation ===" -ForegroundColor Cyan
Write-Host "Validating setup at: $root" -ForegroundColor Green

# Check if root directory exists
if (-not (Test-Path $root)) {
    $issues += "Root directory does not exist: $root"
    Write-Host "❌ Root directory missing" -ForegroundColor Red
} else {
    Write-Host "✓ Root directory exists" -ForegroundColor Green
}

# Check directory structure
$requiredDirs = @(
    "config", "installers", "scripts", "tools", "docs", "metrics",
    "analytics\Kelly", "projects\_Shared", "projects\Kelly",
    "renders\Kelly"
)

foreach ($dir in $requiredDirs) {
    $fullPath = Join-Path $root $dir
    if (Test-Path $fullPath) {
        Write-Host "✓ Directory exists: $dir" -ForegroundColor Green
    } else {
        $issues += "Missing directory: $dir"
        Write-Host "❌ Missing directory: $dir" -ForegroundColor Red
    }
}

# Check scripts
$requiredScripts = @(
    "00_bootstrap.ps1", "01_install_deps.ps1", "02_detect_reallusion.ps1",
    "10_audio_analyze.ps1", "20_contact_sheet.ps1", "21_frame_metrics.ps1",
    "30_new_character.ps1", "40_write_tasksjson.ps1", "run_pipeline.ps1"
)

foreach ($script in $requiredScripts) {
    $scriptPath = Join-Path $root "scripts\$script"
    if (Test-Path $scriptPath) {
        Write-Host "✓ Script exists: $script" -ForegroundColor Green
    } else {
        $issues += "Missing script: $script"
        Write-Host "❌ Missing script: $script" -ForegroundColor Red
    }
}

# Check Python tools
$pythonTools = @("analyze_audio.py", "frame_metrics.py")
foreach ($tool in $pythonTools) {
    $toolPath = Join-Path $root "tools\$tool"
    if (Test-Path $toolPath) {
        Write-Host "✓ Python tool exists: $tool" -ForegroundColor Green
    } else {
        $issues += "Missing Python tool: $tool"
        Write-Host "❌ Missing Python tool: $tool" -ForegroundColor Red
    }
}

# Check configuration files
$configFiles = @("characters.yml", "VERSIONS.md")
foreach ($file in $configFiles) {
    $filePath = Join-Path $root "config\$file"
    if ($file -eq "VERSIONS.md") { $filePath = Join-Path $root "docs\$file" }
    
    if (Test-Path $filePath) {
        Write-Host "✓ Config file exists: $file" -ForegroundColor Green
    } else {
        $warnings += "Missing config file: $file"
        Write-Host "⚠ Missing config file: $file" -ForegroundColor Yellow
    }
}

# Check system dependencies
Write-Host "`nChecking system dependencies..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = py -3.11 --version 2>$null
    if ($pythonVersion) {
        Write-Host "✓ Python 3.11: $pythonVersion" -ForegroundColor Green
    } else {
        $issues += "Python 3.11 not found"
        Write-Host "❌ Python 3.11 not found" -ForegroundColor Red
    }
} catch {
    $issues += "Python 3.11 not accessible"
    Write-Host "❌ Python 3.11 not accessible" -ForegroundColor Red
}

# Check FFmpeg
try {
    $ffmpegVersion = ffmpeg -version 2>$null | Select-Object -First 1
    if ($ffmpegVersion) {
        Write-Host "✓ FFmpeg: $ffmpegVersion" -ForegroundColor Green
    } else {
        $issues += "FFmpeg not found"
        Write-Host "❌ FFmpeg not found" -ForegroundColor Red
    }
} catch {
    $issues += "FFmpeg not accessible"
    Write-Host "❌ FFmpeg not accessible" -ForegroundColor Red
}

# Check Git
try {
    $gitVersion = git --version 2>$null
    if ($gitVersion) {
        Write-Host "✓ Git: $gitVersion" -ForegroundColor Green
    } else {
        $issues += "Git not found"
        Write-Host "❌ Git not found" -ForegroundColor Red
    }
} catch {
    $issues += "Git not accessible"
    Write-Host "❌ Git not accessible" -ForegroundColor Red
}

# Check Git LFS
try {
    $gitLfsVersion = git lfs version 2>$null | Select-Object -First 1
    if ($gitLfsVersion) {
        Write-Host "✓ Git LFS: $gitLfsVersion" -ForegroundColor Green
    } else {
        $issues += "Git LFS not found"
        Write-Host "❌ Git LFS not found" -ForegroundColor Red
    }
} catch {
    $issues += "Git LFS not accessible"
    Write-Host "❌ Git LFS not accessible" -ForegroundColor Red
}

# Check Reallusion software
Write-Host "`nChecking Reallusion software..." -ForegroundColor Yellow

$reallusionPaths = @{
    "Character Creator 5" = "C:\Program Files\Reallusion\Character Creator 5\Bin64\CharacterCreator.exe"
    "iClone 8" = "C:\Program Files\Reallusion\iClone 8\Bin64\iClone.exe"
}

$reallusionFound = 0
foreach ($app in $reallusionPaths.Keys) {
    if (Test-Path $reallusionPaths[$app]) {
        Write-Host "✓ $app installed" -ForegroundColor Green
        $reallusionFound++
    } else {
        $warnings += "$app not found at expected location"
        Write-Host "⚠ $app not found" -ForegroundColor Yellow
    }
}

# Check Python libraries
Write-Host "`nChecking Python libraries..." -ForegroundColor Yellow

$pythonLibs = @("numpy", "pandas", "matplotlib", "librosa", "soundfile", "opencv-python")
foreach ($lib in $pythonLibs) {
    try {
        $result = py -3.11 -c "import $lib; print('OK')" 2>$null
        if ($result -eq "OK") {
            Write-Host "✓ $lib" -ForegroundColor Green
        } else {
            $issues += "Python library not found: $lib"
            Write-Host "❌ $lib" -ForegroundColor Red
        }
    } catch {
        $issues += "Python library not accessible: $lib"
        Write-Host "❌ $lib" -ForegroundColor Red
    }
}

# Check content files
Write-Host "`nChecking content files..." -ForegroundColor Yellow

$audioFile = Join-Path $root "projects\Kelly\Audio\kelly25_audio.wav"
if (Test-Path $audioFile) {
    Write-Host "✓ Kelly audio file exists" -ForegroundColor Green
} else {
    $warnings += "Kelly audio file not found: $audioFile"
    Write-Host "⚠ Kelly audio file not found" -ForegroundColor Yellow
}

$videoFile = Join-Path $root "renders\Kelly\kelly_test_talk_v1.mp4"
if (Test-Path $videoFile) {
    Write-Host "✓ Kelly test video exists" -ForegroundColor Green
} else {
    $warnings += "Kelly test video not found: $videoFile"
    Write-Host "⚠ Kelly test video not found" -ForegroundColor Yellow
}

# Summary
Write-Host "`n=== Validation Summary ===" -ForegroundColor Cyan

if ($issues.Count -eq 0) {
    Write-Host "✅ No critical issues found!" -ForegroundColor Green
} else {
    Write-Host "❌ $($issues.Count) critical issues found:" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "   • $issue" -ForegroundColor Red
    }
}

if ($warnings.Count -gt 0) {
    Write-Host "⚠ $($warnings.Count) warnings:" -ForegroundColor Yellow
    foreach ($warning in $warnings) {
        Write-Host "   • $warning" -ForegroundColor Yellow
    }
}

# Recommendations
Write-Host "`n=== Recommendations ===" -ForegroundColor Cyan

if ($issues.Count -gt 0) {
    Write-Host "1. Fix critical issues before proceeding" -ForegroundColor Yellow
    Write-Host "2. Run the bootstrap script: .\scripts\00_bootstrap.ps1" -ForegroundColor White
    Write-Host "3. Install dependencies: .\scripts\01_install_deps.ps1" -ForegroundColor White
} elseif ($warnings.Count -gt 0) {
    Write-Host "1. Address warnings for optimal performance" -ForegroundColor Yellow
    Write-Host "2. Install Reallusion software if not already done" -ForegroundColor White
    Write-Host "3. Add your audio and video files" -ForegroundColor White
} else {
    Write-Host "1. Setup looks good! Ready to run the pipeline" -ForegroundColor Green
    Write-Host "2. Run: .\scripts\run_pipeline.ps1" -ForegroundColor White
}

Write-Host "`nValidation complete." -ForegroundColor Cyan