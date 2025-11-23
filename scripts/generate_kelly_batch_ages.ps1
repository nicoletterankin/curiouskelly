# Kelly Age-Progressive Batch Generator
# Generates all 72 images (6 ages × 4 poses × 3 formats) for lesson age slider

param(
    [string]$OutputDir = "projects\Kelly\assets\age_progressive",
    [switch]$SkipExisting = $false
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# Configuration
$PresetDir = "presets\age_progressive"
$PythonScript = "scripts\generate_kelly_multiformat.py"
$LogFile = "$OutputDir\generation_log.txt"

# Statistics
$script:TotalGenerated = 0
$script:TotalFailed = 0
$script:TotalSkipped = 0
$script:StartTime = Get-Date

# Age groups and poses
$AgeGroups = @("3", "9", "15", "27", "48", "82")
$Poses = @("pose1", "pose2", "pose3", "pose4")
$Formats = @("16x9", "1x1", "3x4")

# Banner
Write-Host ""
Write-Host "Kelly Age-Progressive Image Generation" -ForegroundColor Cyan
Write-Host "6 Ages x 4 Poses x 3 Formats = 72 Images" -ForegroundColor Cyan
Write-Host ""

# Setup
Write-Host "Setting up..." -ForegroundColor Yellow
Write-Host "  Output directory: $OutputDir"
Write-Host "  Preset directory: $PresetDir"
Write-Host "  Log file: $LogFile"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\renders" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\manifests" | Out-Null

# Initialize log
"Kelly Age-Progressive Generation Log" | Out-File -FilePath $LogFile -Encoding UTF8
"Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $LogFile -Append -Encoding UTF8
"" | Out-File -FilePath $LogFile -Append -Encoding UTF8

$TotalPresets = $AgeGroups.Count * $Poses.Count
$CurrentPreset = 0

Write-Host "Generating $TotalPresets presets x 3 formats = $($TotalPresets * 3) total images" -ForegroundColor Green
Write-Host ""

# Main generation loop
foreach ($age in $AgeGroups) {
    Write-Host "-------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  AGE GROUP: Kelly at age $age" -ForegroundColor Cyan
    Write-Host "-------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""
    
    foreach ($pose in $Poses) {
        $CurrentPreset++
        $PresetFile = "kelly_age${age}_${pose}_v001.yaml"
        $PresetPath = Join-Path $PresetDir $PresetFile
        
        if (-not (Test-Path $PresetPath)) {
            Write-Host "  WARNING: Preset not found: $PresetFile" -ForegroundColor Yellow
            "SKIPPED: $PresetFile (not found)" | Out-File -FilePath $LogFile -Append -Encoding UTF8
            $script:TotalSkipped++
            continue
        }
        
        Write-Host "  [$CurrentPreset/$TotalPresets] Generating: $PresetFile" -ForegroundColor White
        
        # Check if outputs already exist
        $BaseOutputName = "kelly_age${age}_${pose}"
        $AllExist = $true
        
        if ($SkipExisting) {
            foreach ($format in $Formats) {
                $OutputFile = "$OutputDir\renders\${BaseOutputName}_${format}.png"
                if (-not (Test-Path $OutputFile)) {
                    $AllExist = $false
                    break
                }
            }
            
            if ($AllExist) {
                Write-Host "    SKIP: All formats exist" -ForegroundColor DarkGray
                "SKIPPED: $PresetFile (already exists)" | Out-File -FilePath $LogFile -Append -Encoding UTF8
                $script:TotalSkipped += 3
                continue
            }
        }
        
        # Generate using Python script
        try {
            $Output = & python $PythonScript $PresetPath --outdir $OutputDir --formats "16:9" "1:1" "3:4" 2>&1
            $ExitCode = $LASTEXITCODE
            
            if ($ExitCode -eq 0) {
                Write-Host "    SUCCESS: Generated 3 formats" -ForegroundColor Green
                "SUCCESS: $PresetFile" | Out-File -FilePath $LogFile -Append -Encoding UTF8
                $script:TotalGenerated += 3
            } else {
                Write-Host "    ERROR: Failed with exit code $ExitCode" -ForegroundColor Red
                Write-Host "    Error: $Output" -ForegroundColor Red
                "FAILED: $PresetFile (exit code $ExitCode)" | Out-File -FilePath $LogFile -Append -Encoding UTF8
                "  Error: $Output" | Out-File -FilePath $LogFile -Append -Encoding UTF8
                $script:TotalFailed += 3
            }
        }
        catch {
            Write-Host "    ERROR: Exception: $($_.Exception.Message)" -ForegroundColor Red
            "FAILED: $PresetFile (exception: $($_.Exception.Message))" | Out-File -FilePath $LogFile -Append -Encoding UTF8
            $script:TotalFailed += 3
        }
        
        Write-Host ""
    }
    
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $script:StartTime

Write-Host "-------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  GENERATION COMPLETE" -ForegroundColor Cyan
Write-Host "-------------------------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Statistics:" -ForegroundColor Yellow
Write-Host "   Generated: $($script:TotalGenerated) images" -ForegroundColor Green
Write-Host "   Failed:    $($script:TotalFailed) images" -ForegroundColor Red
Write-Host "   Skipped:   $($script:TotalSkipped) images" -ForegroundColor Gray
Write-Host "   Duration:  $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host ""

# Write summary to log
"" | Out-File -FilePath $LogFile -Append -Encoding UTF8
"Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $LogFile -Append -Encoding UTF8
"Duration: $($Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $LogFile -Append -Encoding UTF8
"Generated: $($script:TotalGenerated) | Failed: $($script:TotalFailed) | Skipped: $($script:TotalSkipped)" | Out-File -FilePath $LogFile -Append -Encoding UTF8

# Next steps
Write-Host "Next Steps:" -ForegroundColor Yellow

if ($script:TotalFailed -gt 0) {
    Write-Host "   1. Review log file: $LogFile" -ForegroundColor White
    Write-Host "   2. Check failed images and retry if needed" -ForegroundColor White
} else {
    Write-Host "   1. Review generated images in: $OutputDir\renders\" -ForegroundColor White
}

Write-Host "   2. Open review gallery: $OutputDir\review.html" -ForegroundColor White
Write-Host "   3. Validate age consistency: python scripts\validate_age_consistency.py" -ForegroundColor White
Write-Host ""



