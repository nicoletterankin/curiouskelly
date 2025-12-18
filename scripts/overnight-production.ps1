# OVERNIGHT VIDEO PRODUCTION SCRIPT
# Generates Kelly videos for Days 352-365

param(
    [int]$StartDay = 352,
    [int]$EndDay = 365,
    [switch]$DryRun,
    [switch]$RedubOnly,
    [switch]$FreshOnly,
    [int]$ReferenceDay = 351
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  OVERNIGHT VIDEO PRODUCTION" -ForegroundColor Magenta
Write-Host "  Sync Labs Re-Dub + Fresh Generation Pipeline" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

$totalDays = $EndDay - $StartDay + 1
$redubVideos = $totalDays * 9
$freshVideos = $totalDays * 3
$totalVideos = $redubVideos + $freshVideos

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "   Days: $StartDay - $EndDay ($totalDays days)"
Write-Host "   Reference Day: $ReferenceDay"
Write-Host "   Re-dub videos: $redubVideos (9 archetypes x $totalDays days)"
Write-Host "   Fresh videos: $freshVideos (3 archetypes x $totalDays days)"
Write-Host "   Total videos: $totalVideos"
Write-Host "   Estimated time: ~$([math]::Round($totalVideos * 2 / 60, 1)) hours"
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN MODE - No videos will be generated" -ForegroundColor Yellow
    Write-Host ""
}

if (-not (Test-Path "scripts\sync-labs-video-redub.ts")) {
    Write-Host "Error: Please run from the UI-TARS-desktop root directory" -ForegroundColor Red
    exit 1
}

$startTime = Get-Date
Write-Host "Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
Write-Host ""

$redubSuccess = @()
$redubFailed = @()
$freshSuccess = @()
$freshFailed = @()

# PHASE 1: Re-dub Pipeline
if (-not $FreshOnly) {
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "PHASE 1: Re-Dub Pipeline (9 archetypes per day)" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""

    for ($day = $StartDay; $day -le $EndDay; $day++) {
        $dayStart = Get-Date
        Write-Host "  Day $day " -NoNewline
        
        if ($DryRun) {
            Write-Host "[DRY RUN] " -ForegroundColor DarkGray -NoNewline
            Start-Sleep -Milliseconds 100
            Write-Host "Would generate 9 archetypes" -ForegroundColor Green
            $redubSuccess += $day
        }
        else {
            try {
                $output = npx tsx scripts/sync-labs-video-redub.ts --day $day --reference-day $ReferenceDay 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $duration = ((Get-Date) - $dayStart).TotalMinutes
                    Write-Host "Complete ($([math]::Round($duration, 1)) min)" -ForegroundColor Green
                    $redubSuccess += $day
                }
                else {
                    Write-Host "Failed" -ForegroundColor Red
                    $redubFailed += $day
                }
            }
            catch {
                Write-Host "Error: $_" -ForegroundColor Red
                $redubFailed += $day
            }
        }
    }
    Write-Host ""
}

# PHASE 2: Fresh Generation
if (-not $RedubOnly) {
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "PHASE 2: Fresh Generation (explorer, mystic, provider)" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""

    for ($day = $StartDay; $day -le $EndDay; $day++) {
        $dayStart = Get-Date
        Write-Host "  Day $day " -NoNewline
        
        if ($DryRun) {
            Write-Host "[DRY RUN] " -ForegroundColor DarkGray -NoNewline
            Start-Sleep -Milliseconds 100
            Write-Host "Would generate 3 archetypes" -ForegroundColor Green
            $freshSuccess += $day
        }
        else {
            try {
                $output = npx tsx scripts/sync-labs-batch-generate.ts --day $day --only explorer,mystic,provider 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $duration = ((Get-Date) - $dayStart).TotalMinutes
                    Write-Host "Complete ($([math]::Round($duration, 1)) min)" -ForegroundColor Green
                    $freshSuccess += $day
                }
                else {
                    Write-Host "Failed" -ForegroundColor Red
                    $freshFailed += $day
                }
            }
            catch {
                Write-Host "Error: $_" -ForegroundColor Red
                $freshFailed += $day
            }
        }
    }
    Write-Host ""
}

# SUMMARY
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "PRODUCTION SUMMARY" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
Write-Host ""

Write-Host "Re-Dub Pipeline:" -ForegroundColor White
Write-Host "   Success: $($redubSuccess.Count) days" -ForegroundColor Green
if ($redubFailed.Count -gt 0) {
    Write-Host "   Failed: $($redubFailed.Count) days ($($redubFailed -join ', '))" -ForegroundColor Red
}

Write-Host ""
Write-Host "Fresh Generation:" -ForegroundColor White
Write-Host "   Success: $($freshSuccess.Count) days" -ForegroundColor Green
if ($freshFailed.Count -gt 0) {
    Write-Host "   Failed: $($freshFailed.Count) days ($($freshFailed -join ', '))" -ForegroundColor Red
}

Write-Host ""
$totalSuccess = $redubSuccess.Count * 9 + $freshSuccess.Count * 3
$totalFailed = $redubFailed.Count * 9 + $freshFailed.Count * 3
Write-Host "Total Videos: $totalSuccess successful, $totalFailed failed" -ForegroundColor Cyan
Write-Host ""

Write-Host "Output Locations:" -ForegroundColor Yellow
Write-Host "   Re-dub manifests: generated-videos/sync-labs-redub/"
Write-Host "   Fresh manifests: generated-videos/sync-labs-production/"
Write-Host ""

if (-not $DryRun) {
    Write-Host "PRODUCTION COMPLETE!" -ForegroundColor Green
}
else {
    Write-Host "Dry run complete. Run without -DryRun to generate videos." -ForegroundColor Yellow
}
Write-Host ""
