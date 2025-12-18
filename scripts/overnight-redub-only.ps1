# OVERNIGHT RE-DUB PRODUCTION
# Days 352-365 | 9 archetypes per day | ~4 hours
# Uses HeyGen Day 351 videos as motion base for consistent Kelly

param(
    [int]$StartDay = 352,
    [int]$EndDay = 365,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  OVERNIGHT RE-DUB PRODUCTION" -ForegroundColor Magenta
Write-Host "  High-Quality Kelly Videos (HeyGen Motion Base)" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

$totalDays = $EndDay - $StartDay + 1
$videosPerDay = 9
$totalVideos = $totalDays * $videosPerDay
$estHours = [math]::Round($totalVideos * 2 / 60, 1)

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "   Days: $StartDay - $EndDay ($totalDays days)"
Write-Host "   Videos per day: $videosPerDay (high-quality archetypes only)"
Write-Host "   Total videos: $totalVideos"
Write-Host "   Estimated time: ~$estHours hours"
Write-Host ""
Write-Host "Archetypes:" -ForegroundColor Cyan
Write-Host "   scientist, rebel, architect, diplomat, empath," 
Write-Host "   macgyver, storyteller, strategist, survivor"
Write-Host ""
Write-Host "Skipping (no HeyGen base):" -ForegroundColor Yellow
Write-Host "   explorer, mystic, provider"
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN MODE]" -ForegroundColor Yellow
    Write-Host ""
}

$startTime = Get-Date
Write-Host "Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "GENERATING..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

$successDays = @()
$failedDays = @()

for ($day = $StartDay; $day -le $EndDay; $day++) {
    $dayStart = Get-Date
    $progress = [math]::Round(($day - $StartDay) / $totalDays * 100)
    Write-Host "[$progress%] Day $day " -NoNewline
    
    if ($DryRun) {
        Start-Sleep -Milliseconds 200
        Write-Host "- would generate 9 videos" -ForegroundColor DarkGray
        $successDays += $day
    }
    else {
        try {
            $output = npx tsx scripts/sync-labs-video-redub.ts --day $day --reference-day 351 2>&1
            $exitCode = $LASTEXITCODE
            
            if ($exitCode -eq 0) {
                $duration = [math]::Round(((Get-Date) - $dayStart).TotalMinutes, 1)
                Write-Host "- DONE ($duration min)" -ForegroundColor Green
                $successDays += $day
            }
            else {
                Write-Host "- FAILED" -ForegroundColor Red
                $failedDays += $day
            }
        }
        catch {
            Write-Host "- ERROR: $_" -ForegroundColor Red
            $failedDays += $day
        }
    }
}

# Summary
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "COMPLETE" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
Write-Host "Success: $($successDays.Count) days ($($successDays.Count * 9) videos)" -ForegroundColor Green

if ($failedDays.Count -gt 0) {
    Write-Host "Failed: $($failedDays.Count) days - $($failedDays -join ', ')" -ForegroundColor Red
    Write-Host ""
    Write-Host "To retry failed days:" -ForegroundColor Yellow
    foreach ($fd in $failedDays) {
        Write-Host "  npx tsx scripts/sync-labs-video-redub.ts --day $fd --reference-day 351"
    }
}

Write-Host ""
Write-Host "Output: generated-videos/sync-labs-redub/" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "This was a dry run. Execute without -DryRun to generate." -ForegroundColor Yellow
}
