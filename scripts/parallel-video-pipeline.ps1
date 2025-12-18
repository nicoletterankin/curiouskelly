# PARALLEL VIDEO PIPELINE
# Runs HeyGen monitoring + Sync Labs generation simultaneously

param(
    [int]$TargetDay = 352,
    [switch]$DryRun
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  PARALLEL VIDEO PIPELINE" -ForegroundColor Magenta
Write-Host "  HeyGen Monitor + Sync Labs Generation" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

# Start HeyGen monitor in background
Write-Host "[1/2] Starting HeyGen queue monitor..." -ForegroundColor Cyan
$heygenJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    while ($true) {
        npx tsx scripts/heygen-queue-monitor.ts 2>&1 | Out-Null
        Start-Sleep -Seconds 120
    }
}
Write-Host "  HeyGen monitor running (Job ID: $($heygenJob.Id))" -ForegroundColor Green

# Run Sync Labs for target day
Write-Host ""
Write-Host "[2/2] Running Sync Labs re-dub for Day $TargetDay..." -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "  [DRY RUN] Would run: npx tsx scripts/sync-labs-video-redub.ts --day $TargetDay --reference-day 351" -ForegroundColor Yellow
} else {
    npx tsx scripts/sync-labs-video-redub.ts --day $TargetDay --reference-day 351
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "PIPELINE STATUS" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "HeyGen Monitor: Running in background (Job $($heygenJob.Id))" -ForegroundColor Cyan
Write-Host "Sync Labs: Day $TargetDay processing" -ForegroundColor Cyan
Write-Host ""
Write-Host "To check HeyGen: npx tsx scripts/heygen-queue-monitor.ts" -ForegroundColor DarkGray
Write-Host "To stop monitor: Stop-Job $($heygenJob.Id)" -ForegroundColor DarkGray
Write-Host ""
