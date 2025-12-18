# HEYGEN AUTO-POLL SCRIPT
# Continuously monitors queue and downloads completed videos

param(
    [int]$IntervalSeconds = 60,
    [int]$MaxRuns = 60
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  HEYGEN AUTO-POLL" -ForegroundColor Magenta
Write-Host "  Monitoring queue every $IntervalSeconds seconds" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

$startTime = Get-Date
$runs = 0

while ($runs -lt $MaxRuns) {
    $runs++
    $now = Get-Date -Format "HH:mm:ss"
    Write-Host "[$now] Run $runs/$MaxRuns" -ForegroundColor Cyan
    
    # Run monitor
    $output = npx tsx scripts/heygen-queue-monitor.ts 2>&1
    
    # Extract key info
    $day351Match = $output | Select-String "Summary: (\d+) completed"
    if ($day351Match) {
        $completed = $day351Match.Matches.Groups[1].Value
        Write-Host "  Day 351: $completed/12 completed" -ForegroundColor $(if ($completed -eq "12") { "Green" } else { "Yellow" })
        
        if ($completed -eq "12") {
            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Green
            Write-Host "  DAY 351 COMPLETE!" -ForegroundColor Green
            Write-Host "============================================================" -ForegroundColor Green
            break
        }
    }
    
    # Wait for next interval
    if ($runs -lt $MaxRuns) {
        Write-Host "  Waiting $IntervalSeconds seconds..." -ForegroundColor DarkGray
        Start-Sleep -Seconds $IntervalSeconds
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "POLLING COMPLETE" -ForegroundColor Magenta
Write-Host "  Duration: $($duration.Hours)h $($duration.Minutes)m" -ForegroundColor Cyan
Write-Host "  Runs: $runs" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
