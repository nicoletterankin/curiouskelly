# ============================================================================
# DAY 1 MISSING ARCHETYPES - Regenerate the 7 that failed
# ============================================================================
# The previous run failed for archetypes 4-10 due to missing voiceSettings
# This script regenerates only those archetypes
# ============================================================================

$ErrorActionPreference = "Continue"
$StartTime = Get-Date

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "DAY 1 MISSING ARCHETYPES - Starting $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Only the 7 archetypes that failed (missing voice settings bug)
$archetypes = @(
    "The Architect",
    "The Diplomat",
    "The Empath",
    "The MacGyver",
    "The Mystic",
    "The Storyteller",
    "The Survivor"
)

$completed = @()
$failed = @()

foreach ($archetype in $archetypes) {
    $archetypeStart = Get-Date
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "Starting: $archetype ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
    
    try {
        npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --archetype $archetype
        
        $duration = (Get-Date) - $archetypeStart
        Write-Host "$archetype completed in $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        $completed += $archetype
    }
    catch {
        Write-Host "$archetype FAILED: $_" -ForegroundColor Red
        $failed += $archetype
    }
    
    Write-Host "Cooling down for 30 seconds..." -ForegroundColor Gray
    Start-Sleep -Seconds 30
}

$TotalDuration = (Get-Date) - $StartTime

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "DAY 1 MISSING ARCHETYPES COMPLETE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total runtime: $($TotalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Completed: $($completed.Count)/7 archetypes" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed.Count) - $($failed -join ', ')" -ForegroundColor Red
}
Write-Host ""
Write-Host "Next: npx tsx scripts/lesson-factory/verify-day1-assets.ts" -ForegroundColor Yellow
Write-Host ""










