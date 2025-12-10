# ============================================================================
# DAY 1 COMPLETE GENERATION SCRIPT
# ============================================================================
# Runs all 10 archetypes sequentially to generate complete, consistent assets
# Expected runtime: ~5 hours
# Output: 200 videos in Supabase (10 archetypes x 20 videos each)
# ============================================================================

$ErrorActionPreference = "Continue"
$StartTime = Get-Date

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "DAY 1 COMPLETE GENERATION - Starting $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# All 10 archetypes in order
$archetypes = @(
    "The Explorer",
    "The Rebel", 
    "The Scientist",
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

# ============================================================================
# SUMMARY
# ============================================================================
$TotalDuration = (Get-Date) - $StartTime

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "DAY 1 GENERATION COMPLETE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total runtime: $($TotalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "Completed: $($completed.Count)/10 archetypes" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed.Count) - $($failed -join ', ')" -ForegroundColor Red
}
Write-Host ""
Write-Host "Next: npx tsx scripts/lesson-factory/verify-day1-assets.ts" -ForegroundColor Yellow
Write-Host ""
