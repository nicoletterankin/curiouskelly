# Parallel Content Generation Script
# Runs multiple generation processes for different day ranges

Write-Host "Starting Parallel Content Generation..." -ForegroundColor Cyan

# Define day ranges for parallel processing
$ranges = @(
    "1-50",
    "51-100",
    "101-150",
    "151-200",
    "201-250",
    "251-300",
    "301-365"
)

# Start each range in a new process
$jobs = @()
foreach ($range in $ranges) {
    Write-Host "Starting generation for days $range..." -ForegroundColor Yellow
    
    $job = Start-Job -ScriptBlock {
        param($range, $projectPath)
        Set-Location $projectPath
        python scripts/generate_all_content.py --days $range --skip-existing --yes 2>&1
    } -ArgumentList $range, (Get-Location).Path
    
    $jobs += $job
    
    # Small delay to avoid overwhelming the API
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "Started $($jobs.Count) parallel generation jobs" -ForegroundColor Green
Write-Host "Job IDs: $($jobs.Id -join ', ')"
Write-Host ""
Write-Host "Monitor progress with: Get-Job | Receive-Job -Keep"
Write-Host "Check completion with: Get-Job"





