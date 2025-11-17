# Master Server Automation Script
# Starts all services and file watchers automatically
# Run this once - it handles everything

param(
    [switch]$SkipHtmlWatcher,
    [switch]$SkipInfrastructure
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")

Write-Host "=== Auto-Server Manager ===" -ForegroundColor Cyan
Write-Host "Starting all services automatically..." -ForegroundColor Green
Write-Host ""

# Start infrastructure if not skipped
if (-not $SkipInfrastructure) {
    Write-Host "[1/3] Starting infrastructure (Postgres, Redis, Meilisearch, ClickHouse)..." -ForegroundColor Yellow
    & "$scriptRoot\dev-server.ps1" -Target deps -KeepDependencies -ErrorAction SilentlyContinue | Out-Null
    Start-Sleep -Seconds 3
}

# Start HTML file watcher in background
if (-not $SkipHtmlWatcher) {
    Write-Host "[2/3] Starting HTML auto-opener..." -ForegroundColor Yellow
    $htmlWatcherJob = Start-Job -ScriptBlock {
        param($scriptPath, $repoPath)
        Set-Location $repoPath
        & $scriptPath -WatchPath $repoPath
    } -ArgumentList "$scriptRoot\auto-open-html.ps1", $repoRoot
    
    Write-Host "  ✓ HTML files will auto-open when created/modified" -ForegroundColor Green
}

# Start dev servers
Write-Host "[3/3] Starting dev servers (Gateway + Classroom)..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Servers are running! Press Ctrl+C to stop everything." -ForegroundColor Green
Write-Host ""

try {
    & "$scriptRoot\dev-server.ps1" -Target stack -SkipDependencies
} finally {
    # Cleanup
    if (-not $SkipHtmlWatcher -and $htmlWatcherJob) {
        Write-Host "Stopping HTML watcher..." -ForegroundColor Yellow
        Stop-Job $htmlWatcherJob -ErrorAction SilentlyContinue
        Remove-Job $htmlWatcherJob -ErrorAction SilentlyContinue
    }
    
    Write-Host "All services stopped." -ForegroundColor Cyan
}




