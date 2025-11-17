# Auto-Open HTML File Watcher
# Watches for HTML file changes and automatically opens them in the default browser
# Runs silently in the background - you'll never think about it

param(
    [string]$WatchPath = $PSScriptRoot + "\..",
    [int]$DebounceMs = 500
)

$ErrorActionPreference = "SilentlyContinue"
$watchedFiles = @{}
$debounceTimers = @{}

function Open-HtmlFile {
    param([string]$FilePath)
    
    $fullPath = Resolve-Path $FilePath -ErrorAction SilentlyContinue
    if (-not $fullPath) {
        return
    }
    
    $uri = "file:///$($fullPath.Path.Replace('\', '/'))"
    Start-Process $uri -ErrorAction SilentlyContinue | Out-Null
    Write-Host "[Auto-Open] Opened: $($fullPath.Name)" -ForegroundColor Green
}

function Register-FileWatcher {
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $WatchPath
    $watcher.Filter = "*.html"
    $watcher.IncludeSubdirectories = $true
    $watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor [System.IO.NotifyFilters]::LastWrite
    
    $action = {
        $path = $Event.SourceEventArgs.FullPath
        $changeType = $Event.SourceEventArgs.ChangeType
        
        # Only handle Created and Changed events
        if ($changeType -notin @("Created", "Changed")) {
            return
        }
        
        # Debounce rapid changes
        if ($debounceTimers.ContainsKey($path)) {
            $debounceTimers[$path].Dispose()
        }
        
        $timer = New-Object System.Timers.Timer
        $timer.Interval = $DebounceMs
        $timer.AutoReset = $false
        $timer.Add_Elapsed({
            param($sender, $e)
            if (Test-Path $path) {
                Start-Sleep -Milliseconds 200
                Open-HtmlFile -FilePath $path
            }
            $debounceTimers.Remove($path) | Out-Null
            $sender.Dispose()
        })
        
        $debounceTimers[$path] = $timer
        $timer.Start()
    }
    
    Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action | Out-Null
    Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action | Out-Null
    
    $watcher.EnableRaisingEvents = $true
    
    Write-Host "[Auto-Open] Watching for HTML files in: $WatchPath" -ForegroundColor Cyan
    Write-Host "[Auto-Open] HTML files will automatically open when created or modified" -ForegroundColor Gray
    
    # Keep script running
    try {
        while ($true) {
            Start-Sleep -Seconds 60
        }
    } finally {
        $watcher.EnableRaisingEvents = $false
        $watcher.Dispose()
        $debounceTimers.Values | ForEach-Object { $_.Dispose() }
    }
}

Register-FileWatcher




