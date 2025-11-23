param(
    [ValidateSet("stack", "gateway", "classroom", "deps")]
    [string]$Target = "stack",
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start",
    [switch]$SkipDependencies,
    [switch]$KeepDependencies,
    [switch]$SkipHtmlWatcher
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptRoot "..")
$composeFile = Join-Path $repoRoot "docker-compose.dev.yml"
$dockerArgs = @("compose", "-f", $composeFile)

function Require-Command {
    param(
        [string]$Name,
        [string]$InstallUrl
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "Required command '$Name' was not found on PATH." -ForegroundColor Red
        if ($InstallUrl) {
            Write-Host "Install instructions: $InstallUrl" -ForegroundColor Yellow
        }
        throw "Missing dependency: $Name"
    }
}

function Invoke-DockerCompose {
    param(
        [string[]]$AdditionalArgs
    )

    Require-Command -Name "docker" -InstallUrl "https://docs.docker.com/desktop/install/windows-install/"

    if (-not (Test-Path $composeFile)) {
        throw "Could not find docker-compose.dev.yml at $composeFile"
    }

    & docker @($dockerArgs + $AdditionalArgs) | Write-Output
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Command was: docker $($dockerArgs + $AdditionalArgs)" -ForegroundColor Red
        throw "docker compose command failed ($LASTEXITCODE)"
    }
}

function Wait-ForPort {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $isOpen = Test-NetConnection -ComputerName "localhost" -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($isOpen) {
            Write-Host "✓ $Name ready on port $Port" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for $Name on port $Port"
}

function Start-Dependencies {
    Write-Host "Starting local infrastructure (Postgres, Redis, Meilisearch, ClickHouse)..." -ForegroundColor Cyan
    Invoke-DockerCompose -AdditionalArgs @("up", "-d", "--remove-orphans")

    Wait-ForPort -Port 5432 -Name "Postgres"
    Wait-ForPort -Port 6379 -Name "Redis"
    Wait-ForPort -Port 7700 -Name "Meilisearch"
    Wait-ForPort -Port 8123 -Name "ClickHouse"
}

function Stop-Dependencies {
    param(
        [switch]$Silent
    )

    try {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            if (-not $Silent) {
                Write-Host "Docker CLI not available; nothing to stop." -ForegroundColor Yellow
            }
            return
        }

        if (-not (Test-Path $composeFile)) {
            if (-not $Silent) {
                Write-Host "Compose file missing; skipping teardown." -ForegroundColor Yellow
            }
            return
        }

        Write-Host "Stopping local infrastructure..." -ForegroundColor Cyan
        & docker @($dockerArgs + @("down"))
        if ($LASTEXITCODE -ne 0 -and -not $Silent) {
            Write-Host "docker compose down exited with $LASTEXITCODE" -ForegroundColor Yellow
        }
    } catch {
        if (-not $Silent) {
            Write-Host "Failed to stop services: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

function Show-Status {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker CLI not found; install Docker Desktop to inspect status." -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $composeFile)) {
        Write-Host "Compose file missing at $composeFile" -ForegroundColor Yellow
        return
    }

    Invoke-DockerCompose -AdditionalArgs @("ps")
}

function Open-Browser {
    param(
        [string]$Url,
        [string]$Description
    )
    
    try {
        Start-Process $Url -ErrorAction SilentlyContinue | Out-Null
        Write-Host "  ✓ Opened $Description" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Could not auto-open $Description (visit $Url manually)" -ForegroundColor Yellow
    }
}

function Run-DevScript {
    param(
        [string]$ScriptName,
        [switch]$AutoOpen
    )

    Require-Command -Name "pnpm" -InstallUrl "https://pnpm.io/installation"
    Push-Location $repoRoot
    try {
        Write-Host "Running pnpm $ScriptName (Ctrl+C to stop)..." -ForegroundColor Green
        
        # Auto-open relevant pages based on what's starting
        if ($AutoOpen) {
            Start-Job -ScriptBlock {
                param($scriptName)
                Start-Sleep -Seconds 5
                switch ($scriptName) {
                    "dev:gateway" {
                        Start-Process "http://localhost:4000/docs" -ErrorAction SilentlyContinue
                        Start-Process "http://localhost:4000/health" -ErrorAction SilentlyContinue
                    }
                    "dev:classroom" {
                        Start-Process "http://localhost:4100/health" -ErrorAction SilentlyContinue
                    }
                    "dev:stack" {
                        Start-Sleep -Seconds 3
                        Start-Process "http://localhost:4000/docs" -ErrorAction SilentlyContinue
                    }
                }
            } -ArgumentList $ScriptName | Out-Null
        }
        
        & pnpm run $ScriptName
        $exit = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    if ($exit -ne 0 -and $exit -ne 130 -and $exit -ne -1073741510) {
        throw "pnpm $ScriptName exited with code $exit"
    }
}

if ($Action -eq "status") {
    Show-Status
    exit 0
}

if ($Action -eq "stop") {
    Stop-Dependencies
    exit 0
}

if ($Target -eq "deps" -and $SkipDependencies) {
    Write-Host "You chose the 'deps' target but also set -SkipDependencies. Nothing to start." -ForegroundColor Yellow
    exit 0
}

if ($Target -eq "deps") {
    Start-Dependencies
    if ($KeepDependencies) {
        Write-Host "Leaving infrastructure running. Use '-Action stop' later to stop containers." -ForegroundColor Yellow
        exit 0
    }

    Write-Host "Infrastructure is running. Press Ctrl+C to stop it." -ForegroundColor Yellow
    try {
        while ($true) {
            Start-Sleep -Seconds 60
        }
    } finally {
        Stop-Dependencies
    }
}

$depsStarted = $false
$htmlWatcherJob = $null

if (-not $SkipDependencies) {
    Start-Dependencies
    $depsStarted = $true
} else {
    Write-Host "Skipping Docker dependencies. Make sure Postgres/Redis/Meilisearch/ClickHouse are already running." -ForegroundColor Yellow
}

# Start HTML file watcher in background
if (-not $SkipHtmlWatcher) {
    $htmlWatcherScript = Join-Path $scriptRoot "auto-open-html.ps1"
    if (Test-Path $htmlWatcherScript) {
        Write-Host "Starting HTML auto-opener..." -ForegroundColor Cyan
        $htmlWatcherJob = Start-Job -ScriptBlock {
            param($scriptPath, $repoPath)
            Set-Location $repoPath
            & $scriptPath -WatchPath $repoPath
        } -ArgumentList $htmlWatcherScript, $repoRoot
        Write-Host "  ✓ HTML files will auto-open when created/modified" -ForegroundColor Green
    }
}

try {
    switch ($Target) {
        "stack" {
            Run-DevScript -ScriptName "dev:stack" -AutoOpen
        }
        "gateway" {
            Run-DevScript -ScriptName "dev:gateway" -AutoOpen
        }
        "classroom" {
            Run-DevScript -ScriptName "dev:classroom" -AutoOpen
        }
    }
} finally {
    # Stop HTML watcher
    if ($htmlWatcherJob) {
        Write-Host "Stopping HTML watcher..." -ForegroundColor Yellow
        Stop-Job $htmlWatcherJob -ErrorAction SilentlyContinue
        Remove-Job $htmlWatcherJob -ErrorAction SilentlyContinue
    }
    
    if ($depsStarted -and -not $KeepDependencies) {
        Stop-Dependencies -Silent
    } elseif ($depsStarted -and $KeepDependencies) {
        Write-Host "Leaving infrastructure containers running as requested (-KeepDependencies)." -ForegroundColor Yellow
    }
}

