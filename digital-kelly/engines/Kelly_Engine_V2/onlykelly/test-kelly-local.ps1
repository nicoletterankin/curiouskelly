# ╔══════════════════════════════════════════════════════════════════╗
# ║              KELLY V2 - LOCAL TEST SERVER                        ║
# ║                Test before deploying!                             ║
# ╚══════════════════════════════════════════════════════════════════╝

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🧪 KELLY V2 LOCAL TEST SERVER" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ProjectRoot = $PSScriptRoot
$BuildPath = Join-Path $ProjectRoot "Builds\WebGL"
$Port = 8000
$Url = "http://localhost:$Port"

# Step 1: Verify Build Exists
Write-Host "📁 Checking build folder..." -ForegroundColor Yellow

if (-not (Test-Path $BuildPath)) {
    Write-Host "❌ ERROR: Build folder not found!" -ForegroundColor Red
    Write-Host "   Expected: $BuildPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Run Unity build first:" -ForegroundColor Yellow
    Write-Host "   Kelly > Build > 🚀 Build WebGL (Production)" -ForegroundColor Yellow
    exit 1
}

$IndexFile = Join-Path $BuildPath "index.html"
if (-not (Test-Path $IndexFile)) {
    Write-Host "❌ ERROR: index.html not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build found: $BuildPath" -ForegroundColor Green
Write-Host ""

# Step 2: Check for Python
Write-Host "🐍 Checking for Python..." -ForegroundColor Yellow

$PythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonCmd = "python3"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = "py"
}

if (-not $PythonCmd) {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host "   Install Python from https://python.org" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Alternative: Use Node.js http-server" -ForegroundColor Yellow
    Write-Host "   npm install -g http-server" -ForegroundColor Gray
    Write-Host "   cd Builds/WebGL && http-server -p 8000" -ForegroundColor Gray
    exit 1
}

Write-Host "✅ Python found: $PythonCmd" -ForegroundColor Green
Write-Host ""

# Step 3: Start Server
Write-Host "🌐 Starting local server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║   Kelly is running at: $Url                        ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "║   Press Ctrl+C to stop the server                                ║" -ForegroundColor Cyan
Write-Host "║                                                                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Open browser
Write-Host "🌍 Opening browser..." -ForegroundColor Yellow
Start-Process $Url

# Change to build directory and start server
Set-Location $BuildPath
& $PythonCmd -m http.server $Port

