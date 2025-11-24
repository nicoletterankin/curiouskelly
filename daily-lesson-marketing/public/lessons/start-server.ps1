# Simple HTTP Server for Calendar Page
# This script starts a local web server to avoid CORS issues

Write-Host "Starting local web server for Calendar Page..." -ForegroundColor Green
Write-Host ""
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Open: http://localhost:8000/calendar-page.html" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Check if Python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
}

if ($pythonCmd) {
    Write-Host "Using Python HTTP server..." -ForegroundColor Green
    & $pythonCmd -m http.server 8000
} else {
    Write-Host "Python not found. Trying Node.js..." -ForegroundColor Yellow
    
    # Try Node.js http-server
    if (Get-Command npx -ErrorAction SilentlyContinue) {
        Write-Host "Using Node.js http-server..." -ForegroundColor Green
        npx --yes http-server -p 8000 -c-1
    } else {
        Write-Host "ERROR: Neither Python nor Node.js found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install one of the following:" -ForegroundColor Yellow
        Write-Host "  - Python 3: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "  - Node.js: https://nodejs.org/" -ForegroundColor White
        Write-Host ""
        Write-Host "Or use VS Code's Live Server extension" -ForegroundColor Yellow
        exit 1
    }
}

