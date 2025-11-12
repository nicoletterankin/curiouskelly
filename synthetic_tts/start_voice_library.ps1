# Voice Library Manager Startup Script
# PowerShell version for Windows

Write-Host "🎵 Starting Voice Library Manager..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required files exist
if (-not (Test-Path "voice_library_data.json")) {
    Write-Host "📊 Generating voice library data..." -ForegroundColor Yellow
    try {
        python generate_voice_library_data.py --verbose
        if ($LASTEXITCODE -ne 0) {
            throw "Data generation failed"
        }
        Write-Host "✅ Voice library data generated successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to generate voice library data" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

if (-not (Test-Path "voice_library_manager_enhanced.html")) {
    Write-Host "❌ HTML interface file not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🌐 Starting web server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "🎵 Voice Library Manager will open in your browser" -ForegroundColor Green
Write-Host "📁 Server: http://localhost:8000" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    python serve_voice_library.py --port 8000
} catch {
    Write-Host "❌ Failed to start server" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
