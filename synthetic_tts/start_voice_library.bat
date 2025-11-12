@echo off
echo 🎵 Starting Voice Library Manager...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "voice_library_data.json" (
    echo 📊 Generating voice library data...
    python generate_voice_library_data.py --verbose
    if errorlevel 1 (
        echo ❌ Failed to generate voice library data
        pause
        exit /b 1
    )
)

if not exist "voice_library_manager_enhanced.html" (
    echo ❌ HTML interface file not found
    pause
    exit /b 1
)

echo 🌐 Starting web server...
echo.
echo 🎵 Voice Library Manager will open in your browser
echo 📁 Server: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python serve_voice_library.py --port 8000

pause
