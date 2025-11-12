@echo off
echo 🎵 Starting Voice Library Audio Test...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo 🌐 Starting web server...
echo.
echo 🎵 Voice Library Audio Test will open in your browser
echo 📁 Server: http://localhost:8000
echo 🧪 Test Page: http://localhost:8000/quick_audio_test.html
echo.
echo Press Ctrl+C to stop the server
echo.

python serve_voice_library.py --port 8000

pause
