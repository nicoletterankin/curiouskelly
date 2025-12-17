@echo off
echo ============================================
echo   Curious Kelly Local Development Server
echo ============================================
echo.
echo Starting local server at http://localhost:3000
echo.
echo URL Parameters for testing:
echo   ?autoplay=false  - Disable auto-advance
echo   ?phase=N         - Start at phase N (0-6)
echo   ?debug=true      - Enable debug logging
echo   ?day=N           - Load specific day lesson
echo.
echo Press Ctrl+C to stop
echo ============================================
echo.
cd /d "%~dp0public"
npx serve -l 3000





