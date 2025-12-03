@echo off
title Kelly Visual Identity - Post-Training
echo.
echo ========================================
echo   KELLY VISUAL IDENTITY - TOMORROW
echo   Run after LoRA training completes!
echo ========================================
echo.
echo Starting post-training script...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0KELLY_TOMORROW.ps1"
pause




