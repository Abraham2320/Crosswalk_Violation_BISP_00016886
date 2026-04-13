@echo off
title Camera Bridge for Colab

REM ── Configuration ─────────────────────────────────────────────────────────
REM Change YOUR-STATIC-DOMAIN to your actual ngrok free domain.
REM Get one free at: https://dashboard.ngrok.com/cloud-edge/domains
set NGROK_DOMAIN=YOUR-STATIC-DOMAIN.ngrok-free.app

REM Camera index: 0 = first webcam, 1 = second webcam, etc.
set CAMERA_INDEX=0

REM ── Activate virtualenv ───────────────────────────────────────────────────
cd /d "%~dp0"
call CV_venv\Scripts\activate.bat

REM ── Start camera bridge in background window ──────────────────────────────
echo Starting camera bridge on port 8080...
start "Camera Bridge" python camera_bridge.py

REM ── Wait 2 seconds then start ngrok tunnel ────────────────────────────────
timeout /t 2 /nobreak >nul

echo Starting ngrok tunnel to %NGROK_DOMAIN%...
echo.
echo ==========================================================
echo  Stream URL for Colab Cell B3:
echo  https://%NGROK_DOMAIN%/video
echo ==========================================================
echo.
echo Paste that URL into CAMERA_SOURCE in Colab and run Cell B3.
echo Close this window to stop the tunnel.
echo.

ngrok http 8080 --domain=%NGROK_DOMAIN%

pause
