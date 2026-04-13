@echo off
title Camera Bridge for Colab

REM Camera index: 0 = first webcam, 1 = second webcam, etc.
set CAMERA_INDEX=0

REM ── Install cloudflared (one-time, if not already installed) ──────────────
where cloudflared >nul 2>&1
if errorlevel 1 (
    echo Installing cloudflared (one-time setup)...
    winget install Cloudflare.cloudflared
    if errorlevel 1 (
        echo.
        echo ERROR: winget failed. Download cloudflared manually from:
        echo   https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
        echo Rename it to cloudflared.exe and place it next to this bat file, then re-run.
        pause
        exit /b 1
    )
)

REM ── Activate virtualenv ───────────────────────────────────────────────────
cd /d "%~dp0"
call CV_venv\Scripts\activate.bat

REM ── Start camera bridge in background window ──────────────────────────────
echo Starting camera bridge on port 8080...
start "Camera Bridge" python camera_bridge.py

REM ── Wait 2 seconds then start cloudflared tunnel ─────────────────────────
timeout /t 2 /nobreak >nul

echo Starting Cloudflare tunnel (no account needed)...
echo.
echo ==========================================================
echo  Watch for a line that says:
echo    https://xxxx-xxxx-xxxx.trycloudflare.com
echo  Copy that URL and add /video at the end.
echo  Paste it into CAMERA_SOURCE in Colab Cell B3.
echo ==========================================================
echo.
echo Close this window to stop the tunnel.
echo.

cloudflared tunnel --url http://localhost:8080

pause
