@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launch_scan.py
) else (
  python launch_scan.py
)
pause
