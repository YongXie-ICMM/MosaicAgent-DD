@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m pip install -r requirements_scan.txt
) else (
  python -m pip install -r requirements_scan.txt
)
pause
