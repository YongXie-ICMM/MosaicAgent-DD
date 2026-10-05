@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo Python environment creation failed. Install Python 3.10+ and retry.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-analysis.txt
if errorlevel 1 (
  echo Installation failed. Keep this message and contact the project lead.
  pause
  exit /b 1
)
echo Ready. Run 02_run_layer_demo.bat, then 03_open_workbench.bat.
pause
