@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_install.bat first. Use the complete student package with data and weights.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -u tools\student_demo.py workbench
if errorlevel 1 echo Not completed. Retain the error message; existing data are preserved.
pause
