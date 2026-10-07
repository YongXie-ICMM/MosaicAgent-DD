@echo off
rem Student / supervisor entry on the analysis computer: process ONE scan handover folder
rem (packed on the instrument computer by acquisition\Auto_Scan\04_pack_handover.bat).
rem Steps: verify -> assemble runs -> incident timeline -> colour/illumination correction
rem -> stitch -> colour check -> report. Results go to <folder>\analysis\; finished steps
rem are skipped when the folder is processed again.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Run 01_install.bat first. Use the complete student package with data and weights.
  pause
  exit /b 1
)
set "FOLDER=%~1"
if "%FOLDER%"=="" (
  echo Drag the handover folder onto this window, or type its path, then press Enter.
  set /p "FOLDER=Handover folder: "
)
if "%FOLDER%"=="" (
  echo No folder given.
  pause
  exit /b 1
)
set "FOLDER=%FOLDER:"=%"
if not exist "%FOLDER%\handover.json" (
  echo "%FOLDER%" has no handover.json. Choose the folder created by 04_pack_handover.bat.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -u tools\handover.py run "%FOLDER%" %2 %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 (
  echo Not completed. Read the last lines above and "%FOLDER%\analysis\handover_status.json"; existing data are preserved.
) else (
  echo Done. Open "%FOLDER%\analysis\REPORT_zh.md" and "%FOLDER%\analysis\mosaic_preview.jpg".
)
pause
