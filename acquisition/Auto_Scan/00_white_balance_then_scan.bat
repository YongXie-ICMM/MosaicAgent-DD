@echo off
rem Student entry for a scan day: calibrate the camera colour balance on a clean bare
rem substrate first, then start the unchanged scanner (02_start_scan.bat).
rem The scanner runtime files are not modified by this launcher.
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>nul
if %errorlevel%==0 (set "PYRUN=py -3") else (set "PYRUN=python")
echo.
echo ==== Step 1 / 2: white balance on a clean bare-substrate field ====
echo Point the microscope at clean bare substrate (no crystals, dust or edges), focus,
echo then press any key. Auto white balance and auto exposure will be switched off.
echo (Chinese guidance is printed by the program. Press Ctrl+C to abort.)
pause >nul
%PYRUN% wb_calibrate.py --auto
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
  echo White balance is within tolerance and recorded under colour_calibration\.
) else if "%RC%"=="2" (
  echo Recorded, but NOT within tolerance. The analysis colour check will flag these images.
) else if "%RC%"=="3" (
  echo Nothing recorded: the field was not clean bare substrate, or the run was stopped.
) else (
  echo The white-balance program reported an error ^(code %RC%^). Keep this window's text.
)
echo.
echo ==== Step 2 / 2: scanner ====
echo Keep lamp, exposure and white balance unchanged from now on.
choice /C YN /M "Start the scanner now"
if errorlevel 2 goto :end
call 02_start_scan.bat
goto :eof
:end
echo Scanner not started. Run 02_start_scan.bat later; do not change camera settings in between.
pause
