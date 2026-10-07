@echo off
rem Student entry after a scan day: pack every scan session of today, the console logs,
rem the acquisition journals, camera and white-balance records into ONE handover folder
rem with a manifest and checksums (pack_handover.py). Nothing is merged or edited; the
rem scanner runtime files are not touched.
cd /d "%~dp0"
set PYTHONUTF8=1
where py >nul 2>nul
if %errorlevel%==0 (set "PYRUN=py -3") else (set "PYRUN=python")
echo.
echo ==== Pack today's scan for the analysis computer ====
echo The handover folder will be created under handover\ next to this script unless you
echo type another destination (for example a USB drive such as E:\handover).
set /p "DEST=Destination folder (Enter = handover\ here): "
set "ZIPFLAG="
choice /C YN /M "Also create a .zip of the folder (needs twice the space)"
if errorlevel 2 goto :nozip
set "ZIPFLAG=--zip"
:nozip
if "%DEST%"=="" (
  %PYRUN% pack_handover.py %ZIPFLAG%
) else (
  %PYRUN% pack_handover.py --out "%DEST%" %ZIPFLAG%
)
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
  echo Done. Hand over the whole folder printed above, or its .zip, unchanged.
) else (
  echo Packing did not complete. Keep this window's text and ask the project lead.
)
pause
