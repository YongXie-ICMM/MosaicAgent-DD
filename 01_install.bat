@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PYTHONUTF8=1"
set "DD_REQUIREMENTS=requirements-analysis.txt"
if not exist "trial_kit.json" goto select_python
if not exist "requirements-student-py39.txt" goto requirements_failed
set "DD_REQUIREMENTS=requirements-student-py39.txt"

:select_python
if exist ".venv\Scripts\python.exe" goto check_environment
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.9 --version >nul 2>nul
if errorlevel 1 goto use_python
py -3.9 "tools\dd_trial.py" --check-python
if errorlevel 1 goto python_failed
py -3.9 -m venv ".venv"
if errorlevel 1 goto venv_failed
goto check_environment

:use_python
where python >nul 2>nul
if errorlevel 1 goto use_py_default
python "tools\dd_trial.py" --check-python
if errorlevel 1 goto python_failed
python -m venv ".venv"
if errorlevel 1 goto venv_failed
goto check_environment

:use_py_default
where py >nul 2>nul
if errorlevel 1 goto python_failed
py -3 "tools\dd_trial.py" --check-python
if errorlevel 1 goto python_failed
py -3 -m venv ".venv"
if errorlevel 1 goto venv_failed

:check_environment
if not exist ".venv\Scripts\python.exe" goto venv_failed
".venv\Scripts\python.exe" "tools\dd_trial.py" --check-python
if errorlevel 1 goto python_failed
".venv\Scripts\python.exe" -m pip install -r "%DD_REQUIREMENTS%"
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" "tools\dd_trial.py" --check-environment
if errorlevel 1 goto environment_failed
echo Ready. Run 03_open_workbench.bat for the complete workflow.
echo Optional: 02_run_layer_demo.bat checks the supplied demonstration images.
pause
exit /b 0

:python_failed
echo Python check failed. Use 64-bit Python 3.9+ with Tcl/Tk.
echo The instrument's Python 3.9.18 is supported; no upgrade is required.
echo Read the check details above. Existing environments and data were preserved.
pause
exit /b 1

:venv_failed
echo Could not create or find the local .venv environment.
echo Use 64-bit Python 3.9+ with venv and Tcl/Tk, then retry.
pause
exit /b 1

:requirements_failed
echo The complete trial kit is missing requirements-student-py39.txt.
echo Re-extract the complete kit. No unpinned fallback was installed.
pause
exit /b 1

:install_failed
echo Dependency installation failed. An internet connection is required on first setup.
echo Keep this message and ask the teacher to check the installation.
pause
exit /b 1

:environment_failed
echo The complete environment check did not pass. Read its JSON details above.
echo Install the missing dependencies or Windows runtime listed by the check, then retry.
echo No camera or motion-controller SDK was opened by these checks.
pause
exit /b 1
