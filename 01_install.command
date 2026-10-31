#!/bin/sh
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
export PYTHONUTF8=1
DD_REQUIREMENTS=requirements-analysis.txt
if [ -f trial_kit.json ]; then
  if [ ! -f requirements-student-py39.txt ]; then
    echo "The complete trial kit is missing requirements-student-py39.txt. Re-extract it."
    exit 1
  fi
  DD_REQUIREMENTS=requirements-student-py39.txt
fi
if [ ! -x .venv/bin/python ]; then
  if command -v python3.9 >/dev/null 2>&1; then
    DD_PYTHON=python3.9
  elif command -v python3 >/dev/null 2>&1; then
    DD_PYTHON=python3
  else
    echo "Use 64-bit Python 3.9+ with venv. Python 3.9.18 does not need an upgrade."
    exit 1
  fi
  "$DD_PYTHON" "tools/dd_trial.py" --check-python || {
    echo "Python check failed. Read the details above; Python 3.9.18 is supported."
    exit 1
  }
  "$DD_PYTHON" -m venv ".venv" || {
    echo "Could not create .venv. Use Python 3.9+ with venv, then retry."
    exit 1
  }
fi
.venv/bin/python "tools/dd_trial.py" --check-python || {
  echo "The existing .venv did not pass the Python check and was preserved."
  echo "Ask the teacher to check this environment before continuing."
  exit 1
}
.venv/bin/python -m pip install -r "$DD_REQUIREMENTS" || {
  echo "Dependency installation failed. First setup requires an internet connection."
  exit 1
}
.venv/bin/python "tools/dd_trial.py" --check-environment || {
  echo "Environment check failed. Read its details above and install the missing dependencies."
  exit 1
}
echo "Ready. Run 03_open_workbench.command for the complete workflow."
echo "Optional: 02_run_layer_demo.command checks the supplied demonstration images."
echo "Mac/Linux supports local analysis; camera and motion-controller acquisition requires Windows."
printf "Press Return to close. "
read -r answer
exit 0
