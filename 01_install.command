#!/bin/sh
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "Install Python 3.10+ including venv, then retry."; exit 1; }
fi
.venv/bin/python -m pip install -r requirements-analysis.txt || exit 1
echo "Ready. Run 02_run_layer_demo.command, then 03_open_workbench.command."
printf "Press Return to close. "
read -r answer
