#!/bin/sh
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Run 01_install.command first. Use the complete student package with data and weights."
  exit 1
fi
.venv/bin/python -B -u tools/student_demo.py workbench
result=$?
if [ "$result" -ne 0 ]; then echo "Not completed. Retain the error message; existing data are preserved."; fi
printf "Press Return to close. "
read -r answer
exit "$result"
