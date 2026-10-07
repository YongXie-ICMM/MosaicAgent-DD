#!/bin/sh
# Entry on the analysis computer: process ONE scan handover folder (packed on the
# instrument computer by acquisition/Auto_Scan/04_pack_handover.bat). Results go to
# <folder>/analysis/; finished steps are skipped when the folder is processed again.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Run 01_install.command first. Use the complete student package with data and weights."
  exit 1
fi
folder="$1"
if [ -z "$folder" ]; then
  printf "Drag the handover folder into this window (or type its path) and press Return: "
  read -r folder
fi
folder="${folder%/}"
folder="$(printf '%s' "$folder" | sed "s/^'//; s/'$//; s/\\ / /g")"
if [ ! -f "$folder/handover.json" ]; then
  echo "'$folder' has no handover.json. Choose the folder created by 04_pack_handover.bat."
  exit 1
fi
shift 2>/dev/null
.venv/bin/python -B -u tools/handover.py run "$folder" "$@"
result=$?
if [ "$result" -ne 0 ]; then
  echo "Not completed. Read the last lines above and $folder/analysis/handover_status.json; existing data are preserved."
else
  echo "Done. Open $folder/analysis/REPORT_zh.md and $folder/analysis/mosaic_preview.jpg."
fi
printf "Press Return to close. "
read -r answer
exit "$result"
