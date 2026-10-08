# Instructions for coding agents

This repository is the publication-scoped DD workflow: optical microscope acquisition, mosaic stitching, semantic layer-number segmentation (0 substrate, 1 bilayer, 2 monolayer, 3 thick) with statistics, and optical spectroscopy records. Read `README.md`, `docs/CLAUDE_HANDOFF.md` and `docs/CHATGPT_HANDOFF_zh.md` (current state, evidence and backlog) before editing.

## Rules

1. **Scanner runtime is locked.** Do not modify the 16 hash-locked runtime files listed in `acquisition/Auto_Scan/delivery_manifest.json`. Scanner behaviour (for example the motion-failure policy) is changed in the scanner's source repository and validated on the instrument, not here. Check after every change:
   `python -c "import sys; sys.path.insert(0, 'acquisition/Auto_Scan'); import pack_handover as p; from pathlib import Path; print(p.check_runtime(Path('acquisition/Auto_Scan'))['ok'])"` must print `True`.
2. **Originals are read-only.** Never edit or overwrite source images, `weights/`, `data/demo/` assets, historical outputs or manifests. Derived data go to new folders (copies or hard links) with recorded provenance and hashes.
3. **No silent default changes.** Changing default preprocessing, thresholds, model selection or QC behaviour needs a controlled before/after comparison with numbers, a test, a note in the docs, and a flag that keeps the old behaviour.
4. **Sample and model.** The current samples are on 260 nm SiO2/Si with the unchanged 0409 checkpoint. Never substitute the 70 nm / 0815 configuration or other weights.
5. **Scope.** Workbench entry 03 is semantic layer-number analysis only. Do not restore domain/instance annotation tools or add twist-angle inference.
6. **Credentials.** Never print, log, commit or upload `.env` contents or `KIMI_API_KEY`. Tests run offline (`--no-ai`, `ai_mode: "off"`) and never call external models. The internal bundle `MosaicAgent-DD-student-complete-with-kimi.zip` must never become a release asset; the public bundle stays key-free.
7. **Evidence.** Predicted labels are not measurements. Do not report accuracy without independent reference labels; label synthetic tests as synthetic.
8. **Stability for students.** Do not refactor directories or rename the numbered launchers (`01_`–`04_`, `acquisition/Auto_Scan/00_`, `04_pack_handover.bat`) or public CLI flags.
9. **Git.** Work on a branch and open a pull request (or hand over a patch). Do not push to `main`, tag or publish releases; the owner does that.

## Checks

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q .
node --check tools/analysis_workbench/static/app.js
node --check tools/stitch_setup/static/app.js
python register.py            # registration self-test
```

CI ("Offline source checks", `.github/workflows/offline-tests.yml`) runs the same pytest, compileall and node checks on every push.

## Conventions

- One topic per pull request; small, evidence-backed changes with tests on synthetic fixtures. `tests/handover_fixture.py` writes scan records with the scanner's own classes; reuse it for handover and stitching tests.
- Commit messages: an imperative summary line, then what changed and why, with numbers from real data where available.
- Student-facing text is plain Chinese. After editing `docs/STUDENT_GUIDE_zh.md`, re-render the PDF with `python tools/build_student_guide_pdf.py`; the owner rebuilds bundles with `python tools/build_student_bundle.py`.
- Keep `README.md`, `README_zh.md` and `docs/CLAUDE_HANDOFF.md` consistent with behaviour changes.
