# Microscope acquisition

This directory contains the source-only acquisition component of the Digital
Discovery workflow. Release **954-1080p-A1** requests **1920 × 1080** pixels and
checks the actual received and saved images. Images are not resized to satisfy
this requirement. The runtime files are unchanged from the recorded acquisition
revision; file hashes are listed in [acquisition_source.json](acquisition_source.json).

For the analysis workbench, use this `acquisition/` directory as the scanner
root. The accompanying `Auto_Scan/delivery_manifest.json` identifies the original
scanner revision and checks all 16 runtime files, even in a downloaded repository
ZIP with no Git history. It does not require a separate scanner Git checkout.
An intact source manifest confirms source consistency; it does not certify that
the external vendor drivers are installed or that the instrument is ready.

The application controls an XIMC XY stage, acquires optical microscope images,
offers manual candidate review, and preserves image and movement records for
stitching. It does not perform crystal-domain separation or infer crystal
orientation. The English/Chinese switch changes the GUI language.

## Installation on the instrument computer

1. Use Windows with 64-bit Python 3.10 or newer, including Tcl/Tk. Install the
   stage and capture-device drivers required by the instrument manufacturer.
2. Extract the repository into a writable local folder. Keep the complete
   `acquisition/Auto_Scan/` directory together. Run `Auto_Scan/01_setup.bat` to
   install NumPy and OpenCV, or run:

   ```powershell
   python -m pip install -r acquisition/Auto_Scan/requirements_scan.txt
   ```

3. Obtain the appropriate licensed XIMC SDK from the equipment supplier. The
   source bindings require **libximc 3.0.2**. Place the 64-bit `libximc.dll` and
   its required dependency DLLs in `Auto_Scan/drivers/`. The program checks the
   library version before using the controller structures. No vendor DLLs or
   vendor Python wrappers are distributed here. A historical source comment
   referring to a bundled DLL describes the internal distribution, not this
   source-only public repository.
4. Review the configuration block near the top of
   `Auto_Scan/03Auto_Snake_Scan_Camera_v3.py` with the instrument operator.
   Its COM7/COM10 axis assignments, direction conventions, field of view and
   calibration constants belong to the original apparatus and are not portable
   calibration results. Confirm these settings for the actual instrument before
   motion. In particular, `PX_PER_STEP = 25.6` is a legacy default, not a verified
   conversion for the current 1920 × 1080 camera mode.
5. On a scan day start with `Auto_Scan/00_white_balance_then_scan.bat`. It first runs
   `wb_calibrate.py` on a clean bare-substrate field (see below) and then calls the
   unchanged `02_start_scan.bat`, which records the console transcript and opens the
   scanner. Choose **English** in the GUI for an English interface. Connect and
   confirm the microscope preview and actual 1920 × 1080 image size. Use only one
   application instance to control the instrument.

### Colour balance before scanning (added 2026-10-05)

The layer model was reviewed on a capture mode whose bare substrate reads about
RGB (219, 171, 170); the 2026-10-05 comparison showed that a different colour balance
makes it label bare substrate as monolayer ([record](../docs/diagnostics/20261005_colour_balance/README.md)).
`Auto_Scan/wb_calibrate.py` opens the camera through the same discovery and
1920 × 1080 frame contract as the scanner (never the stage), measures the bare-substrate
colour with the analysis estimator, compares it with
`configs/reference_substrate_colour.json` (± 5 %), and:

- with `--auto` switches auto exposure / auto white balance off and adjusts exposure,
  then colour temperature and tint (SDK backend) or the white-balance temperature
  (UVC backend) by measurement-driven secant steps — no direction convention is assumed;
- without writable controls (HDMI capture card, limited driver) shows a live reading and
  guidance while the operator adjusts the camera's own menu; press `a` to accept once the
  reading is within tolerance, `f` to force a record outside tolerance, `q` to quit;
- records `colour_reference_frame.png`, `camera_colour_settings.json` and every step
  under `Auto_Scan/colour_calibration/<timestamp>/` (plus `latest.json`), marking
  `calibrated: false` whenever the reading is outside tolerance, and refuses to record a
  field that is not clean bare substrate (flatness, one luminance plateau, low spread).

The scanner runtime still only reads and records camera settings; it does not apply
this file. Settings can be lost when the camera is reopened or power-cycled, so the
helper runs at the start of each scan day and the analysis colour check on the acquired
images remains the gate. `python wb_calibrate.py --auto --simulate-camera` is an offline
self-test with a rendered field; its records go to `colour_calibration/simulation/` and
are never `latest.json`. Offline tests: `python -m pytest -q acquisition/Auto_Scan/test_wb_calibrate.py`.
Validation on the instrument (does the backend expose the controls, do the settings
survive a reconnect) is still pending.

### Handing a scan day over (added 2026-10-07)

`Auto_Scan/04_pack_handover.bat` runs `pack_handover.py`: it lists the scan sessions under
`mosaic_photos/`, copies the chosen ones (today's by default) intact into one handover folder
together with the day's `history/console_logs/`, `shared_history/`, `camera_history/`,
`colour_calibration/` records and `delivery_manifest.json`, verifies every copied tile against
the hash the scanner recorded in `events.jsonl`, re-checks the 16 runtime files, and writes
`handover.json` plus a Chinese README. Options: `--session <stamp>` (repeatable), `--since`,
`--all`, `--out <dir>`, `--zip`, `--yes`. Runs are never merged or renamed here; the analysis
side (`tools/handover.py`) assembles them. The runtime files are not modified. Offline tests:
`python -m pytest -q acquisition/Auto_Scan/test_pack_handover.py`.

### Camera backend

For a UVC/HDMI capture device, the supplied `Camera_v2.py` uses OpenCV without
`amcam.py`. The default `auto` setting tries a locally installed supported native
SDK and otherwise uses OpenCV. The operator must verify that the selected camera
is the microscope; an explicitly selected **Next camera** action can try another
device. A wrong image size does not silently trigger a different camera.

Native AmScope/ToupTek SDK acquisition is optional. Obtain the matching vendor
Python wrapper and DLLs from the manufacturer, follow its installation and
licensing instructions, and make them discoverable beside `Auto_Scan/` or in the
appropriate environment. Those SDK files are not included in this repository.

## Minimal acquisition-to-stitching check

Start with a **3 × 3** scan in step-confirmation mode. Keep the objective,
illumination, exposure, gain, white balance, axis conventions and physical stage
steps recorded and consistent. A reduction in image pixel count alone does not
justify halving the stage travel. Inspect each accepted frame, normally stop and
disconnect, then close the application and repeat a separate 3 × 3 scan after
reopening. Keep both sessions intact.

Open **Scan and stitching setup** in the analysis workbench and select the
session's raw-image directory. Keep its `session.json` and `events.jsonl` beside
the raw grid images. The setup checks image dimensions and available acquisition
metadata and prepares a profile bound to those inputs. Select or calibrate the
pixel displacement for the actual optical setup; matching dimensions alone do
not establish matching physical fields of view. Check recognizable landmarks
across at least three row and column seams before a longer scan.

The current release still requires instrument acceptance. Offline tests do not
establish long-run camera, USB, stage or Windows/DPI stability. Static stage
parameters are cached for a connection; they do not reliably detect external
changes during a run. After changing the controller configuration, end the run
and reconnect. Camera property readbacks can be incomplete, so retain the actual
acquisition settings and report properties the driver cannot read or lock.

## Recorded outputs and recovery

- The session folder retains accepted raw images, `session.json`, `events.jsonl`
  and diagnostic/review candidates where applicable.
- Camera connection history records the requested and actual image mode. Shared
  history links movement, capture, review and error events for later inspection.
- The launcher retains its transcript under `Auto_Scan/history/console_logs/`.
- A size mismatch blocks acceptance and retains diagnostic evidence where the
  storage device permits it. Correct the camera mode before starting a new run.
- If an image has been saved but logging fails, retain the folder and resolve
  the storage error. Use the program's **Check and continue** recovery flow only
  when it verifies the checkpoint and device state; do not edit JSON records to
  bypass the check or open a second process while the first controls the stage.

The acquisition contract distinguishes image-mode verification from physical
calibration. The fields `expected_image_size`, `actual_image_size`,
`verification_status`, `verification_phase`, `calibration_status` and
`raw_images_resized` are recorded under
`session.scan_config.acquisition_contract`. This release deliberately leaves
physical calibration as `unverified_for_current_mode`.

## Offline validation

From the repository root:

```sh
python -m pip install -r acquisition/Auto_Scan/requirements_scan.txt pytest
python -m pytest -q acquisition/Auto_Scan
python -m pytest -q tools/analysis_workbench/test_bundled_scanner.py
```

On 2026-10-05 the exported suite completed with **168 passed and 27 passed
subtests**. It uses synthetic pixels, fake devices, AST-loaded logic and temporary
records; it opens no camera or stage and makes no model calls. Coverage includes
actual-pixel gates, candidate identity, pre-scan and resume blocking, logging
failure, checkpoint recovery, preview worker lifecycle and console supervision.
The test export contains the fake-DLL helper needed by the motion-timeout test;
its test import is adjusted and recorded in the source manifest. It does not
export unrelated experimental controller test suites or private validation data.
Three additional workbench integration tests check the runtime inventory, avoid
inheriting the enclosing repository's revision, and reject a changed helper
before launching any child process.

Do not use the acquisition program's `--sim` option as an offline test: it
simulates the **stage only** and can still open a real camera. Run the test command
above when no instrument interaction is intended.
