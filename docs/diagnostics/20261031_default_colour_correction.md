# Default stitching colour correction, 31 October 2026

The owner requested automatic colour correction in ordinary stitching. The CLI and setup-generated command now estimate per-tile RGB gains from registered neighbouring overlaps by default. Students do not select a colour mode.

## Evidence

Received inputs: `scan_diagnostics_3x3_20261031.zip`, `scan_diagnostics_1000_20261031.zip`, `260116PM_5mg_part1.zip`, `260116PM_5mg_part2.zip` and `3816.pic_hd`. The last file is an Inkscape-exported PNG, not the original stitching state. Sources and hashes remain in the local experiment archive; raw data are not committed here.

The independent session records show 9/9 and 1000/1000 saved positions, no terminal/capture failure, and zero command return codes for all 8 and 999 scan movements. Both diagnostic ZIPs contain identical process-wide communications/XILOG files; they are one device run, not two independent copies of each movement.

The original image grid has20columns by50rows and `invert_x=true`. Late acquisition columns lie towards the mosaic's left. Registered original-image overlaps show a colour change around point879, c17/r21, 07:37:32 UTC, and recovery at point949, c18/r48, 08:04:51 UTC. The inferred interval spans about70 positions/27minutes; RGB differences are about4–6%. It crosses parts of two columns. One gain per entire column also alters an unaffected half-column.

A diagnostic comparison with the unchanged historical `colour_match_grid.solve_gains` used430 accepted horizontal/vertical fitting overlaps in the last5columns (250tiles): log-ratio RMS2.156% before,1.292% with column gains,0.394% with tile gains. These are fitting residuals, not independent accuracy or complete-mosaic acceptance. All100 sampled image hashes match their capture records; source session/events match the1000-position diagnostic ZIP byte-for-byte.

## Implementation and controls

- `stitch_colour.py` reads original ZIP/loose-file pixels, uses the renderer's RGB-converted flat field, and calls the existing overlap measurement and tile solver. Native dimensions, IDs, positions, QC thresholds and model weights do not change.
- `blend.render(..., tile_gains_rgb=...)` combines gains/flat field in float before one clipping step. A tile crossing rendering bands is not corrected twice. Omitting gains reproduces the deterministic historical decoded-pixel control.
- `work/colour_correction.json` saves source binding, accepted/rejected edges, connected components, RGB factors and residuals. `state.json` records the outcome. Old success is cleared before a rerun, including one that subsequently fails.
- Automatic matching requires a contiguous complete rectangular grid and connected measured overlap graph. Unsupported gains stop the run instead of silently producing an uncorrected success. The older incomplete-grid backlog remains.
- Prior `colour_match_manifest.json` is accepted only when its complete image set and all selected SHA-256 hashes match. Such data are not corrected twice; a changed manifest changes the work binding.
- Read-only preflight does not estimate colour, create caches, render, or call a model. Switching auto/off or reusing old unlabelled work needs a fresh work directory; `--force` cannot bypass the binding.

Developer reproduction: `--colour-correction off` with separate work/output paths. Students keep the normal entry without choosing. The handover colour step remains; its verified derived dataset bypasses new gain estimation.

## Acquisition limits and follow-up

Relative gains cannot recover clipping, establish reflectance, or replace reference-substrate calibration. They cannot establish why the camera image changed. AE/AWB modes are not in the received logs, and exposure/white-balance value readback is unavailable. The C1 scanner did not call mode-changing setters.

The bounded structured log retains32,317records from07:57:26–08:33:31 UTC; preview grab-frame begin/end accounts for70.48% of this tail. Independent scan events remain complete; XILOG spans02:22:09–08:33:28 UTC. The1000-scan diagnostic pointer retains some old3×3context, while actual live event contexts are correct. No hardware fault occurred, so fault-triggered stack/state bundles have not yet been tested on the instrument. These are acquisition-repository follow-up items, not changes to the scanner locked in this repository.

A latent camera-interface defect was also found: `Camera_v2.py` labels DirectShow0.75 as manual and0.25 as auto. OpenCV4.10's [DirectShow setter](https://github.com/opencv/opencv/blob/4.10.0/modules/videoio/src/cap_dshow.cpp#L3233-L3243) uses `cvRound(value)==1` for automatic, with0/1 explicit manual/auto values. This scanner did not call that setter during the trial, so the defect does not establish the cause or contradict the user's observation that auto-exposure was disabled. Instrument-side fixes need separate AmScope-Camera validation.

## Validation and delivery

Local source suite:829passed,1skipped,84subtests passed. New checks cover default/binding/provenance/failure behaviour, ZIP/loose equivalence, a synthetic mid-column colour change, connected evidence, RGB/BGR order, single-clipping/band behaviour and historical decoded pixels. Real1000-tile module-replay results and package hashes are retained in the local31October experiment/handoff folder after completion.

Local candidate branch: `codex/default-colour-match-20261031`, based on986d7c4. No GitHub merge or public release is implied. Extract the analysis package separately; it does not replace the instrument's hardware-validated C1 acquisition installation.
