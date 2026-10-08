# MosaicAgent-DD

A publication-scoped workflow for optical microscope acquisition, mosaic construction, semantic layer-number segmentation, and optical spectroscopy records in the Digital Discovery materials study.

[中文操作入口](README_zh.md) · [Architecture](docs/ARCHITECTURE.md) · [Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) · [Resolution and inference](docs/RESOLUTION_AND_INFERENCE.md) · [Acquisition](acquisition/README.md) · [Spectroscopy](docs/SPECTRA.md) · [Status](#status-2026-10-08)

## Start here

**[Download the complete student bundle](https://github.com/YongXie-ICMM/MosaicAgent-DD/releases/latest)**: choose `MosaicAgent-DD-student-complete.zip` (code + two original 1920 × 1080 images + trained weights + numbered launchers). GitHub’s source-code ZIP does not include the image archive or checkpoint. Every bundle records the commit it was built from in `bundle_manifest.json`; a release can be older than `main` (see [Status](#status-2026-10-08)).

**Students receiving the complete trial bundle:** extract the entire folder, then use these numbered launchers:

1. `01_install.bat` (Windows) or `01_install.command` (macOS): create a local Python environment and install the analysis dependencies. Python 3.10+ and internet access for dependency installation are required.
2. `02_run_layer_demo.bat` / `.command`: verify the supplied image and checkpoint hashes, run real U-Net inference on the supplied images, and save a new layer-number result directory.
3. `03_open_workbench.bat` / `.command`: open the local analysis entry (normally `http://127.0.0.1:8792/`; the launcher chooses another port if it is occupied).
4. `04_process_handover.bat` / `.command`: drag in the handover folder of a scan day (see [One folder from the scan day](#one-folder-from-the-scan-day)) and wait for `analysis/REPORT_zh.md`.

On the instrument computer the entries are `acquisition/Auto_Scan/00_white_balance_then_scan.bat` (white balance, then the unchanged scanner) and, after scanning, `acquisition/Auto_Scan/04_pack_handover.bat`.

The complete trial bundle includes the example image archive, trained checkpoint and asset manifest. A Git clone contains code, configuration examples and tests; it is not a substitute for those research assets. See [the data guide](docs/DATA_AND_REPRODUCIBILITY.md). No camera, stage or external model is called by the layer demo. Predicted labels are not independent physical reference measurements.

The step-by-step student manual (Chinese, plain language) is [docs/STUDENT_GUIDE_zh.md](docs/STUDENT_GUIDE_zh.md); students open its PDF, `00_STUDENT_GUIDE_zh.pdf`, at the top of the bundle (rendered from the Markdown by `tools/build_student_guide_pdf.py`). On the instrument computer a scan day starts with `acquisition/Auto_Scan/00_white_balance_then_scan.bat`, which calibrates the camera colour balance on a clean bare-substrate field against `configs/reference_substrate_colour.json` and then calls the unchanged scanner ([details](acquisition/README.md#colour-balance-before-scanning-added-2026-10-05)). Teachers rebuild the bundle with `python tools/build_student_bundle.py`; it verifies the assets, lists every packed file with its SHA-256 in `bundle_manifest.json` and re-checks the archive. `--with-env <file>` additionally packs the group's `.env` (Kimi credentials) into the separately named internal variant `MosaicAgent-DD-student-complete-with-kimi.zip`: the group's students run the stitching with Kimi as in the original pipeline (QC votes, duplicate choice, seam inspection), and the stitch-setup tool omits `--no-ai` whenever credentials are configured. That archive is for the group only; the public bundle stays key-free.

**For code review or a manual installation:**

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements-analysis.txt
python tools/analysis_workbench/server.py
```

The workbench has four entries:

| Entry | Current operation |
|---|---|
| 01 Scan acquisition | Launch the bundled Windows acquisition GUI; inspect recorded acquisition history. |
| 02 Images and stitching | Review source images and outputs; prepare image-bound stitching settings. |
| 03 Layer-number analysis | Compare original images and semantic layer predictions; review counting support and area fractions. |
| 04 Spectra and positions | Open available spectral outputs and source correspondence records. |

The workbench reviews saved products and prepares tool handoffs. New inference is run through the supplied demo or CLI. Entry 03 is restricted to semantic layer-number analysis. Existing research data and development tools are preserved separately.

> **Current demonstration status:** the two original 1920 × 1080 images are on **260 nm SiO₂/Si**, confirmed by the user on 2026-10-05; “276 nm” was their historical label. The supplied 0409 checkpoint is unchanged. The user has now also confirmed that only camera resolution changed from 3840 × 2160 to 1920 × 1080, with the same objective, magnification and physical field of view. This operator confirmation supports the pixel-scale conversion; it is not an independent micrometre calibration. The native baseline and scale comparison have run, but their comparison did not resolve the visual concern that broad apparent-substrate areas are labelled monolayer. A [controlled comparison on 2026-10-05](docs/diagnostics/20261005_colour_balance/README.md) located the cause: not the pixel-count change (a synthetic 1080p downsample of a historical 4K frame is labelled correctly by the same checkpoint, substrate recall ≥ 99 %) but the **colour balance of the new capture mode** (bluer, less green). A per-channel gain alone restores the expected class structure, and the inverse gain reproduces the failure on a historical frame. The demo now records a colour check against `configs/reference_substrate_colour.json` in every run and warns when the balance differs; `python tools/student_demo.py colour-probe` measures it. There are still no independent reference labels or quantified accuracy results for these images, so corrected outputs remain diagnostic previews, not accepted layer measurements. Correct the balance at acquisition first; the separate 70 nm configuration is not substituted for this sample.

## From scan day to mosaic

### One folder from the scan day

On the instrument computer `acquisition/Auto_Scan/04_pack_handover.bat` (`pack_handover.py`, standard library only) copies every scan session of the day intact (tiles, `session.json`, `events.jsonl`, program snapshot), the console transcripts, the acquisition journals, the camera-connection and white-balance records and the scanner manifest into one handover folder with `handover.json` (SHA-256 of every file, one record per run with grid, counts, camera readback and incidents such as `move_failed`); runs are never merged by hand any more. On the analysis computer `04_process_handover` (`python tools/handover.py run <folder>`) verifies the folder, assembles the runs into one flat grid by their original row/column numbers (provenance of every tile, duplicates, missing positions, a registration check across the seams where runs meet), writes the incident timeline, runs the colour/illumination correction and the stitcher, performs the reference-colour check and writes `analysis/REPORT_zh.md`. Each step reads the previous step's outputs from `<folder>/analysis/`, records its result in `handover_status.json` and is skipped when its inputs are unchanged (`--from <step>` redoes a step and everything after it). The report also compares the solved tile positions with a rigid grid model and lists any tile further than a quarter of a tile from it, and it splits the reference-colour check into overall brightness and colour balance, so a darker but correctly balanced substrate can be told apart from a colour cast (the verdict field itself still uses the combined per-channel gain; see [Status](#status-2026-10-08)). When the handover folder is written on the same disk as the scanner's sessions, `pack_handover.py --link` hard-links the files instead of copying them.

### Colour and illumination correction

When a scan was interrupted and resumed, the camera's automatic exposure drifted, or the field illumination is uneven, the mosaic shows steps at tile boundaries. `python tools/colour_match_grid.py --data <scan folder> [--stitch]` estimates the illumination field from the tiles themselves (per-pixel median of mean-normalised tiles), registers every neighbouring pair and measures the colour ratio in the overlap, then solves one level per column plus a penalised deviation per tile so that all 1600 overlaps agree (originals untouched: the derived dataset holds corrected copies, with hard links for tiles that need no change), writes derived `session.json`/`events.jsonl` with the new hashes, the field and the gain of every tile, a measured stitch profile, and optionally runs `run_stitch.py` (Kimi when configured). `--report-only` measures without writing a dataset; `--gain-mode column` keeps one gain per column; `--no-flat-field` skips the illumination correction. First use on 2026-10-06: a 16 × 53 scan whose last two columns came from a second camera session (gain 27 vs 21) needed column gains of about (0.95, 0.94, 0.91); two smaller steps (≈3 %) inside the first run and per-tile exposure differences of up to 4 % were corrected as well, and the illumination falloff of 7 % (left/right) and 3 % (top/bottom) was removed before stitching. On the 18 × 50 scan 261002AM (one run) the same tool found no column steps, per-tile deviations up to 4 % and an illumination falloff of 3–4 % (left/right) and ≤ 1 % (top/bottom).

### Registration guard (added 2026-10-07)

`register.py` builds column chains from the vertical overlaps and, where a chain breaks, decides each segment's row offset from lateral correlations. A segment now leaves the stage's row numbering only on real evidence (`decide_segment_offset`): a single lateral edge must reach NCC 0.5, and a non-zero offset must beat "no offset" by 0.1 in mean NCC unless it has more good edges; otherwise the segment keeps its recorded rows. This fixed two chip-edge tiles of 261002AM, half bare substrate and half the dark area beyond the chip, that one edge at NCC 0.27 had placed 2–3 rows too high (a ghost over the substrate and a hole at the chip edge). Multi-tile segments with strong evidence, such as those of Figure 3a (NCC 0.78–0.92), are unaffected; `tests/test_register_segment_offset.py` fixes the behaviour. `run_stitch.py` also saves `positions_tids.json` beside `positions.npy`, which the handover report uses for its placement check.

## Run on your own data

Run commands from the repository root. Start with the input check in **02 → Scan and stitching setup**, and retain the generated command and profile. Image file size in MB does not determine stitching geometry. For a new camera mode, verify actual pixels, overlap and physical calibration.

Stitch setup also saves `layer_input_contract.json` beside `profile.json` and `preflight.json`. It carries actual raw-image dimensions, source bindings and acquisition verification into layer analysis; it does not select a checkpoint or establish physical calibration. For the current two-image bundle, the demo reads the recorded field-of-view confirmation and actual image dimensions, then automatically uses **256-pixel native tiles / 32-pixel overlap**, resampled to the unchanged **512-pixel model window**. Only legacy assets without a `capture_geometry` field retain the native 512/64 baseline. An explicit but unconfirmed or invalid geometry record stops the run for correction. Read [Resolution and inference](docs/RESOLUTION_AND_INFERENCE.md) before using a new capture mode.

For an existing mosaic, edit `configs/layer_analysis.example.json` to point to your trusted checkpoint and image, then run:

```sh
python flakepipeline/run.py --config configs/layer_analysis.example.json --plan
python flakepipeline/run.py --config configs/layer_analysis.example.json --no-ai --no-repair
```

The example explicitly disables model supervision and automatic repair. Missing inputs produce an error or blocked plan. Outputs include stage manifests, masks, statistical records and review requests when applicable. Changing a proposed mask or exclusion requires an evidence check; a successful process exit alone does not certify a measurement.

| Label value | Meaning | RGB display |
|---|---|---|
| 0 | Bare substrate | 0, 0, 0 |
| 1 | Bilayer (2L) | 128, 0, 0 |
| 2 | Monolayer (1L) | 0, 128, 0 |
| 3 | Thick layer (TL) | 128, 128, 0 |

The layer-fraction denominator is **1L + 2L + TL**, excluding bare substrate. Valid support and any region exclusion must be recorded alongside the result. The demo applies no region exclusion.

## Status (2026-10-08)

- **Checked on real scans.** *260128* (16 × 53 = 848 tiles; the first run stopped after ≈ 152 min with `Y movement failed: rc=-1`, and a second camera session at gain 27 instead of 21 re-shot the last two columns): column steps of 3.6 %, 3.8 % and 5.2 % were corrected; the overlap mismatch fell from 1.28 % to 0.58 % RMS between neighbouring columns and from 0.59 % to 0.23 % along columns; 848/848 tiles stitched with a registration residual of 0.88 px; rebuilt as a two-run handover folder, the processor produced a byte-identical mosaic. *261002AM* (18 × 50 = 900 tiles, one run in step-confirm mode): no column steps, per-tile deviations ≤ 4 %, 900/900 tiles, residual 1.95 px, every tile within 46 px of the rigid grid model after the registration guard; the bare substrate is 16 % darker than the reference while its colour balance is within 0.4 %. That scan came from a modified v3.4 scanner that no longer matches its own manifest, and its handover lacked console transcripts, journals and white-balance records because the packer was not used.
- **Not yet validated.** White-balance calibration on the instrument (`wb_calibrate.py` is tested offline only); layer-number accuracy on the new 1080p images (there are no independent reference labels, so all outputs there remain diagnostic); the motion-failure policy sketched in [the stage-error log](docs/diagnostics/stage_error_incident_log_zh.md), which belongs in the scanner's source repository and needs instrument tests.
- **Known issues.** (1) The stitcher's closed-form focus gate is content-biased on single-shot flat grids: it marked 453 of 900 tiles of 261002AM (455 of 848 of 260128) as defocused; all were rescued because nothing else covers their positions, but they cost model calls and clutter the QC record. (2) The reference-colour verdict `colour_balance_differs_from_reference` is raised whenever any channel gain differs by more than the tolerance, including a pure brightness difference; the handover report prints the split, the demo's warning does not. (3) Colour correction needs a complete grid; an incomplete handover skips the colour and stitch steps. (4) The workbench has no page for handover folders yet. (5) The release asset v0.1.0 was built from `5a20621` on 2026-10-05, before the colour diagnosis, white-balance calibration, student manual, colour correction, handover and registration work; a new public bundle built from `main` should be published.
- **Continuing the review.** [docs/CHATGPT_HANDOFF_zh.md](docs/CHATGPT_HANDOFF_zh.md) (Chinese) holds the current state, the evidence, the rules and an ordered backlog, with a prompt that can be pasted into ChatGPT or Codex; [AGENTS.md](AGENTS.md) lists the rules and checks for any coding agent working in this repository.

## Optional model supervision

Kimi supervision exists in the stitching, image-audit and spectral tools; it is not needed to launch the workbench or run the offline demo. Read [the architecture](docs/ARCHITECTURE.md) before enabling it. Configure credentials explicitly with environment variables or a local `.env` based on `.env.example`. Credentials and local paths are not committed. Enabling an external model may send the selected images or spectral plots to that service; `--no-ai` disables these requests.

## Validation and availability

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary records, synthetic inputs and mocked devices; GitHub Actions ("Offline source checks") runs them, `compileall` and `node --check` on the two workbench scripts on every push. Instrument validation remains separate; the exported scanner adds mode checks to a previously tested acquisition base and still needs acceptance on the actual microscope. The complete student bundle is also checked using the real supplied checkpoint and example images. That is a reproducibility check, not a new accuracy evaluation.

Source provenance, modifications for public portability, and third-party dependency boundaries are documented in [PROVENANCE.md](docs/PROVENANCE.md). No repository-wide license grant has been added to this initial public snapshot; third-party dependencies and vendor SDKs retain their own terms.

Development follow-up: [Claude handoff and remaining experiments](docs/CLAUDE_HANDOFF.md) · [continuation prompt and backlog for ChatGPT/Codex (Chinese)](docs/CHATGPT_HANDOFF_zh.md) · [rules for coding agents](AGENTS.md).
