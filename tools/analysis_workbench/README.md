# MosaicAgent-DD analysis workbench

[中文说明](README_zh.md)

A local interface for the Digital Discovery workflow:

1. **Scan acquisition** opens the [AmScope acquisition tool](../../acquisition/README.md) on the Windows instrument computer.
2. **Images and stitching** reviews saved images and provenance; the stitching setup page checks raw image dimensions, the scan grid and acquisition records before preparing parameters.
3. **Layer-number analysis** compares original images with saved U-Net semantic layer predictions and provides the associated area-statistics records.
4. **Spectra and positions** displays existing spectroscopy and sample correspondence records.

The web interface reviews saved products. It does not run new semantic segmentation, collect spectra or issue external model requests. Preparing stitching parameters is distinct from executing the generated command. Acquisition is controlled in the separate native scanner window.

## Start

The workbench itself requires Python 3.10+ and the standard library. Stitching and inference retain the dependencies in the repository requirements; the scanner has its own Windows/vendor setup described in the acquisition guide.

- Windows: double-click `start_workbench.bat`.
- macOS: double-click `start_workbench.command`.
- Terminal, from the repository root: `python3 tools/analysis_workbench/server.py`.
- Open `http://127.0.0.1:8792/` on the same computer.

New configurations have an empty project path. In **Connect tools**, select your project/data folder. The bundled `acquisition/Auto_Scan` is used when available; a saved scanner folder setting takes precedence. The optional **Layer inference results folder** selects a folder containing `results/run_manifest.json`.

Settings are stored in ignored `workbench.local.json`. To keep them elsewhere, run with `--config /path/to/workbench.local.json`. Workbench events and scanner launch logs are saved in `workbench_history/` beside this configuration. Experimental images, weights and personal configuration are not supplied by this interface. Missing inputs produce an empty view or setup message, never demonstration results presented as measurements.

## Read saved layer predictions

Set `inference_results` to an existing inference folder. The current adapter reads `results/run_manifest.json`, whose `samples` entries contain:

- `sample_id`, pairing the original with its saved results;
- `archive`, `member`, `member_sha256`, identifying an original-image ZIP, its member and source hash;
- `outputs`, recording saved output filenames and their SHA-256 values where available.

For each case, the semantic layer map is `results/<sample_id>/mask_color.png`. Optional `overlay_full.jpg` is off by default. Original images are read directly from ZIP members; no extraction or duplicate dataset is created. Source and result hashes are checked when specified in the manifest. Archives must be inside a configured project or inference folder. Recorded foreign absolute paths can be relocated under the configured project using the matching project-directory name; path bounds remain enforced.

If no inference folder is specified, the adapter checks the established DD path `06_analysis/runs/20260924_0409_inference_examples` within the selected project. This compatibility path does not cause automatic access to a private project.

Optional `06_analysis/Figure3_current/fig3_recount_overview.png` and `fig3_recount.json` provide saved statistics and their source regions, numerators and denominators. Opening these files performs no new recount. Model predictions are not independent reference labels. New inference is performed using the pipeline CLI, separately from this review interface.

## Scanner and stitching handoff

The scanner button launches `launch_scan.py` using the current Python interpreter and the scanner's working directory. Connecting hardware and starting acquisition still require the scanner controls. Follow the acquisition guide for required vendor drivers; this public source does not replace them.

The adapter supports an `Auto_Scan` source folder and a flat delivered package containing `launch_scan.py` plus `delivery_manifest.json`. For delivered packages, every manifest-listed file is checked using SHA-256; missing or modified entries block launch. The interface does not automatically replace local files.

Only one process should control the instrument. The wrapper rejects duplicate launches and existing scanner locks. Closing a browser tab does not terminate acquisition. Process exit codes and console-log tails are recorded, but a process starting or exiting is not evidence that acquisition completed.

History summaries distinguish the expected and received image dimensions, pending checks and mismatches. Pixel dimensions do not establish physical calibration. Before stitching, use **Images and stitching → Check new scan images & stitching settings** and validate measured image displacement; do not infer displacement from a change in resolution alone.

On macOS/Linux, acquisition launch is unavailable; review remains usable. The scanner's XY simulation option does not guarantee a simulated camera. Offline software tests are not instrument validation.

## Data and interface safeguards

- Step 03 is read-only, starts no child process and offers no editing or object-partitioning workflow.
- All views retain originals; absent outputs remain absent. Source histories are read rather than repaired or rewritten.
- The application binds to loopback, validates Host/Origin and a process-local action token, and serves registered files only within configured folders.
- HTML artifacts are served without scripts. No arbitrary command execution endpoint is provided.
- Chinese and English controls are available. Spectral requests or record links do not imply new measurements.

## Offline checks

```bash
python3 -m unittest discover -s tools/analysis_workbench -p 'test_*.py'
node --check tools/analysis_workbench/static/app.js
```

The tests cover mocked scanner launching, process/lock handling, package integrity, generated schema-compatible history records, original/result pairing, artifact hashes and path bounds, settings, and read-only layer review. Synthetic test events are constructed in temporary directories; no private experimental log is required. The public test workflow is `.github/workflows/offline-tests.yml`. Test coverage does not claim hardware or physical validation.
