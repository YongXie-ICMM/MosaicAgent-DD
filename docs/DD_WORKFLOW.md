# Guided DD workflow: acquisition to spatial layer maps

This guide describes the current local workflow and the key-free student trial kit prepared on 1 November 2026. The browser interface is currently in Chinese; the seven numbered steps below explain its operation in English. The integration remains subject to a complete trial on the Windows instrument computer.

**Scan → log intake → automatic colour correction → stitching → review → layer inference → Figure 4 spatial maps**

## Start and prerequisites

- The trial kit supports the instrument's **64-bit Python 3.9.18**, including Tcl/Tk. `01_install.bat` creates a local `.venv` using the pinned `requirements-student-py39.txt` profile; it does not require upgrading the system Python. Dependency installation requires internet access.
- Run `03_open_workbench.bat` on Windows, or the corresponding `.command` launcher on macOS. Keep its terminal open and use the local browser address it prints. The server is local to that computer; paths must exist on the computer running it.
- Acquisition requires Windows x64, the camera/stage USB drivers and the vendor system runtimes. Analysis may run on another computer after the source images and records are transferred. Preserve the instrument logs and collect their snapshots before moving to a computer that cannot resolve their Windows paths.
- A supervisor must confirm the scanner location, stitching geometry, sample/model applicability and inference scale. In the trial kit, a verified `trial_kit.json` selects the separate, unchanged `scanner_C1` on first launch. Existing supervisor settings are preserved. The guided scanner entry rejects a missing or mismatched communication-logging package; it does not silently launch the historical scanner.

## The seven steps

| Step | Operation and evidence |
|---|---|
| **1. Scan** | Open the verified acquisition GUI. Connect the camera and stage, confirm the live microscope view, focus manually and complete a **3 × 3** trial before a large scan. Scan parameters and operator review remain in the native acquisition window. Opening that window does not mark acquisition complete. |
| **2. Log intake** | Select one completed scan folder containing the original images, `session.json` and `events.jsonl`. Intake checks actual image dimensions, coordinates, counts, capture records and hashes before recording a verified handover. Incomplete, changed or ambiguous inputs require attention. This step saves diagnostic snapshots and references, without making another ZIP or merging samples. |
| **3. Automatic colour correction** | Open stitching setup, inspect the images and have the supervisor verify the displacement profile. Pixel dimensions or stage-step metadata alone do not establish physical calibration. Ordinary stitching enables registered-overlap RGB matching by default; students do not select a colour algorithm. |
| **4. Stitching** | Save the prepared configuration, then start automatic colour correction and stitching from the page. A registered, source-bound command runs in the background. Per-tile gains are applied during rendering without modifying raw images or solved positions. Valid correction manifests prevent applying gains twice; insufficient or conflicting evidence stops processing. |
| **5. Review** | Inspect the mosaic, coverage, seams, colour transitions and source correspondence. Completion requires bound output evidence, including actual rendered tile IDs and load-failure checks, rather than exit code zero alone. Declared artifacts are checked by hash before serving them. Human image review remains necessary. |
| **6. Layer inference** | Supply the supervisor-confirmed JSON configuration. Preparation requires explicit sample/model and geometry confirmations, checks the original **0409** checkpoint against `data/demo/assets.json`, and binds the selected mosaic and configuration by hash. The existing semantic U-Net pipeline writes a new result directory. Guided layer jobs disable external AI and automatic repair. |
| **7. Figure 4 spatial maps** | Use the saved native layer mask, valid scan support and optional exclusions. A completed single-sample layer job can populate these paths. The existing exporter generates **8 × 15** heatmaps and **6 × 11** combined compositions, whole-map/row/column statistics, PNG/SVG figures, CSV/NPZ data and a source manifest. These are analysis products; manuscript layout is a separate task. |

Changing the selected batch does not make an older batch's results current. Navigating between steps never marks an experiment complete. The original image, layer and spectroscopy review tools remain available at `/review/`.

## Model, geometry and counting definitions

The current demonstration domain is **260 nm SiO₂/Si with the unchanged 0409 checkpoint**. A different substrate, optical mode or sample requires an explicit applicability review; the workflow does not substitute the separate 70 nm/0815 configuration.

`configs/layer_workflow.teacher_template.json` is deliberately unconfirmed. Its confirmation flags cannot replace a valid `inference_geometry` record. The supervisor must reconcile the current mosaic's `layer_input_contract.json`, image dimensions and output scale. The native 256/32 tiling resampled to the 512-pixel model window in the 1080p demonstration must not be copied blindly to a scaled mosaic.

Native mask IDs are **0 = substrate, 1 = bilayer (2L), 2 = monolayer (1L), 3 = thicker layers (TL)**. Use the native integer NPY mask or an exact-colour label PNG, rather than a rendered preview or overlay. Valid-support and exclusion masks must align with its native dimensions. Without a valid-support file, the operator must explicitly confirm that the entire canvas has real scan coverage; padded pixels are not assumed valid.

For each cell, a layer fraction is its retained class area divided by **retained 1L + 2L + TL area within that cell**, after applying valid support and exclusions. Bare substrate is excluded from that denominator. These fractions describe local composition, not crystal coverage. Cells with no retained classified MoS₂ have no reported composition. The guided spatial exporter uses mask-pixel coordinates and does not invent millimetre dimensions.

## Source-bound Memory

The configured project's `Memory/` centralizes append-only workflow events, acquisition snapshots, source hashes, prepared jobs, process logs and output manifests:

```text
Memory/
  events.jsonl, index.json       Workflow index and historical-source references
  acquisition/                  Per-scan record snapshots and source hashes
  device_runs/                  Shared device-session diagnostic records
  dd_workflow/
    events.jsonl, prepared/     Job preparation and lifecycle evidence
    jobs/, stitch_runs/         Process logs, settings and stitching evidence
    analysis_outputs/           New layer and spatial-analysis results
```

Actual locations follow the index. If the configured project is a raw scan folder, Memory is placed beside it to avoid contaminating its image inventory. Original histories remain intact. Shared communication logs from one application process are linked and deduplicated rather than counted as separate device runs. Snapshots record retention limits, missing sources, truncation and changes during copying; they are not atomic hardware captures. Photos, DLLs and credentials are not copied into diagnostic snapshots. Memory is excluded from Git and student packages.

Job preparation and state queries do not execute arbitrary commands. Inputs are rechecked before execution and completion. An interrupted workbench job becomes an attention item after restart; it is not silently replayed. Use the page's complete-history and per-job log links to inspect evidence or return it to the supervisor.

## AI assistance and validation limits

Kimi is optional for stitching QC votes, duplicate selection and seam inspection when credentials are configured. Its client uses caching, finite retries and bounded voting. `--no-ai` provides the offline path; the current guided layer-analysis jobs explicitly use it. Public and trial packages contain no API key. Statistical colour matching and spatial counting do not require an LLM.

The scanner has **no automatic focusing**: automatic blur retries acquire more frames, while refocusing and retake decisions remain operator actions. Communication diagnostics add SDK returns, controller states, camera events and fault evidence; they do not replay failed relative movements or establish their physical cause.

On 31 October, the separate C1 scanner completed 9/9 and 1000/1000 images without a terminal failure in one application process. This tests normal acquisition and log writing, not the real fault-triggered stack/state path. A 1000-tile colour/render module replay retained all IDs and reduced fitted overlap log-RMS from 1.480% to 0.770%; that is a fitting residual, not independent colour accuracy or a full QC/Kimi run. Synthetic workflow tests and extracted Python 3.9 checks likewise do not establish Windows hardware acceptance. Layer predictions and area fractions require comparison with independent reference evidence before reporting scientific accuracy.

See [spatial definitions and CLI](SPATIAL_LAYER_MAPS_zh.md), [default-colour evidence](diagnostics/20261031_default_colour_correction.md), [model/scale handling](RESOLUTION_AND_INFERENCE.md), and [the Chinese operational guide](DD_WORKFLOW_zh.md).
