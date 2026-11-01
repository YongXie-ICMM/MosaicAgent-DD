# MosaicAgent-DD

A reproducible software workflow for large-area optical microscopy of CVD-grown MoS₂: acquisition records, image stitching, semantic layer-number mapping, spatial layer statistics and optical spectroscopy records. This repository accompanies the Digital Discovery study.

**[English documentation](#documentation) · [中文说明](README_zh.md)**

## Workflow

**Acquire → verify images and logs → match colour → stitch → inspect → identify layers → map spatial composition**

| Stage | Implementation and output |
|---|---|
| Acquisition | Windows camera/XY-stage scanner with timestamped images, session metadata, device communication and per-operation records. The current communication-logging scanner is maintained in [AmScope-Camera](https://github.com/YongXie-ICMM/AmScope-Camera). |
| Intake | Validate one scan's image inventory, recorded hashes and acquisition events; retain correspondence to the source session. |
| Colour matching and stitching | Register neighbouring overlaps, estimate illumination and per-tile RGB gains, and apply correction during rendering. Colour matching is enabled by default. Verified recorded grids preserve acquisition row/column identities when texture is ambiguous. |
| Result inspection | Check actual contributing images, missing positions, registration and colour records. A successful process exit alone does not validate a mosaic. |
| Layer inference | Apply the explicitly selected 0409 U-Net checkpoint with recorded capture geometry. Retain native semantic masks, valid support, exclusions and review decisions. |
| Spatial statistics | Export per-layer heatmaps, combined layer-composition maps and numerical tables from existing masks. No additional model call is needed. |
| Provenance | Store source-bound job plans, progress, errors, artifact hashes and review history under one project `Memory/` directory. |

The guided browser interface opens this seven-stage workflow. The earlier review tools remain available at `/review/`. See the [English workflow guide](docs/DD_WORKFLOW.md).

## Reproduce the analysis

### Environment

The unified student trial supports the instrument's existing **64-bit Python 3.9.18**. It does not require upgrading the instrument's Python. The pinned profile retains NumPy 1.26.2 and OpenCV 4.10.0.84 from the successful instrument runs; dependencies are installed in a local environment.

```sh
python -m venv .venv
# Activate this environment for your platform, then:
python -m pip install -r requirements-student-py39.txt
```

Windows users receiving the complete trial kit can run `01_install.bat`, followed by `03_open_workbench.bat`. The macOS equivalents end in `.command`; acquisition itself requires Windows x64 and the appropriate vendor drivers. Camera and stage SDKs are not installed by pip. A Git clone contains source, not the checkpoint or a configured instrument.

### Example layer inference

The complete example distribution supplies two original 1920×1080 images, `data/demo/assets.json` and the unchanged `weights/model_0409_all.pth`. With these trusted assets placed at their manifest-defined paths:

```sh
python tools/student_demo.py check
python tools/student_demo.py run
python tools/student_demo.py workbench
```

The two images are an executable inference demonstration, not an independent accuracy test or a verified adjacent pair for stitching. Each run creates new outputs. Read [data, checkpoint and reproducibility requirements](docs/DATA_AND_REPRODUCIBILITY.md) before interpreting predictions.

### Stitch an experimental scan

Prepare a profile using the actual image dimensions and verified acquisition geometry. Preserve the generated `profile.json`, `preflight.json` and `layer_input_contract.json`. For an offline run:

```sh
python run_stitch.py --data /path/to/scan --work /path/to/new_work \
  --out /path/to/new_mosaic.png --stitch-profile /path/to/profile.json --no-ai
```

The guided interface prepares the same fixed command and records its execution. Default colour correction uses neighbouring registered overlaps; insufficient or inconsistent evidence stops the run. The explicit historical control is `--colour-correction off` with a separate work directory. Original images are never overwritten. [Correction evidence](docs/diagnostics/20261031_default_colour_correction.md).

For layer analysis of a new mosaic, first confirm the sample/checkpoint domain and inference geometry. The supplied teacher template is deliberately unconfirmed; its booleans and example paths are not calibration. Keep the model's 512-pixel window distinct from the native capture/mosaic scale. [Resolution and inference](docs/RESOLUTION_AND_INFERENCE.md).

### Export spatial layer maps

```sh
python tools/layer_spatial_maps.py --mask /path/to/native_mask.npy \
  --valid-mask /path/to/valid_support.npy --sample sample_id \
  --output /path/to/new_spatial_output
```

Supply `--exclusions` when exclusions apply. Outputs include PNG, editable SVG, CSV, NPZ and a source-bound JSON manifest. Counts are computed from native labels and retained support; they are not inferred from a coloured preview.

| Native label | Meaning | Display RGB |
|---|---|---|
| 0 | Substrate/background | 0,0,0 |
| 1 | Bilayer, 2L | 128,0,0 |
| 2 | Monolayer, 1L | 0,128,0 |
| 3 | Thicker-layer class, TL | 128,128,0 |

**Layer composition** divides each layer's retained area by retained classified MoS₂ area, 1L+2L+TL. **Coverage** divides by retained scan area, including substrate. Whole-map and row/column fractions are calculated from summed pixel counts, rather than means of cell percentages. TL is a combined thicker-layer class, not an isolated trilayer measurement. Physical coordinates require an independent spatial calibration.

## Optional AI supervision

Kimi supervision is available in selected stitching, image-review and spectroscopy tools. Deterministic extraction, registration, layer inference and spatial counting remain explicit Python operations; a model response is not an experimental measurement. The offline commands above do not require an API key. When external supervision is enabled, selected images or plots may be transmitted to the configured provider; record the invocation and decision scope. Credentials and local environment files are not included in Git or the key-free trial kit.

## Validation and scientific scope

- **Software:**1,014 offline tests passed and 1 skipped under actual Python 3.9.18 using the pinned dependencies. Python compilation, JavaScript syntax and registration self-checks passed. CI checks both Python 3.9 and 3.12; its live result is visible in [GitHub Actions](https://github.com/YongXie-ICMM/MosaicAgent-DD/actions).
- **Packaging:** the 270-file local trial kit passed nested source, scanner and checkpoint hash checks. A synthetic background spatial job run from the extracted kit produced 16 verified artifacts and a Memory completion event. Synthetic checks are identified as such.
- **Stitching:** recorded-grid protection corrected 16 misplaced rows in the 864-image low-texture case while retaining every image. The 1000-image colour/render replay retained all image IDs and reduced overlap log-RMS from 1.480% to 0.770%. These are geometry/consistency checks, not segmentation accuracy estimates.
- **Acquisition:** the C1 scanner completed 9-image and 1000-image scans on 31 October 2026 in one application process. No motion fault occurred. This validates those normal runs and logging, not elimination of the earlier SDK `command_movr`−1 fault.
- **Remaining acceptance:** a complete run through the new integrated interface on the Windows instrument remains pending. Independent reference labels are still needed for classification accuracy on the new capture conditions. Colour matching does not repair a physically incorrect scan or establish checkpoint applicability.

Historical data, checkpoints, scanner snapshots and decision records remain preserved. The DD repository's 16 hash-locked scanner runtime files are unchanged; the complete kit binds the separately verified C1 scanner. Neither acquisition nor blur retries imply automatic Z-axis focusing. The [English incident history](docs/INCIDENT_HISTORY.md) distinguishes confirmed application fixes, recovered read errors, diagnostic improvements and unresolved causes.

## Code, data and release provenance

This source branch includes the current guided workflow. GitHub's latest release asset may have been built from an older commit; inspect its manifest rather than assuming it matches the current branch. The new local key-free student kit has not been published as a GitHub release. Source downloads do not include the original checkpoint, full manuscript datasets or a vendor-driver installation.

[Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) describes the required assets and boundaries. [Source provenance](docs/PROVENANCE.md) records portability changes and third-party components. This repository does not assign a new license to third-party SDKs or claim that all manuscript data are present. No archival DOI is invented here; the manuscript's data-availability statement identifies the corresponding data archive.

## Documentation

- [Guided workflow and Memory — English](docs/DD_WORKFLOW.md)
- [Incident-to-code history — English](docs/INCIDENT_HISTORY.md)
- [Data, checkpoints and reproducibility — English](docs/DATA_AND_REPRODUCIBILITY.md)
- [Capture scale and inference — English](docs/RESOLUTION_AND_INFERENCE.md)
- [Architecture — English](docs/ARCHITECTURE.md)
- [Spectroscopy records — English](docs/SPECTRA.md)
- [Source and dependency provenance — English](docs/PROVENANCE.md)
- [Student instructions — Chinese](README_zh.md)
- [Spatial-map commands and counting definitions — Chinese](docs/SPATIAL_LAYER_MAPS_zh.md)
- [Developer rules](AGENTS.md)
