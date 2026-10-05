# MosaicAgent-DD

A publication-scoped workflow for optical microscope acquisition, mosaic construction, semantic layer-number segmentation, and optical spectroscopy records in the Digital Discovery materials study.

[中文操作入口](README_zh.md) · [Architecture](docs/ARCHITECTURE.md) · [Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) · [Acquisition](acquisition/README.md) · [Spectroscopy](docs/SPECTRA.md)

## Start here

**[Download the complete student bundle](https://github.com/YongXie-ICMM/MosaicAgent-DD/releases/latest)** (code + two original 1920 × 1080 images + trained weights + numbered launchers). Choose `MosaicAgent-DD-student-complete.zip` from the release assets, rather than GitHub’s source-code ZIP.


**Students receiving the complete trial bundle:** extract the entire folder, then use these numbered launchers:

1. `01_install.bat` (Windows) or `01_install.command` (macOS): create a local Python environment and install the analysis dependencies. Python 3.10+ and internet access for dependency installation are required.
2. `02_run_layer_demo.bat` / `.command`: verify the supplied image and checkpoint hashes, run real U-Net inference on the supplied images, and save a new layer-number result directory.
3. `03_open_workbench.bat` / `.command`: open the local analysis entry at `http://127.0.0.1:8792/`.

The complete trial bundle includes the example image archive, trained checkpoint and asset manifest. A Git clone contains code, configuration examples and tests; it is not a substitute for those research assets. See [the data guide](docs/DATA_AND_REPRODUCIBILITY.md). No camera, stage or external model is called by the layer demo. Predicted labels are not independent physical reference measurements.

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

The workbench reviews saved products and prepares tool handoffs. New inference is run through the supplied demo or CLI. Instance annotation, touching-domain separation and crystal orientation tools are outside this submission release. Existing research data and development tools are preserved separately.

> **Current demonstration status:** the supplied historical 0409 checkpoint runs on the two current 1920 × 1080 images, but visual inspection found extensive apparent-substrate misclassification. The bundle is a software/diagnostic preview, not an accepted layer measurement. Verify the substrate, image acquisition and applicable checkpoint before using any fraction in a scientific conclusion.

## Run on your own data

Run commands from the repository root. Start with the input check in **02 → Scan and stitching setup**, and retain the generated command and profile. Image file size in MB does not determine stitching geometry. For a new camera mode, verify actual pixels, overlap and physical calibration.

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

## Optional model supervision

Kimi supervision exists in the stitching, image-audit and spectral tools; it is not needed to launch the workbench or run the offline demo. Read [the architecture](docs/ARCHITECTURE.md) before enabling it. Configure credentials explicitly with environment variables or a local `.env` based on `.env.example`. Credentials and local paths are not committed. Enabling an external model may send the selected images or spectral plots to that service; `--no-ai` disables these requests.

## Validation and availability

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary records, synthetic inputs and mocked devices. Instrument validation remains separate; the exported scanner adds mode checks to a previously tested acquisition base and still needs acceptance on the actual microscope. The complete student bundle is also checked using the real supplied checkpoint and example images. That is a reproducibility check, not a new accuracy evaluation.

Source provenance, modifications for public portability, and third-party dependency boundaries are documented in [PROVENANCE.md](docs/PROVENANCE.md). No repository-wide license grant has been added to this initial public snapshot; third-party dependencies and vendor SDKs retain their own terms.
