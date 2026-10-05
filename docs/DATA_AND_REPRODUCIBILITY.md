# Data, checkpoints and reproducibility

[Main entry](../README.md)

The Git repository contains the submission-scoped software. The complete student trial distribution adds the original example images and the actual checkpoint used for inference. Download `MosaicAgent-DD-student-complete.zip` from the [release assets](https://github.com/YongXie-ICMM/MosaicAgent-DD/releases/latest). The package contains:

```text
MosaicAgent-DD/
  data/demo/source_images.zip
  data/demo/assets.json
  weights/model_0409_all.pth
  01_install.bat / .command
  02_run_layer_demo.bat / .command
  03_open_workbench.bat / .command
  00_STUDENT_GUIDE_zh.pdf              step-by-step student manual (copy of docs/STUDENT_GUIDE_zh.pdf)
  bundle_manifest.json                 every packed file with its SHA-256, source commit, build time
  acquisition/Auto_Scan/               unchanged scanner runtime + 00_white_balance_then_scan.bat, wb_calibrate.py
  docs/STUDENT_GUIDE_zh.md / .pdf      the manual's Markdown source and its PDF (tools/build_student_guide_pdf.py)
  (all other tracked sources: code, configs, docs, tests)
```

The manifest binds each source image and the checkpoint to SHA-256 values. The launcher verifies the assets before loading the model. The bundle is built with `python tools/build_student_bundle.py` from a checkout that holds the assets: it refuses to pack when an asset is missing or does not match `data/demo/assets.json`, packs everything tracked by Git (outputs, environments and caches excluded), writes `bundle_manifest.json` into the archive and beside it, and re-reads the archive to verify every member; `--check <zip>` repeats that verification on a downloaded bundle. Two variants exist: the public `MosaicAgent-DD-student-complete.zip` (no credentials; stitching falls back to statistics with `--no-ai`) and, built with `--with-env <path to the group's .env>`, the internal `MosaicAgent-DD-student-complete-with-kimi.zip` whose manifest records `credentials_included: true`. The owner decided on 2026-10-05 that the group's students stitch with Kimi exactly as the original pipeline did, so the internal variant is what students receive; it is never attached to a GitHub release. Credential files are excluded from enumeration and only enter an archive through `--with-env`. Model files are trusted research inputs: the inherited PyTorch loader supports legacy checkpoints and must not be used with untrusted downloaded files. Legacy spectral pickle files have the same trusted-input restriction.

The example images retain their original bytes and pixels; they have not been resized from older 3840 × 2160 records. Their scan-grid indices and session metadata were not supplied, so they are not an established adjacent-pair stitching test. The checkpoint is the unchanged 0409 model previously checked for executable inference. The native and scale-adjusted trials both ran. Their comparison did not resolve the visual concern that broad apparent-substrate areas are labelled monolayer; there are no independent reference labels or quantified accuracy results for these two images. These outputs are not accepted measurements. The user confirmed these samples are on **260 nm SiO₂/Si**, historically labelled “276 nm”. This correction is specific to the supplied examples, not a blanket rewrite of spectroscopy reference calibrations. The 0815 checkpoint belongs to a 70 nm sample configuration and is not used in the bundle. Review model training conditions, preprocessing and acquisition colour settings; the visual concern has not been attributed to any single cause. The user subsequently confirmed on 2026-10-05 that only the camera resolution changed from 3840 × 2160 to 1920 × 1080, while the objective, magnification and physical field of view remained unchanged. The asset manifest records this confirmation under `capture_geometry`, including its source. This operator confirmation supports the pixel-scale mapping but is not an independent micrometre calibration. The current bundle automatically uses native 256/32 tiles/overlap resampled to the unchanged 512-pixel model window; only legacy assets without a `capture_geometry` field retain the native 512/64 baseline. If that field is present but unconfirmed or invalid, the run stops with an error rather than silently falling back. The earlier diagnostic comparison used an assumed-field-of-view hypothesis at the time; the subsequent confirmation does not turn those predictions into validated measurements. See [Resolution and inference](RESOLUTION_AND_INFERENCE.md) for the recorded geometry and limitations.

`configs/reference_substrate_colour.json` records the bare-substrate colour of the capture mode on which the checkpoint was previously reviewed (derived from its own predictions on two historical frames, with their hashes). The demo compares each image against it and records the result; it is a colour-balance reference, not a physical calibration or a label. Re-derive it when the camera, illumination, objective or oxide thickness changes. The 2026-10-05 comparison that motivated it is in [docs/diagnostics/20261005_colour_balance](diagnostics/20261005_colour_balance/README.md).

The trial uses two current 1920 × 1080 optical microscope fields, not an independent test set. Its purpose is to show that the supplied checkpoint and current code execute and produce reviewable layer masks. It does not establish classification accuracy, cross-batch generalization, or an agent benefit. Original data and historical predictions are retained; each run creates a new output directory.

All original experimental archives, full scan grids, historical annotations and previously used weights remain in the laboratory project. They are not replaced by the small trial bundle. Local saved paths remain usable. The portable student package resolves its own paths so that a student does not need the researcher's username or home folder.

## Using new experimental data

1. Preserve the source archive or scan folder and session metadata.
2. Record the sample/batch ID and the acquisition settings. Confirm real physical calibration separately from image dimensions.
3. Bind stitching settings to the actual raw grid using the workbench setup tool. Keep `layer_input_contract.json`, `profile.json` and `preflight.json` together; they record native tile dimensions and hash-bound handoff metadata, not proof of a completed mosaic or applicable model.
4. Use an explicit checkpoint, inference tile size, overlap, filler/support convention and counting denominator. Training crop size and inference tile size are distinct settings.
5. Save predictions, logs, configuration and review decisions together. State any exclusion region and its numerator/denominator effect.

The image-review adapter accepts a saved `results/run_manifest.json` and case-specific `mask_color.png`; see [the workbench schema](../tools/analysis_workbench/README.md). The trial launcher writes that format. It does not automatically reproduce all manuscript figures or historical region exclusions.

Complete manuscript reproducibility additionally needs the full source datasets, calibrated spectroscopy references, measurement correspondence, device/growth records and any historical decision files. Their availability must be described separately in the submitted data-availability statement. This repository does not invent an archival DOI or claim those records are publicly downloadable.
