# Data, checkpoints and reproducibility

[Main entry](../README.md)

The Git repository contains the submission-scoped software. The [GitHub release](https://github.com/YongXie-ICMM/MosaicAgent-DD/releases/latest) provides `MosaicAgent-DD-student-complete.zip` with code and assets together. The student trial distribution adds the original example images and the actual checkpoint used for inference:

```text
MosaicAgent-DD/
  data/demo/source_images.zip
  data/demo/assets.json
  weights/model_0409_all.pth
  01_install.bat / .command
  02_run_layer_demo.bat / .command
  03_open_workbench.bat / .command
```

The manifest binds each source image and the checkpoint to SHA-256 values. The launcher verifies the assets before loading the model. Model files are trusted research inputs: the inherited PyTorch loader supports legacy checkpoints and must not be used with untrusted downloaded files. Legacy spectral pickle files have the same trusted-input restriction.

The example images retain their original bytes and pixels; they have not been resized from older 3840 × 2160 records. Their scan-grid indices and session metadata were not supplied, so they are not an established adjacent-pair stitching test. The checkpoint is the previously validated 0409 model. In this actual trial it labels extensive apparent substrate as monolayer, so its outputs on the current images are not accepted measurements. A separately inspected 0815 model also shows visible disagreement and tile artefacts; it has not been substituted based on appearance. Confirm sample/substrate and preprocessing evidence before selecting a replacement.

The trial uses two current 1920 × 1080 optical microscope fields, not an independent test set. Its purpose is to show that the supplied checkpoint and current code execute and produce reviewable layer masks. It does not establish classification accuracy, cross-batch generalization, or an agent benefit. Original data and historical predictions are retained; each run creates a new output directory.

All original experimental archives, full scan grids, historical annotations and previously used weights remain in the laboratory project. They are not replaced by the small trial bundle. Local saved paths remain usable. The portable student package resolves its own paths so that a student does not need the researcher's username or home folder.

## Using new experimental data

1. Preserve the source archive or scan folder and session metadata.
2. Record the sample/batch ID and the acquisition settings. Confirm real physical calibration separately from image dimensions.
3. Bind stitching settings to the actual raw grid using the workbench setup tool.
4. Use an explicit checkpoint, inference tile size, overlap, filler/support convention and counting denominator. Training crop size and inference tile size are distinct settings.
5. Save predictions, logs, configuration and review decisions together. State any exclusion region and its numerator/denominator effect.

The image-review adapter accepts a saved `results/run_manifest.json` and case-specific `mask_color.png`; see [the workbench schema](../tools/analysis_workbench/README.md). The trial launcher writes that format. It does not automatically reproduce all manuscript figures or historical region exclusions.

Complete manuscript reproducibility additionally needs the full source datasets, calibrated spectroscopy references, measurement correspondence, device/growth records and any historical decision files. Their availability must be described separately in the submitted data-availability statement. This repository does not invent an archival DOI or claim those records are publicly downloadable.
