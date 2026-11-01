# Data, checkpoints and reproducibility

[English README](../README.md) · [Workflow](DD_WORKFLOW.md) · [Source provenance](PROVENANCE.md)

## Source code and research assets

The Git repository supplies the analysis source, configuration examples, tests and documented acquisition interfaces. Executing the layer demonstration additionally requires these trusted research assets:

```text
data/demo/assets.json
data/demo/source_images.zip
weights/model_0409_all.pth
```

`assets.json` binds both the image archive and the checkpoint to SHA-256 values and records each original image's member name, dimensions and hash. `python tools/student_demo.py check` validates the supplied assets before a model is loaded.

The checkpoint identity is `model_0409_all.pth`, SHA-256 `b771fc9cfaa8fdc1eba9d680d7c3b678890a903ddf12f2073b5a8c957b64a022`. The two-image archive SHA-256 is `bae41a3cd7e9536eb5ef4e9f80d3396bba6717df3f58ed16646e304bbe829ae5`. The manifest remains authoritative; the launcher rejects a mismatch.

The original example resources have been distributed with the laboratory's complete student packages. Inspect [GitHub release assets](https://github.com/YongXie-ICMM/MosaicAgent-DD/releases) and their manifests for the available public distribution. A release can contain older code even when its checkpoint is unchanged. Source downloads alone do not supply the model or original images. Obtain missing research assets from the maintainers rather than substituting another checkpoint.

The key-free complete trial kit prepared on 1 November 2026 contains the current analysis branch, unchanged C1 scanner delivery, original 0409 assets and one-page student PDF. Its `bundle_manifest.json` records analysis files; `scanner_C1/delivery_manifest.json` preserves the scanner inventory; `trial_kit_manifest.json` binds the whole package. This local kit has not been published as a GitHub release. It excludes credentials, experimental histories, generated Memory and Python environments. Historical internal distributions that carried a local `.env` are not public reproducibility assets.

Only load trusted checkpoints. The inherited PyTorch loader supports the original legacy checkpoint format; this is not a mechanism for accepting arbitrary model files. Legacy spectral pickle inputs have the same trusted-input boundary.

## Demonstration domain and geometry

The two examples retain their original **1920 × 1080** pixels. The operator confirmed their substrate as **260 nm SiO₂/Si**, correcting an earlier 276 nm label, and confirmed that the objective, magnification and physical field of view were unchanged from the 3840 × 2160 capture mode. These confirmations are recorded in `capture_geometry` and the substrate metadata. They support a pixel-scale mapping; they are not an independent micrometre calibration.

With that recorded geometry, the demonstration samples native 256-pixel windows with 32-pixel overlap and resamples them to the unchanged 512-pixel model window. Explicitly unconfirmed or invalid geometry stops the run. Legacy assets lacking geometry metadata retain their documented baseline. [Resolution and inference](RESOLUTION_AND_INFERENCE.md) explains the distinction between native coordinates, model-window size and physical calibration.

The images are two fields, not an independent test set. Their grid indices and full session metadata were not supplied, so they are not a verified adjacent pair for stitching. Executable inference and reviewable masks establish operation of the supplied model; accuracy and cross-batch generalization require independent reference labels.

A controlled October 2026 comparison traced the apparent-substrate-to-monolayer concern to capture colour balance: synthetic 1080p downsamples of historical 4K fields retained the expected class structure, and controlled channel gains reproduced or reduced the discrepancy. This is a diagnosis of those conditions, not a general colour-invariance claim. Diagnostic corrected predictions remain review outputs. The demo records a colour check; default scientific predictions and weights are not silently replaced by the diagnostic probe. [Controlled comparison](diagnostics/20261005_colour_balance/README.md).

`configs/reference_substrate_colour.json` is a documented image-colour reference derived from historical fields, not a physical reflectance calibration or ground-truth label. Reassess applicability after changes in camera, illumination, objective or substrate. The separate 70 nm/0815 configuration is not substituted for the 0409 domain.

## Using an experimental scan

1. Preserve the original scan/archive, native images, session metadata, program identity and failed as well as successful runs.
2. Record the sample/batch, objective, acquisition mode and calibration evidence. Image dimensions and controller counters alone do not establish physical distance.
3. Bind the stitching profile to the selected scan. Keep `layer_input_contract.json`, `profile.json` and `preflight.json` together. They record native dimensions and hash-bound settings, not proof of a completed mosaic or applicable model.
4. For layer inference, record the explicit checkpoint, current mosaic scale, inference-geometry plan, model window/overlap, valid support and exclusions. `layer_workflow.teacher_template.json` is unconfirmed and needs batch-specific review.
5. Keep masks, numerical tables, provenance manifests, process logs and review decisions with each derived run. Outputs use new directories; original data remain unchanged.

The configured project's `Memory/` keeps the workflow evidence. Shared process-level communication logs are linked/deduplicated; a copied log states its limits and non-atomic snapshot status. Diagnostic snapshots exclude image pixels, DLLs and credentials. [Memory layout and operation](DD_WORKFLOW.md#source-bound-memory).

## Statistical interpretation

Native IDs are 0 substrate, 1 bilayer, 2 monolayer and 3 thicker layers. Valid support and exclusions define the retained domain.

- **Layer composition:** class area divided by retained classified MoS₂ area, 1L+2L+TL.
- **Coverage:** class area divided by retained scan area, including substrate.
- **Aggregated fractions:** sum pixel counts before calculating a ratio; do not average cell percentages.

A cell with no retained MoS₂ has an undefined composition. Equal-sized symbols encode relative composition, not equal crystal coverage. TL remains the trained combined thicker-layer class. Spatial plots use mask coordinates until a physical calibration is explicitly supplied.

The spatial exporter preserves the source labels. Consistent pixel counts across export grids validate computation, not the accuracy of segmentation labels.

## Manuscript-level reproducibility

The two-image demonstration does not reproduce every manuscript figure. Full reproduction additionally requires the complete source mosaics, native segmentation/support/exclusion masks, calibrated spectral references, sample/measurement correspondence, device dimensions and experimental decision records. Their archive links and access conditions belong in the manuscript's data-availability statement. This repository does not claim those complete datasets are included in Git.

Code publication, a generated student package, normal-path instrument trials, synthetic tests and independent scientific validation are reported separately. See [validation in the README](../README.md#validation-and-scientific-scope) and the [incident history](INCIDENT_HISTORY.md).
