# Resolution changes and layer inference

[Main entry](../README.md) · [Data and reproducibility](DATA_AND_REPRODUCIBILITY.md) · [中文操作入口](../README_zh.md)

## What the current examples establish

The bundle contains two original **1920 × 1080** camera images. The user confirmed **260 nm SiO₂/Si** on 2026-10-05; their historical label was “276 nm”. They were not made by resizing older 3840 × 2160 images. The supplied `model_0409_all.pth` and network are unchanged. The separate 70 nm configuration is not used for these examples.

The user subsequently confirmed that **only the camera resolution changed from 3840 × 2160 to 1920 × 1080; the physical field of view, objective and magnification remained unchanged**. The asset manifest records `capture_geometry.same_physical_fov_confirmed: true` and the confirmation source. This operator confirmation supports a pixel-scale mapping. It does not supply an independent micrometre calibration or validate the model on these images.

Both direct inference and the optional scale comparison have run. The scale comparison did **not** resolve the visual concern that broad apparent-substrate areas are labelled monolayer. This observation has no independent reference labels or quantified accuracy assessment. Neither result has been accepted for scientific measurement, and the cause remains unresolved. A runnable checkpoint, a changed tile size and a visually plausible map are each insufficient to establish accuracy.

Students keep using the same three launchers: **01_install → 02_run_layer_demo → 03_open_workbench**. For the current bundle, the demo reads the recorded confirmation and actual image dimensions, then automatically crops native **256 × 256** tiles with **32-pixel overlap** and resamples each to the unchanged **512 × 512** model window. It records the geometry with the result. Only legacy assets without a `capture_geometry` field retain native **512 × 512 / 64-pixel overlap**. If that field is present but unconfirmed or invalid, the run stops with an error and requires correction. The explicit comparison below retains that native baseline for diagnosis.

## Colour balance is a separate axis from scale

The 2026-10-05 controlled comparison ([record](diagnostics/20261005_colour_balance/README.md)) separated the two. With the unchanged checkpoint and pipeline, a synthetic 1920 × 1080 downsample of a historical 3840 × 2160 frame was labelled like the original (substrate recall ≥ 99 % in both modes), so the geometry plan below handles the pixel-count change. The current images differ from the reviewed capture mode in colour balance instead: their bare substrate reads about (217–223, 156–162, 181–189) against the reference (219, 171, 170), and the network reads every class one level thicker. A per-channel gain of about (1.0, 1.1, 0.9) restored the expected structure; the inverse gain applied to the historical frame reproduced the failure.

Every demo run therefore records a `colour_check` against `configs/reference_substrate_colour.json` (bare-substrate colour of the reviewed capture mode, derived from the same checkpoint's predictions on two historical frames; not a reflectance calibration). A deviation above the recorded tolerance (5 %) prints a bilingual warning and forces review. `python tools/student_demo.py colour-probe` reports the per-image gain; with `--probe-inference` it runs the unchanged model on gain-corrected temporary copies as a diagnostic run (`kind: diagnostic_colour_probe`, `selected_for_measurement: false`, `latest.json` untouched). Correct the balance at the camera first; a software gain is a probe, not calibration, and corrected predictions remain unvalidated without independent reference regions.

## When a twofold scale conversion is justified

Pixel dimensions alone do not establish physical field of view. A change from 3840 × 2160 to 1920 × 1080 might retain the field of view or involve a crop or another acquisition mode. Confirm the actual field using instrument settings and a physical reference before treating it as unchanged. File size in MB does not determine this relationship or the stage step.

**For the operator-confirmed current pair of capture modes**, the earlier mode has twice as many pixels along each axis and four times as many pixels per physical area. A fixed object would therefore have approximately half the native pixel width and one quarter of the native pixel area in the new image. The geometry plan uses:

| Quantity | Native comparison baseline | Current bundle after recorded confirmation |
|---|---:|---:|
| Native image | 1920 × 1080 | 1920 × 1080 |
| Native crop | 512 × 512 | 256 × 256 |
| Native overlap | 64 pixels | 32 pixels |
| Model input window | 512 × 512 | 512 × 512 after crop resampling |
| Checkpoint | Same 0409 | Same 0409 |

The plan records the corresponding native linear and area conversion factors, **1/2** and **1/4**, relative to the earlier mode. These factors are metadata for reviewing pixel-based thresholds; they do not certify physical calibration or automatically validate all downstream measurements. Physical stage motion is unchanged.

The 3840 × 2160 reference is an earlier **image capture mode**, not a training crop. Training crop size and prediction window size are separate settings. Resampling a crop cannot recover optical detail missing from the acquired image or correct an unverified change in colour, focus or substrate response.

## Optional comparison for diagnosis

Run from the repository root in the installed environment:

```sh
python tools/student_demo.py compare-scale
```

This explicitly runs native **512/64** and scale-adjusted **256/32 → 512** modes as two diagnostic results, even though the current bundle defaults to the latter after the recorded confirmation. Both comparison runs remain for review; `outputs/demo/scale_comparison.json` records their paths and geometry with `comparison_only_requires_review` and `selected_for_measurement: false`. Each run gets a new directory. Review their run identities rather than treating the latest displayed result as an accepted measurement.

The first recorded comparison preceded the operator confirmation and was labelled an assumed-same-field-of-view hypothesis. The current two-image bundle now has that confirmation. When an asset has no confirmation, a hypothetical comparison is still only an assumption, not permission to adopt it as the default or a measurement.

The command calls neither the camera nor an external model. Use it to test a preprocessing hypothesis, not as calibration or evidence that the scale-adjusted result is more accurate.

## Carry acquisition geometry into a new analysis

In **02 / Images and stitching**, inspect the actual raw scan folder and prepare the stitching settings. The existing saved-file panel includes:

- `profile.json`: the selected stitching parameters.
- `preflight.json`: input inspection, source inventory and resolved settings.
- `layer_input_contract.json`: native tile dimensions from image headers, source fingerprints, SHA-256 bindings to those two files, acquisition metadata and its verification status.

Keep these files together. The contract is a metadata handoff: it starts at `needs_model_and_scale_check`; it selects no model, executes no inference and establishes neither stitching completion nor physical calibration. Its native dimensions describe individual raw tiles, not the full mosaic dimensions.

For an existing full-resolution mosaic, first set the correct input image and trusted checkpoint in `configs/layer_analysis.example.json`. If the actual physical field of view has been confirmed to match the reference capture mode and the basis of that confirmation recorded, preview the plan using:

```sh
python flakepipeline/run.py --config configs/layer_analysis.example.json \
  --layer-input-contract /path/to/stitch_bundle/layer_input_contract.json \
  --reference-capture-size 3840 2160 \
  --same-fov-confirmed --no-ai --no-repair --plan
```

Replace the contract path with the saved file. The reference dimensions must describe the verified previous capture mode. **Supply `--same-fov-confirmed` only after actual field-of-view confirmation**, not merely because the pixel ratio is two. Once the plan and mosaic provenance have been reviewed, remove `--plan` to run the analysis.

The consumer checks file hashes and the consistency of dimensions, profiles and source fingerprints before using the geometry. Unknown field of view, inconsistent bindings or unsupported geometry block adaptation. The current handoff supports only **`out_scale == 1`**; a reduced mosaic (`out_scale != 1`) is blocked with `mosaic_output_scale_needs_explicit_mapping`. An explicit mapping from rendered mosaic pixels to the source geometry is needed before supporting that case. Do not relabel a reduced image as full resolution to bypass the check.

These checks connect recorded acquisition geometry to explicit inference settings. They do not automatically select a suitable checkpoint, retrain the model, verify the microscope calibration or measure prediction accuracy.
