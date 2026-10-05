# 2026-10-05 controlled comparison: colour balance, not scale

[Main entry](../../../README.md) · [Resolution and inference](../../RESOLUTION_AND_INFERENCE.md) · [Handoff](../../CLAUDE_HANDOFF.md) · machine-readable numbers: [summary.json](summary.json)

**Question.** Both inference modes (native 512/64 and adapted 256/32 → 512) labelled broad apparent bare substrate in the two current 1920 × 1080 images as monolayer (bare substrate 2.3 % / 3.3 % of the frame). Was the cause the pixel-count change, or something else?

**Fixed throughout.** Checkpoint `model_0409_all.pth` (`b771fc9c…`), architecture, preprocessing (RGB/255, ImageNet mean/std) and the shipped `stage_segment` logic, reproduced bit-exactly outside the repository (agreement 1.0 with the shipped masks of runs `20261005T211325` and `20261005T210020`). No weights, thresholds or default preprocessing were changed. The historical masks used below are predictions of the same checkpoint on the 2026-09-24 inference examples; they are the mode on which the model was previously reviewed, not labels.

## Results

| Experiment | What varied | Result |
|---|---|---|
| A. Shipped runs | nothing | bare 2.3 % / 3.3 %, 1L 76 % / 82 % in both modes |
| B. Synthetic 1080p downsample (BOX) of historical 4K frames L18, L34 | pixel count only | substrate recall 99.5 % / 99.3 % (adapted), 99.6 % / 99.2 % (native); agreement with the historical mask 99.2 % / 98.7 % (adapted) |
| Colour statistics | — | historical substrate median RGB (221,171,170); new background (217,156,181) and (223,161,189): blue +12…+19, green −13…−19; new triangles (186,114,188) sit at hue ≈ 298°, closer to historical 2L (≈ 291°) than to historical 1L (≈ 322°) |
| C. Per-channel gain on the new images, (1.02, 1.10, 0.94) and (0.98, 1.05, 0.90), nothing else | colour balance only | bare 70 % / 78 %, 1L 25 % / 16 %, 2L 1–3 %; triangles → 1L, interiors → 2L, bright nuclei → TL |
| D. Inverse gain (0.98, 0.89, 1.07) on the synthetic 1080p historical L18 | colour balance only | 99.3 % of historical-substrate pixels → 1L, 79.7 % of historical-1L pixels → 2L: the failure is reproduced on a frame the model handled |

Reproduce C inside the repository with the real bundle: `python tools/student_demo.py colour-probe --probe-inference` (run `20261005T215006_4243dbe5`, kind `diagnostic_colour_probe`, `selected_for_measurement: false`; gains estimated by `color_diagnostics.background_colour`, bare 70.4 % / 77.5 %). B and D need the laboratory 4K frames, which are not in this repository.

## Conclusion and what is not established

The pixel-count change is handled by the recorded geometry plan. The acquisition colour balance of the new capture mode differs from the mode the checkpoint was reviewed on, that difference alone reproduces the failure, and removing it restores the expected class structure. The network reads the whole new image "one layer thicker".

Not established: classification accuracy on the new images (no independent reference labels); physical pixel calibration; that a three-gain correction removes all spectral differences between capture modes; the cause of the colour change (camera driver path — the 954-point session recorded an OpenCV/DirectShow backend with white-balance and exposure read-back unavailable — white-balance setting, or illumination).

## What changed in the software

- `flakepipeline/color_diagnostics.py`: measures the bare-substrate colour (brightest extended flat luminance plateau) and compares it with `configs/reference_substrate_colour.json`; reports a per-channel gain and a verdict. Pixels and labels are never edited.
- `02_run_layer_demo` records that check in every run manifest (`colour_check`), prints a bilingual warning and forces review when the balance differs. The default inference is unchanged.
- `python tools/student_demo.py colour-probe [--probe-inference]`: measurement, and optionally the diagnostic gain-corrected run, which never becomes `latest.json`.

## Order of remedies (agreed 2026-10-05)

1. Correct the colour balance at acquisition: with auto white balance and auto exposure off, adjust until a clean bare-substrate area reads ≈ (219, 171, 170) ± 5 %, record the settings, and re-acquire. The colour check tells the student immediately whether the balance is back.
2. If existing images must be used, apply an explicit, recorded gain estimated from a human-designated bare-substrate region; the result stays marked for review.
3. Only later, and as a separate evidenced change: fine-tune with colour augmentation using a few labelled new-camera images. Never change the default preprocessing silently.
