# Preserve recorded scanner rows in low-texture mosaics

## Trigger and evidence

The user reported two missing images in the second visible column of the
`261008PM_5mg_70nm` mosaic. The two downloaded ZIP parts belong to one completed
session: 18 columns × 48 rows, 864 distinct 1920 × 1080 images. The images at
the reported positions exist and their SHA-256 hashes agree with the capture
journal. This new sample is on 70 nm substrate; this investigation only runs
stitching and does not select or replace a layer-number model.

An offline reproduction with the same source pixels, measured adjacent-image
geometry, cache divisor 4 and output scale 1/8 reproduced the top-left notch.
The scanner advances columns to the left (`invert_x=True`); source column c16
is the second visible column from the left. All 864 images were actually
rendered. The defect was placement, rather than absent acquisition or final
QC deletion:

1. The inferred column chain selected a reverse step at c16 `r15 → r16`.
2. Lateral voting selected offset +2 for that column's single 48-image segment
   (five qualifying edges, mean NCC 0.7103).
3. Rows r0–r15 were placed at r2–r17, overlapping images already in those rows
   and leaving two row pitches uncovered at the top. The graph remained
   connected and its small residual did not reveal the row identity error.

## Change

`run_stitch.py --grid-policy auto` now applies `preserve` to the completely
validated `mosaic_r<row>_c<col>` flat grid. Other acquisition-order layouts keep
the historical inference mode. `--grid-policy legacy` explicitly restores both
old row inference and the 55% coverage rescue; `preserve` is rejected for a
layout that has not been validated as a complete scanner grid. Choose a new
`--work` directory when switching policy; the binding rejects mixed caches.

In `register.py`, `row_policy="recorded-grid"` preserves each integer,
per-column-unique `nominal_row`. Vertical registration only tests its signed
recorded row difference. A gap without overlap, insufficient NCC or a peak
outside the prior window becomes a weak prior edge at the recorded spacing.
Horizontal comparisons use equal recorded rows; correlations refine positions
without voting images into different row identities. Negative horizontal
progression is supported, and its reported overlap uses displacement magnitude.
The default `build_edges(..., row_policy="infer")` retains historical serpentine
and reshoot reconstruction.

Grid protection also retains every selected grid image nominated for dropping
by QC, while preserving the QC reasons. It does not recalibrate the focus gate.
The renderer records contributed, failed-load and missing-position tile IDs.
A protected grid cannot be reported successful if any selected image is absent
or fails to load in any band, including a tile that contributed to an earlier
band. A rerun clears previous completion records before work starts.

## Controlled result

| Measure | Baseline | Protected grid |
|---|---:|---:|
| Actually rendered | 864 | 864 |
| Wrong row identities | 16 | 0 |
| Final dropped / failed loads | 0 / 0 | 0 / 0 |
| QC defocus / rescued | 389 / 389 | 389 / 389 |
| Registration RMS, full-resolution px | 2.1214 | 1.2757 |
| Affine-grid deviation median / p95 / max, px | 55.4 / 156.3 / 1801.8 | 17.6 / 30.7 / 43.7 |
| Output dimensions at 1/8 | 4171 × 5802 | 4166 × 5802 |

The top notch disappears in the corrected mosaic. The explicit legacy control
produces byte-identical pixels and positions to the old baseline (PNG SHA-256
`39dbe5d0850baac9f0d78a9e81bf2c16f0a6d91bd1482e8692002b902af6291c`).
The corrected PNG SHA-256 is
`60745e658ec8f66deea792346ab72793f711a23f30bc9b4b965a0d01b851a608`.
Geometry and pixel inputs are held constant. These are computational placement
and coverage checks, not independently measured physical accuracy.

The scanner still reports `spatial_coverage_verified=False`. An entirely
unregisterable acquisition still requires measured geometry; no silent nominal
fallback was added. Historical Figure 3a synthetic checks keep all 41 row
identities correct with position error p95 0.60 px.

## Reproduction and follow-up

Keep both original ZIPs and their journal unchanged. Assemble their image union
into a new flat directory without renumbering or editing pixels. Measure an
adjacent-pair profile for the actual dimensions. Run the same command twice
with separate work/output paths, once with `--grid-policy legacy`, once with
`--grid-policy preserve`; use `--no-ai --scale-div 4 --out-scale 0.125` offline.
Compare `positions_tids.json`, `register.stats.row_abs`, actual renderer IDs and
the top-left crop, rather than trusting selected-image counts or RMS alone.

The local analysis record contains source manifests, the measured profile,
three controlled states/logs, a comparison JSON and the before/after crop.
Synthetic regression tests cover false zero/reverse steps, missing-row spans,
descending order, negative horizontal progression, QC coverage, cache binding,
missing loads and partial-band failures. Focus-score content bias remains a
separate classification problem; grid coverage protection prevents it from
silently deleting a unique scan position.
