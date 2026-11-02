# Review exclusion boxes before spatial statistics

The existing DD workbench now opens in **English** and offers **中文** in the header. The preference is shared with the legacy review and stitching-setup pages. Python 3.9.18 remains supported.

## Workbench sequence

1. Complete colour matching and semantic layer inference. In step **07 / Figure 4 maps**, use the native mask, its valid-support mask and the matching optical mosaic. Layer jobs fill these paths from their verified outputs; an unrelated mosaic is never inferred from its filename.
2. Click **Review exclusion boxes**. Python measures connected TL components and suggests a central aggregate rectangle when the recorded geometric policy identifies one candidate. This is a proposal, not a chemical identification or approval. If no unique candidate satisfies the policy, inspect both previews and draw a box yourself.
3. Compare the optical and segmentation previews. Drag to add a rectangle; select a rectangle and edit its native pixel bounds, or delete it. An empty list explicitly retains all editable regions. Existing bitmap exclusions are protected and shown in gray in the review preview.
4. Optionally click **Ask Kimi**. This sends only selected optical/segmentation ROI previews with bounded surrounding context to the configured Kimi provider. It assesses missed target boundaries and ordinary flakes inside the box. Its structured advice cannot change a box or approve an exclusion. Kimi is unnecessary for deterministic counting or manual review.
5. Check the human-review confirmation and click **Confirm regions and generate Figure 4**. Only then does the workbench create the confirmed exclusion record and calculate the retained composition. Changing an input invalidates the previous review. Source changes detected by hashing require a new review.

The figure has three aligned panels: **optical mosaic → actual layer segmentation → local composition**. Cyan dashed rectangles show the applied exclusions. Within each cell, 1L/2L/TL percentages divide by retained classified MoS₂ pixels, excluding substrate. Gray cells indicate partially omitted exclusion or non-scan support; hatched cells have no retained support. Whole-map fractions use summed class counts. Optical and raw prediction evidence remains visible inside the boxes for inspection; the boxes remove those pixels from statistics, rather than relabelling them.

## Provenance and limits

Proposals, preview hashes, automatic policy settings, optional Kimi advice and each human confirmation are saved under `Memory/dd_workflow/spatial_reviews/`. Every confirmation has a new directory. Source images and masks are never overwritten. Final maps and tables share the existing native-pixel counting core, with original and retained totals in the manifest. Layer predictions still require independent validation; a reviewed exclusion is not a new accuracy measurement.

The central proposal policy reuses stride-four TL measurement and 5×5 closing. It requires one sufficiently large interior component whose bounding box contains the canvas centre, rejects edge-connected or overly broad candidates, and pads each side by 8% of the candidate width/height. All parameters and rejected candidates are recorded. This deliberately conservative, geometry-based policy is a starting suggestion for the owner-requested central-aggregate workflow. It does not establish NaCl identity or make all large/thick flakes disposable. The human review step is essential.

## CLI and pipeline controls

Existing commands retain their previous behavior. Adding `--mosaic` creates the aligned three-panel figure. Supply a confirmed rectangle JSON with `--exclusions` to reproduce a human-reviewed result:

```sh
python tools/layer_spatial_maps.py --mask /path/mask.npy \
  --valid-mask /path/valid.npy --mosaic /path/colour_matched_mosaic.png \
  --exclusions /path/confirmed_exclusions.json \
  --sample sample_id --output /path/new_output
```

The explicit `--auto-central-exclusion` control exports a **proposal preview**, not a human-confirmed result. Its manifest records that confirmation is pending. Do not use this bypass as evidence of human approval. Unchecking the suggestion in the workbench preserves manual-only box review; omitting `--mosaic` preserves the previous spatial exports.

The optional integrated pipeline settings are `spatial_maps.figure4=true` and `spatial_maps.auto_central_exclusion=true`, alongside `enabled=true`. The actual optical source and its hash enter the cache key; ordinary area statistics and previews use the same effective exclusion mask as the spatial figure. Automatic outputs remain provisional until a separate human review. No scanner runtime, checkpoint, acquisition threshold or segmentation preprocessing is changed by this feature.
