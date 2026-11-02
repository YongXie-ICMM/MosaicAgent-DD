"""Source-preserving three-panel spatial figure and explicit central-region policy.

Inputs use canonical IDs: 0 background, 1 monolayer, 2 bilayer, 3 TL.
The optional geometric exclusion is not a chemical or layer identification.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np

from .stats import LAYERS, RGB, require


class CentralExclusionNeedsReview(ValueError):
    """Automatic exclusion was not uniquely supported by its stated policy."""

    def __init__(self, message, audit):
        super().__init__("needs_review: " + message)
        self.audit = audit


def _inputs(labels, valid):
    labels = np.asarray(labels)
    valid = np.asarray(valid)
    require(labels.ndim == 2 and min(labels.shape) > 0
            and np.issubdtype(labels.dtype, np.integer)
            and int(labels.min()) >= 0 and int(labels.max()) <= 3,
            "Canonical labels must be a nonempty 2D integer array with IDs 0..3")
    require(valid.shape == labels.shape and valid.dtype.kind in "biuf"
            and np.all((valid == 0) | (valid == 1)),
            "Valid support must be a matching binary mask")
    return labels, valid.astype(bool, copy=False)


def automatic_central_exclusion(labels, valid, *, downsample=4, closing_size=5,
                                min_pct_of_tl=1.0, max_candidates=8,
                                min_component_canvas_fraction=0.01,
                                max_bbox_canvas_fraction=0.40,
                                interior_margin_fraction=0.02,
                                padding_fraction=0.08):
    """Propose one reproducible rectangular exclusion, or stop for review.

    Candidate measurement reproduces ``region_agents.propose_regions``:
    stride-four TL samples, 5x5 binary closing, default scipy connectivity,
    original TL samples only for area, and the eight largest components.
    Selection additionally requires a large interior component whose half-open
    bounding box contains the canvas centre. Exactly one must qualify. There
    is no fallback to the largest flake, edge region, or an external model.

    Candidate areas are subsampling estimates, never exported layer counts.
    Call ``stats.analyze`` on the unchanged native labels for exact statistics.
    The returned rectangle excludes every class inside it. Padding uses ceil
    on each side independently. Inputs are never modified.
    """
    from scipy import ndimage

    labels, valid = _inputs(labels, valid)
    require(isinstance(downsample, int) and not isinstance(downsample, bool)
            and downsample >= 1, "downsample must be a positive integer")
    require(isinstance(closing_size, int) and closing_size > 0
            and closing_size % 2 == 1, "closing_size must be a positive odd integer")
    require(isinstance(max_candidates, int) and max_candidates > 0,
            "max_candidates must be a positive integer")
    require(0 <= min_pct_of_tl <= 100
            and 0 <= min_component_canvas_fraction <= max_bbox_canvas_fraction <= 1
            and 0 <= interior_margin_fraction < .5 and 0 <= padding_fraction <= .5,
            "Invalid central-exclusion policy thresholds")
    height, width = labels.shape
    policy = {
        "name": "central_tl_component_rectangle_v1",
        "canonical_ids": {"background": 0, "1L": 1, "2L": 2, "TL": 3},
        "downsample_stride": downsample, "closing_size": closing_size,
        "connectivity": "scipy.ndimage.label default (4-connected in 2D)",
        "min_pct_of_sampled_tl": float(min_pct_of_tl),
        "max_candidates": max_candidates,
        "min_component_canvas_fraction": float(min_component_canvas_fraction),
        "max_bbox_canvas_fraction": float(max_bbox_canvas_fraction),
        "interior_margin_fraction": float(interior_margin_fraction),
        "interior_margin_rounding": "ceil independently for width and height",
        "padding_fraction_per_side": float(padding_fraction),
        "padding_rounding": "ceil independently for bbox width and height",
        "selection": "exactly one eligible bbox containing the canvas centre",
        "edge_guard": "reject candidates overlapping any border-connected original sampled TL component",
        "bbox_convention": "x1,y1,x2,y2; half-open native saved-mask pixels",
        "legacy_candidate_bbox_convention": "x0,x1,y0,y1",
        "area_estimate": "original sampled TL support times stride squared",
        "exclusion": "all supported pixels inside the padded rectangle",
        "interpretation": "geometric central aggregate candidate; no chemical identity inferred",
        "on_absent_or_ambiguous": "needs_review; no automatic Figure 4 export",
    }
    audit = {"schema_version": 1, "status": "needs_review", "policy": policy,
             "image_width": width, "image_height": height,
             "rectangles": [], "candidates": []}
    sampled = (labels[::downsample, ::downsample] == 3) & valid[::downsample, ::downsample]
    total_tl = int(sampled.sum())
    if not total_tl:
        raise CentralExclusionNeedsReview(
            "No supported TL samples were found. Review the mask and provide an explicit exclusion or retain all regions.", audit)
    closed = ndimage.binary_closing(sampled, np.ones((closing_size, closing_size), dtype=bool))
    components, count = ndimage.label(closed)
    original_support = np.where(sampled, components, 0)
    edge_seed = np.zeros(sampled.shape, dtype=bool)
    edge_seed[0, :] = sampled[0, :]
    edge_seed[-1, :] = sampled[-1, :]
    edge_seed[:, 0] = sampled[:, 0]
    edge_seed[:, -1] = sampled[:, -1]
    edge_connected = ndimage.binary_propagation(edge_seed, mask=sampled) if edge_seed.any() else edge_seed
    sizes = np.bincount(original_support.ravel(), minlength=count + 1)
    sizes[0] = 0
    objects = ndimage.find_objects(components)
    margin_x, margin_y = ceil(width * interior_margin_fraction), ceil(height * interior_margin_fraction)
    canvas_area = width * height
    centre_x, centre_y = width / 2.0, height / 2.0
    eligible = []
    # Preserve the candidate order of the existing propose_regions function.
    for component_id in np.argsort(-sizes)[:max_candidates]:
        component_id = int(component_id)
        size = int(sizes[component_id])
        if not size or size < total_tl * min_pct_of_tl / 100.0:
            break
        ys, xs = objects[component_id - 1]
        x1, x2 = xs.start * downsample, min(width, xs.stop * downsample)
        y1, y2 = ys.start * downsample, min(height, ys.stop * downsample)
        bbox_area = (x2 - x1) * (y2 - y1)
        area = size * downsample * downsample
        reasons = []
        if not (x1 <= centre_x < x2 and y1 <= centre_y < y2):
            reasons.append("does_not_contain_canvas_centre")
        if area < canvas_area * min_component_canvas_fraction:
            reasons.append("component_too_small_for_canvas")
        if bbox_area > canvas_area * max_bbox_canvas_fraction:
            reasons.append("bbox_too_large_for_canvas")
        if x1 < margin_x or y1 < margin_y or x2 > width - margin_x or y2 > height - margin_y:
            reasons.append("bbox_not_interior")
        if np.any(edge_connected[ys, xs] & (original_support[ys, xs] == component_id)):
            reasons.append("connected_to_canvas_edge_in_original_sampled_tl")
        candidate = {
            "rank": len(audit["candidates"]) + 1, "label_id": component_id,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "legacy_bbox_x0_x1_y0_y1": [int(x1), int(x2), int(y1), int(y2)],
            "estimated_tl_area_px": area,
            "sampled_tl_area_px": size,
            "pct_of_sampled_tl": 100.0 * size / total_tl,
            "bbox_canvas_fraction": bbox_area / canvas_area,
            "component_canvas_fraction": area / canvas_area,
            "eligible": not reasons, "rejection_reasons": reasons,
        }
        audit["candidates"].append(candidate)
        if not reasons:
            eligible.append(candidate)
    if len(eligible) != 1:
        reason = "No central aggregate candidate satisfies the policy" if not eligible else "Multiple central candidates satisfy the policy"
        audit["eligible_candidates"] = len(eligible)
        raise CentralExclusionNeedsReview(
            reason + ". Inspect the optical image and choose an explicit exclusion or retain all regions; no region was removed.", audit)
    selected = eligible[0]
    x1, y1, x2, y2 = selected["bbox_xyxy"]
    pad_x, pad_y = ceil((x2 - x1) * padding_fraction), ceil((y2 - y1) * padding_fraction)
    rect = {"x1": max(0, x1 - pad_x), "y1": max(0, y1 - pad_y),
            "x2": min(width, x2 + pad_x), "y2": min(height, y2 + pad_y)}
    audit.update(status="selected", rectangles=[rect], selected=selected,
                 padding_px={"x_per_side": pad_x, "y_per_side": pad_y},
                 eligible_candidates=1)
    return audit


def draw_figure4(data, labels, valid, excluded, photo_path, out):
    """Save optical, actual segmentation and local composition at 180x135 mm.

    ``data`` is the combined-grid result produced by ``stats.analyze`` and the
    spatial exporter. It carries exact pixel counts, grid edges and rectangles.
    Optical and segmentation evidence stays visible inside exclusion boxes.
    All display coordinates are saved-mask pixels; no physical scale is inferred.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    labels, valid = _inputs(labels, valid)
    excluded = np.asarray(excluded)
    require(excluded.shape == labels.shape and excluded.dtype.kind in "biuf"
            and np.all((excluded == 0) | (excluded == 1)),
            "Exclusion mask must be a matching binary mask")
    height, width = labels.shape
    require((data["width"], data["height"]) == (width, height),
            "Figure data dimensions disagree with labels")
    xs, ys = np.asarray(data["xedges"]), np.asarray(data["yedges"])
    require(xs[0] == ys[0] == 0 and xs[-1] == width and ys[-1] == height
            and np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0),
            "Figure grid edges must partition the saved mask")
    table = data["table"]
    require(len(table) == (len(xs) - 1) * (len(ys) - 1), "Incomplete Figure 4 tile table")
    whole = data["global"]
    # Recheck source binding in bounded cell-sized buffers. This also ensures
    # that pre-existing bitmap exclusions and added rectangles are both used,
    # while preserving the unpainted evidence in panels (a) and (b).
    expected_cells = {(int(x0), int(y0), int(x1), int(y1))
                      for y0, y1 in zip(ys[:-1], ys[1:])
                      for x0, x1 in zip(xs[:-1], xs[1:])}
    seen_cells = set()
    for tile in table:
        bounds = tuple(tile[k] for k in ("x0_px", "y0_px", "x1_px", "y1_px"))
        require(bounds in expected_cells and bounds not in seen_cells, "Invalid or duplicate Figure 4 cell bounds")
        seen_cells.add(bounds)
        x0, y0, x1, y1 = bounds
        support = valid[y0:y1, x0:x1]
        keep = support & ~excluded[y0:y1, x0:x1].astype(bool)
        for rect in data.get("rectangles", []):
            ix0, iy0 = max(x0, rect["x1"]), max(y0, rect["y1"])
            ix1, iy1 = min(x1, rect["x2"]), min(y1, rect["y2"])
            if ix1 > ix0 and iy1 > iy0:
                keep[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = False
        counts = np.bincount(labels[y0:y1, x0:x1][keep], minlength=4)
        require(all(int(counts[i]) == tile[f"{layer}_px"] for i, layer in enumerate(LAYERS, 1))
                and int(counts[0]) == tile["substrate_background_px"]
                and int(keep.sum()) == tile["retained_canvas_px"]
                and int(support.sum() - keep.sum()) == tile["excluded_px"],
                "Figure 4 counts do not match labels, valid support and exclusion union")
    for layer in LAYERS:
        require(sum(t[f"{layer}_px"] for t in table) == whole["counts_px"][layer],
                "Figure 4 tile/global counts disagree")
    with Image.open(photo_path) as image:
        require(image.size == (width, height),
                "Colour-matched optical image dimensions must match the saved mask")
        image = image.convert("RGB")
        image.thumbnail((1100, 1750), Image.Resampling.LANCZOS)
        optical = np.asarray(image).copy()
    stride = max(1, int(ceil(max(width / 1100.0, height / 1750.0))))
    segmentation = RGB[labels[::stride, ::stride]].copy()
    segmentation[~valid[::stride, ::stride]] = 255
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(data["panel"]) + "_Figure4"
    paths = {kind: out / (stem + suffix) for kind, suffix in
             (("png", ".png"), ("svg", ".svg"), ("caption", "_caption.txt"))}
    require(not any(p.exists() for p in paths.values()),
            "Figure 4 output already exists; choose a new output directory")
    colours = ("#008000", "#800000", "#808000")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "svg.fonttype": "none", "svg.hashsalt": "mosaic-figure4-v1"}):
        # A noninteractive canvas preserves physical size on macOS as well as
        # Windows; GUI canvases can round figsize to screen-pixel dimensions.
        fig = Figure(figsize=(180 / 25.4, 135 / 25.4), facecolor="white")
        FigureCanvasAgg(fig)
        map_width = .255
        map_height = map_width * 180 / 135 * height / width
        require(map_height < .65, "Figure 4 source aspect ratio is too tall for the fixed layout")
        bottom = .30 + (.56 - map_height) / 2
        axes = [fig.add_axes([left, bottom, map_width, map_height]) for left in (.055, .3775, .70)]
        for ax, title in zip(axes, ("(a) Colour-matched optical", "(b) Layer segmentation", "(c) Local composition")):
            ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
            ax.set_title(title, fontsize=7.5, pad=9, loc="left", fontweight="bold")
            ax.set_xticks(np.linspace(0, width, 3), [f"{int(v):,}" for v in np.linspace(0, width, 3)])
            ax.set_yticks(np.linspace(0, height, 4), [f"{int(v):,}" for v in np.linspace(0, height, 4)])
            ax.tick_params(labelsize=6, length=2, pad=2)
            ax.set_xlabel("x (saved pixels)", fontsize=7)
            for spine in ax.spines.values():
                spine.set(color="#859096", linewidth=.5)
        axes[0].set_ylabel("y (saved pixels)", fontsize=7)
        for ax in axes[1:]:
            ax.set_yticklabels([])
        axes[0].imshow(optical, extent=(0, width, height, 0), interpolation="nearest")
        axes[1].imshow(segmentation, extent=(0, width, height, 0), interpolation="nearest")
        for ax in axes[:2]:
            for x in xs[1:-1]:
                ax.axvline(x, color="white", lw=.25, alpha=.5)
            for y in ys[1:-1]:
                ax.axhline(y, color="white", lw=.25, alpha=.5)
        ax = axes[2]
        for tile in table:
            x, y = tile["x0_px"], tile["y0_px"]
            w, h = tile["x1_px"] - x, tile["y1_px"] - y
            empty = tile["retained_canvas_px"] == 0
            partial = bool(tile["excluded_px"] or tile["non_scan_px"]) and not empty
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#e5e9ec" if empty or partial else "#fbfcfc",
                                   edgecolor="#bfc7cc", linewidth=.4, hatch="////" if empty else None))
            if empty or not tile["crystal_px"]:
                ax.text(x + w / 2, y + h / 2, "—", ha="center", va="center", fontsize=9, color="#68757d")
                continue
            left = x + .075 * w
            for layer, colour in zip(LAYERS, colours):
                value = tile[f"{layer}_px"] / tile["crystal_px"]
                ax.add_patch(Rectangle((left, y + .25 * h), .85 * w * value, .50 * h,
                                       facecolor=colour, edgecolor="white", linewidth=.3))
                left += .85 * w * value
        for ax in axes:
            for rect in data.get("rectangles", []):
                ax.add_patch(Rectangle((rect["x1"], rect["y1"]), rect["x2"] - rect["x1"],
                                       rect["y2"] - rect["y1"], fill=False, edgecolor="#00c6d8",
                                       linewidth=1.15, linestyle=(0, (4, 2)), zorder=10))
        fig.text(.50, .985, str(data["panel"]) + " · spatial layer composition", ha="center", va="top", fontsize=9)
        fig.legend([Patch(facecolor=c) for c in colours] + [Patch(facecolor="black")],
                   list(LAYERS) + ["Background"], loc="upper center", bbox_to_anchor=(.5, .952),
                   ncol=4, frameon=False, fontsize=7, handlelength=1.3, columnspacing=1.6)
        fig.legend([Patch(facecolor="#e5e9ec"), Patch(facecolor="#e5e9ec", hatch="////", edgecolor="#859096"),
                    Patch(fill=False, edgecolor="#00c6d8", linestyle="--")],
                   ["Partly omitted cell", "No retained support", "Excluded rectangle"],
                   loc="lower center", bbox_to_anchor=(.5, .205), ncol=3, frameon=False,
                   fontsize=6.5, handlelength=1.5, columnspacing=1.7)
        bar = fig.add_axes([.15, .135, .72, .045])
        bar.set(xlim=(0, 100), ylim=(0, 1)); bar.axis("off")
        left = 0.
        for layer, colour in zip(LAYERS, colours):
            value = whole["of_crystal_pct"][layer]
            if value is None:
                continue
            bar.add_patch(Rectangle((left, 0), value, 1, facecolor=colour, edgecolor="white", linewidth=.5))
            if value >= 5:
                bar.text(left + value / 2, .5, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=7)
            left += value
        proposal_pending = data.get("exclusion_review_status") == "automatic_proposal_unconfirmed"
        fig.text(.51, .19, "Whole-map composition after proposed exclusions" if proposal_pending
                 else "Whole-map composition after exclusions", ha="center", fontsize=7.5)
        fig.text(.5, .075, "Local bars: 1L + 2L + TL = 100% of retained classified MoS₂ area in each cell.", ha="center", fontsize=6.5)
        fig.text(.5, .045, "Predicted classes; TL = thicker layers. Equal bar size does not encode coverage. No physical scale inferred.",
                 ha="center", fontsize=6.2, color="#49545b")
        if proposal_pending:
            fig.text(.5, .017, "AUTOMATIC BOX PROPOSAL · HUMAN CONFIRMATION PENDING", ha="center",
                     fontsize=6.2, color="#9a3412")
        try:
            fig.savefig(paths["png"], dpi=500, facecolor="white")
            fig.savefig(paths["svg"], facecolor="white")
        finally:
            plt.close(fig)
    caption = (
        "Figure 4. Spatially resolved predicted layer composition. "
        "(a) Colour-matched stitched optical image. (b) Corresponding semantic layer-number prediction, "
        "with 1L, 2L and TL shown in green, dark red and olive; black denotes substrate/background and "
        "white denotes unsupported canvas. Predictions inside the cyan dashed exclusion rectangles remain visible for review. "
        f"(c) Local compositions in a {len(xs)-1} × {len(ys)-1} analysis grid. Each stacked bar is normalized "
        "to retained classified MoS₂ pixels (1L + 2L + TL) within that cell after excluding background and "
        "the indicated rectangles. Grey cells have partly omitted exclusion/non-scan support; hatched cells "
        "have no retained support. A dash also marks retained cells without classified MoS₂. "
        "The lower bar gives the corresponding whole-map fractions calculated from summed native pixel counts. "
        "Equal symbol sizes do not encode crystal coverage. Coordinates refer to saved-mask pixels, with a "
        "top-left origin; no physical calibration or gas-flow direction is inferred. TL denotes the combined "
        "thicker-layer class. These are model predictions, not an independent accuracy assessment.\n"
    )
    if data.get("exclusion_review_status") == "automatic_proposal_unconfirmed":
        caption += "Automatic exclusion-box proposal only; human confirmation is pending. These retained fractions are provisional.\n"
    paths["caption"].write_text(caption, encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}
