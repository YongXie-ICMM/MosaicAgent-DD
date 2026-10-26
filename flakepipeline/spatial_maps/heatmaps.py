#!/usr/bin/env python3
"""Separate spatial composition and coverage maps from saved segmentation.

No inference or image registration is performed. Counts use the full-size
lossless masks; image resizing is used only for the segmentation preview.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig3_spatial_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fig3_spatial_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

from .stats import LAYERS, RGB

def format_pct(value):
    return "N/A" if value is None else f"{value:.1f}%"


def exclusions(ax, rectangles):
    for rect in rectangles:
        ax.add_patch(Rectangle((rect["x1"], rect["y1"]), rect["x2"]-rect["x1"], rect["y2"]-rect["y1"],
                               facecolor="#eeeeee", edgecolor="#777777", linewidth=.7,
                               hatch="////", zorder=5))


def frame(ax, width, height):
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.set_xlabel("x in layer map (px)")
    ax.set_ylabel("y in layer map (px)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.tick_params(labelsize=8)


def heatmap(ax, result, field, label, annotate=False):
    xs, ys = result["xedges"], result["yedges"]
    values = np.array([np.nan if t[field] is None else t[field] for t in result["table"]]).reshape(len(ys)-1, len(xs)-1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(values), cmap=cmap,
                         norm=Normalize(0, 100), shading="flat", edgecolors="#ffffff", linewidth=.35)
    frame(ax, result["width"], result["height"])
    ax.set_title(label, fontsize=12, pad=9)
    if annotate:
        for record in result["table"]:
            value = record[field]
            x = (record["x0_px"]+record["x1_px"])/2
            y = (record["y0_px"]+record["y1_px"])/2
            if value is None or any(record["x0_px"] < b["x2"] and record["x1_px"] > b["x1"] and record["y0_px"] < b["y2"] and record["y1_px"] > b["y1"] for b in result["rectangles"]):
                continue
            partial = record["excluded_px"] + record["non_scan_px"] > 0
            ax.text(x, y, f"{value:.1f}" + ("†" if partial else ""), ha="center", va="center", fontsize=6.3,
                    color="white" if value < 55 else "#182028")
    exclusions(ax, result["rectangles"])
    return mesh


def save(fig, out, stem):
    fig.savefig(out/(stem+".png"), dpi=260, facecolor="white", bbox_inches="tight")
    fig.savefig(out/(stem+".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_sample(result, out):
    panel, sulphur = result["panel"], result["sulfur_mg"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.95), layout="constrained")
    # Preview only: numerical statistics above always use original pixels.
    preview = Image.fromarray(result["preview_rgb"])
    preview.thumbnail((850, 1500), Image.Resampling.NEAREST)
    ax = axes[0]
    ax.imshow(preview, extent=(0, result["width"], result["height"], 0), interpolation="nearest")
    for x in result["xedges"]:ax.axvline(x, color="white", linewidth=.35, alpha=.8)
    for y in result["yedges"]:ax.axhline(y, color="white", linewidth=.35, alpha=.8)
    exclusions(ax, result["rectangles"])
    frame(ax, result["width"], result["height"])
    ax.set_title("Segmentation + analysis grid", fontsize=12, pad=9)
    ax.legend(handles=[Patch(facecolor=colour, label=layer) for layer, colour in zip(LAYERS, ["#008000", "#800000", "#808000"])],
              loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, fontsize=8, frameon=False, handlelength=1)
    for ax, layer in zip(axes[1:], LAYERS):
        mesh = heatmap(ax, result, f"{layer}_of_crystal_pct", "Thicker layers (TL)" if layer == "TL" else layer, True)
    fig.colorbar(mesh, ax=list(axes[1:]), fraction=.025, pad=.02, label="Layer area / local MoS₂ area (%)", ticks=[0, 20, 40, 60, 80, 100], shrink=.9)
    g = result["global"]["of_crystal_pct"]
    fig.suptitle(f"{panel} · spatial layer composition\n"
                 f"{len(result['xedges'])-1} × {len(result['yedges'])-1} analysis tiles · whole-map fractions: 1L {format_pct(g['1L'])}, 2L {format_pct(g['2L'])}, TL {format_pct(g['TL'])}", fontsize=14)
    fig.supxlabel("Each tile: class pixels / (1L + 2L + TL pixels). Background omitted. † Partial support; grey cells: undefined.\n"
                  "TL is the existing thicker-layer class; it is not an independently resolved 3L class.", fontsize=9)
    save(fig, out, f"{panel}_layer_fraction_heatmaps")

    fig, axes = plt.subplots(1, 4, figsize=(13, 5.95), layout="constrained")
    for ax, layer in zip(axes[:3], LAYERS):
        mesh = heatmap(ax, result, f"{layer}_of_retained_canvas_pct", f"{layer} area coverage", True)
    heatmap(axes[3], result, "crystal_coverage_pct", "Total MoS₂ coverage", True)
    fig.colorbar(mesh, ax=list(axes), fraction=.025, pad=.02, label="Area / retained valid mapped area (%)", ticks=[0, 20, 40, 60, 80, 100], shrink=.9)
    fig.suptitle(f"{panel} · spatial area coverage", fontsize=14)
    fig.supxlabel("Denominator includes substrate/background in retained scan support. † Partly omitted tile; values use retained pixels.\n"
                  "1L coverage + 2L coverage + TL coverage = total MoS₂ coverage in each tile.", fontsize=9)
    save(fig, out, f"{panel}_area_coverage_heatmaps")
