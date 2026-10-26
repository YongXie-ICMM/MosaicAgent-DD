"""Offline spatial composition maps from saved semantic layer predictions.

Native MosaicAgent arrays use 0 substrate, 1 bilayer, 2 monolayer, 3 thick.
The numerical core uses named, canonical 0/1L/2L/TL classes internally.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile

import numpy as np
import PIL

from .stats import LAYERS, RGB, analyze, load_labels, require, sha

SPATIAL_MAPS_VERSION = "1.0.0"


def implementation_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_sources(mask_path, valid_path, exclusion_path):
    mask_path = Path(mask_path).resolve()
    sources = {"mask": {"path": str(mask_path), "sha256": sha(mask_path)}}
    if mask_path.suffix.lower() == ".npy":
        native = np.load(mask_path, allow_pickle=False)
        require(native.ndim == 2 and np.issubdtype(native.dtype, np.integer)
                and np.all(np.isin(native, [0, 1, 2, 3])),
                "Native mask must contain integer class IDs 0 substrate, 1=2L, 2=1L, 3=TL")
        labels = np.array([0, 2, 1, 3], dtype=np.uint8)[native]
        encoding = "MosaicAgent native IDs: 0 substrate, 1=2L, 2=1L, 3=TL"
    else:
        labels = load_labels(mask_path)
        encoding = "Exact RGB colours, independent of palette indices"
    valid = np.ones(labels.shape, dtype=bool)
    excluded = np.zeros(labels.shape, dtype=bool)
    rectangles = []
    if valid_path is not None:
        path = Path(valid_path).resolve()
        sources["valid_support"] = {"path": str(path), "sha256": sha(path)}
        valid = np.load(path, allow_pickle=False)
    if exclusion_path is not None:
        path = Path(exclusion_path).resolve()
        sources["exclusions"] = {"path": str(path), "sha256": sha(path)}
        if path.suffix.lower() == ".json":
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            require(labels.shape == (record["image_height"], record["image_width"]),
                    "Mask dimensions disagree with exclusion metadata")
            rectangles = record["excluded_rectangles"]
        else:
            excluded = np.load(path, allow_pickle=False)
    return labels, valid, excluded, rectangles, sources, encoding


def _write_table(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _paired_comparison(data, out):
    from . import heatmaps, combined
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 9))
    fig.subplots_adjust(left=.075, right=.98, bottom=.14, top=.85, wspace=.23)
    for ax, style, title in zip(axes, ("pies", "stacks"), ("Circular pies", "Rectangular stacked bars")):
        combined.spatial(ax, data, style)
        ax.set_title(title, fontsize=13, pad=12)
    fig.suptitle(f"{data['panel']} · same data, two display styles", fontsize=15, y=.975)
    fig.legend([Patch(facecolor=c) for c in combined.COLORS], LAYERS, loc="upper center",
               bbox_to_anchor=(.5,.94), ncol=3, frameon=False, fontsize=12)
    fig.text(.5,.04, "Identical grids, colours and counts. Each tile: 1L + 2L + TL = 100% of local MoS₂ area.\n"
             "† Partly omitted; hatched: no retained scan support. Symbol size does not encode coverage.",
             ha="center", fontsize=9, color="#40484d")
    heatmaps.save(fig, out, f"{data['panel']}_pies_vs_stacks")


def export_spatial_maps(mask_path, valid_path, exclusion_path, output_dir, sample,
                        heatmap_grid=(8, 15), combined_grid=(6, 11)):
    """Export both versions to a new directory, publishing the manifest last.

    RGB/P-mode PNG masks are decoded by exact colours. NPY input has the
    native MosaicAgent class order. Valid support and exclusions are never
    inferred from a downsampled/painted preview. Omitted valid_path explicitly
    means all canvas pixels are supported. No models or hardware are called.
    """
    require(isinstance(sample, str) and bool(sample.strip()) and len(sample) <= 120
            and not re.search(r'[<>:"/\\|?*\x00-\x1f]', sample)
            and sample not in (".", "..") and not sample.endswith((" ", ".")),
            "Sample name must be a safe filename label")
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Output already exists; choose a new directory to preserve previous results")
    for grid in (heatmap_grid, combined_grid):
        require(isinstance(grid, (tuple, list)) and len(grid) == 2,
                "Grid must be [columns, rows]")
    labels, valid, excluded, rectangles, sources, encoding = _load_sources(mask_path, valid_path, exclusion_path)
    results = []
    # Count and validate both grids before writing any plot, even in batch mode.
    for columns, rows in (heatmap_grid, combined_grid):
        table, xs, ys, whole = analyze(labels, rectangles, columns, rows,
                                      valid_mask=valid, exclusion_mask=excluded)
        results.append(dict(panel=sample, sulfur_mg=None, width=labels.shape[1], height=labels.shape[0],
                            rectangles=rectangles, global_counts=whole, **{"global": whole},
                            xedges=xs, yedges=ys, table=table))
    require(results[0]["global"] == results[1]["global"], "Different grids produced different whole-map totals")
    # Preview sampling has no effect on the full-resolution numerical counts.
    step = max(1, int(np.ceil(max(labels.shape[1]/850, labels.shape[0]/1500))))
    preview = RGB[labels[::step, ::step]].copy()
    support = valid[::step, ::step].astype(bool)
    omitted = excluded[::step, ::step].astype(bool)
    preview[~support] = (255,255,255)
    preview[support & omitted] = (150,150,150)
    results[0]["preview_rgb"] = preview
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".spatial-staging-", dir=output_dir.parent))
    try:
        from . import heatmaps, combined
        import matplotlib
        import matplotlib.pyplot as plt
        with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                             "svg.fonttype":"none", "svg.hashsalt":"mosaic-spatial-1"}):
            heatmaps.plot_sample(results[0], staging)
            _write_table(staging/f"{sample}_heatmap_tiles.csv", results[0]["table"])
            data = combined.prepare(results[1])
            combined.export_data(data, staging)
            for style in ("pies", "stacks"):
                combined.draw(data, staging, style)
            _paired_comparison(data, staging)
        for name, result in zip(("heatmaps", "combined"), results):
            numeric = {}
            for field in result["table"][0]:
                if field == "status":
                    continue
                numeric[field] = np.array([np.nan if t[field] is None else t[field] for t in result["table"]]).reshape(
                    len(result["yedges"])-1, len(result["xedges"])-1)
            np.savez_compressed(staging/f"{sample}_{name}_arrays.npz", xedges_px=result["xedges"],
                                yedges_px=result["yedges"], **numeric)
        for record in sources.values():
            require(sha(Path(record["path"])) == record["sha256"], "Source changed during analysis")
        manifest = {
            "schema_version":1, "kind":"layer_spatial_maps", "sample":sample,
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "implementation_version":SPATIAL_MAPS_VERSION,
            "implementation_sha256":implementation_fingerprint(),
            "versions":{"python":platform.python_version(),"numpy":np.__version__,
                        "pillow":PIL.__version__,"matplotlib":matplotlib.__version__},
            "sources":sources,"input_encoding":encoding,
            "support_policy":"Explicit valid-support mask" if valid_path is not None else "All canvas pixels assumed supported; no support mask supplied",
            "grids":{"heatmaps":list(heatmap_grid),"combined":list(combined_grid)},
            "edges":{name:{"x_px":r["xedges"].tolist(),"y_px":r["yedges"].tolist()}
                     for name,r in zip(("heatmaps","combined"),results)},
            "rectangles":rectangles,"global":results[0]["global"],
            "denominators":{"composition":"retained 1L + 2L + TL pixels",
                            "coverage":"retained valid scan pixels, including substrate"},
            "coordinates":"Mask pixels; top-left origin; analysis tiles, not camera frames; no physical scale inferred",
            "interpretation":"Existing segmentation predictions; TL is a combined thicker-layer class; not a new accuracy assessment",
            "outputs":[{"path":p.name,"sha256":sha(p),"role":"figure" if p.suffix in (".png",".svg") else "data"}
                       for p in sorted(staging.iterdir()) if p.is_file()],
        }
        (staging/"spatial_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        require(not output_dir.exists(), "Output appeared during analysis; refusing to overwrite it")
        staging.rename(output_dir)
        return {"manifest":str(output_dir/"spatial_manifest.json"),
                "outputs":[str(output_dir/item["path"]) for item in manifest["outputs"]]+[str(output_dir/"spatial_manifest.json")],
                "summary":{"global":manifest["global"],"grids":manifest["grids"]}}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
