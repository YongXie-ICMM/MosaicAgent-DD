#!/usr/bin/env python3
"""Create spatial layer heatmaps and combined maps from saved segmentation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakepipeline.spatial_maps import export_spatial_maps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True, type=Path, help="Original colour mask PNG or native MosaicAgent .npy mask")
    parser.add_argument("--valid-mask", type=Path, help="Native valid scan support .npy; omitted means the complete canvas is supported")
    parser.add_argument("--exclusions", type=Path, help="Exclusion .npy or rectangle JSON with source image dimensions")
    parser.add_argument("--sample", default="sample", help="Sample label used in figures and filenames")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; previous results are never overwritten")
    parser.add_argument("--heatmap-grid", nargs=2, type=int, metavar=("COLS","ROWS"), default=(8,15))
    parser.add_argument("--combined-grid", nargs=2, type=int, metavar=("COLS","ROWS"), default=(6,11))
    args = parser.parse_args(argv)
    try:
        result = export_spatial_maps(args.mask,args.valid_mask,args.exclusions,args.output,args.sample,
                                     args.heatmap_grid,args.combined_grid)
    except (OSError,ValueError,KeyError) as exc:
        parser.error(str(exc))
    print("Saved spatial layer maps / 已保存空间层数统计："+result["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
