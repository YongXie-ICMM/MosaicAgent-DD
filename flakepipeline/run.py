#!/usr/bin/env python3
"""FlakePipeline CLI.

Example -- take three existing stitched maps and run segmentation -> region
adjudication -> statistics:
    python3 run.py --config configs/figure3.json

Inference only, no training. Segmentation uses the .pth that `weights` points at in the
config file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from orchestrator import Orchestrator


def main():
    ap = argparse.ArgumentParser(description="FlakePipeline: stitch -> segment -> region adjudication -> statistics")
    ap.add_argument("--config", required=True, help="path to the JSON configuration")
    ap.add_argument("--no-ai", action="store_true", help="skip the Kimi calls; keep every region")
    ap.add_argument("--force", action="store_true", help="ignore the cache and re-run everything")
    ap.add_argument("--only", default=None, help="run a single sample by name")
    ap.add_argument("--plan", action="store_true",
                    help="print what would run (cache / run / blocked) and exit")
    ap.add_argument("--replay", action="store_true",
                    help="rebuild region verdicts from each sample's recorded exclusions.json; no model call")
    ap.add_argument("--verdicts", default=None,
                    help="human_verdicts.json to apply (default: <work_root>/REVIEW/human_verdicts.json)")
    ap.add_argument("--no-repair", action="store_true",
                    help="disable the orchestrator's automatic parameter-repair tools")
    ap.add_argument("--layer-input-contract", help="layer_input_contract.json from stitch setup")
    ap.add_argument("--reference-capture-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                    help="previous model inference capture size, not the training crop")
    ap.add_argument("--same-fov-confirmed", action="store_true",
                    help="operator has verified unchanged physical FOV and objective (not cropping)")
    a = ap.parse_args()
    if (a.layer_input_contract is None) != (a.reference_capture_size is None):
        ap.error("--layer-input-contract and --reference-capture-size must be supplied together")
    if a.same_fov_confirmed and not a.layer_input_contract:
        ap.error("--same-fov-confirmed requires the recorded input contract")

    cfg = json.loads(Path(a.config).read_text())
    samples = cfg.pop("samples")
    if a.layer_input_contract:
        from inference_geometry import geometry_from_layer_contract
        try:
            geometry = geometry_from_layer_contract(a.layer_input_contract, a.reference_capture_size,
                        same_physical_fov_confirmed=a.same_fov_confirmed,
                        model_tile=cfg.get("tile", 512), model_overlap=cfg.get("overlap", 64))
        except (OSError, ValueError) as exc:
            ap.error("Invalid layer input contract / 层数输入记录核验失败: " + str(exc))
        if not geometry["adapted_mode_allowed"]:
            ap.error("Layer scale is not verified / 层数识别尺度尚未核对: " + ", ".join(geometry["reasons"]))
        cfg["inference_geometry"] = geometry
    if a.no_ai:
        cfg["no_ai"] = True
    if a.force:
        cfg["force"] = True
    if a.replay:
        cfg["replay"] = True
    if a.verdicts:
        cfg["human_verdicts"] = a.verdicts
    if a.no_repair:
        cfg["no_repair"] = True
    if a.only:
        samples = [s for s in samples if s["name"] == a.only]
        if not samples:
            sys.exit(f"没有名为 {a.only} 的样品")

    orch = Orchestrator(cfg)
    if a.plan:
        orch.print_plan(orch.plan(samples))
        print(f"\n计划已写入 {orch.work_root / 'plan.json'}（只看不跑）")
        return
    rep = orch.run(samples)
    print("\n===== 汇总 =====")
    for s in rep["samples"]:
        r = s["ratios_pct"]
        print(f"  {s['sample']:<12} 1L {r.get('1L'):>6}%  2L {r.get('2L'):>6}%  TL {r.get('TL'):>6}%"
              f"   （未剔除区域时：{s['ratios_pct_without_region_exclusion']}）")
    print("\n----- 一致性 -----")
    for c in rep["consistency"]:
        print(f"  [{'ok' if c['pass'] else '注意'}] {c['check']}\n      {c['detail']}")
    print(f"\n报告：{Path(rep['config']['work_root']) / 'report.json'}")


if __name__ == "__main__":
    main()
