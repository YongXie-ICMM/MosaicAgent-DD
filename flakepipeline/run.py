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
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    samples = cfg.pop("samples")
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
