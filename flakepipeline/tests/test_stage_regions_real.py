"""Exercise the REAL stages.stage_regions on a synthetic layer map (no torch, no
Kimi): measurement -> no-AI verdicts -> human override (with boundary refinement
running on the excluded region) -> --replay without a model."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import review as RV               # noqa: E402
import stages                     # noqa: E402


def _synthetic(tmp_path):
    H, W = 400, 400
    mask = np.zeros((H, W), np.uint8)
    mask[:, :] = 2                                  # monolayer everywhere
    yy, xx = np.mgrid[0:H, 0:W]
    pool = (yy - 200) ** 2 + (xx - 220) ** 2 < 60 ** 2   # one big thick-layer blob
    mask[pool] = 3
    mask[50:70, 50:70] = 3                           # a small thick flake
    photo = np.full((H, W, 3), 200, np.uint8)
    photo[pool] = (90, 110, 170)
    valid = np.ones((H, W), bool)
    mp = tmp_path / "mosaic.png"
    Image.fromarray(photo).save(mp)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "valid.npy", valid)
    return mp, tmp_path / "mask.npy", tmp_path / "valid.npy"


def _ctx(tmp_path, **kw):
    ctx = {"sample": "T", "work": tmp_path / "_work", "no_ai": True, "votes": 3,
           "min_region_pct": 1.0, "max_regions": 8, "region_scale_div": 1,
           "refine_min_pct": 5.0, "refine_rounds": 1}
    ctx["work"].mkdir(exist_ok=True)
    ctx.update(kw)
    return ctx


def test_regions_no_ai_then_human_then_replay(tmp_path):
    mp, mk, vp = _synthetic(tmp_path)

    # 1. no API key: every region kept, flagged uncertain, escalates
    r1 = stages.stage_regions(_ctx(tmp_path), mp, mk, vp)
    assert r1.escalate and "没有经过模型判定" in r1.escalate_reason
    v = r1.data["verdicts"]
    assert len(v) >= 1 and all(x["exclude"] is False for x in v)
    big = max(v, key=lambda x: x["area_px"])
    assert big["pct_of_class"] > 90

    # 2. the human excludes the big blob -> applied, boundary refinement runs, no escalation for it
    hv = [{"region_key": RV.region_key("T", big), "category": "precursor_melt",
           "exclude": True, "decided_by": "Yong Xie", "note": "液滴状熔池"}]
    r2 = stages.stage_regions(_ctx(tmp_path, human_verdicts=hv), mp, mk, vp)
    vb = max(r2.data["verdicts"], key=lambda x: x["area_px"])
    assert vb["exclude"] is True and vb["decided_by"] == "Yong Xie"
    assert vb["agent_exclude"] is False and vb["agent_category"] == "uncertain"
    assert r2.data["human_overrides"] and r2.data["excluded"] == 1
    assert r2.data["excluded_area_pct_of_valid"] > 5
    rec = json.loads((tmp_path / "_work" / "03_regions" / "T_exclusions.json").read_text())
    assert rec["human_overrides"][0]["human"]["decided_by"] == "Yong Xie"
    # the big blob is settled; only the small flake (still uncertain, no AI) can escalate
    assert "1 块没有经过模型判定" in r2.escalate_reason or not r2.escalate

    # 3. replay from the record: no model, same exclusion, flagged as replayed
    r3 = stages.stage_regions(_ctx(tmp_path, replay=True), mp, mk, vp)
    assert r3.data["replayed"] is True
    vr = max(r3.data["verdicts"], key=lambda x: x["area_px"])
    assert vr["exclude"] is True and vr.get("_replayed") and vr["decided_by"] == "Yong Xie"
    assert abs(r3.data["excluded_area_pct_of_valid"] - r2.data["excluded_area_pct_of_valid"]) < 0.01
