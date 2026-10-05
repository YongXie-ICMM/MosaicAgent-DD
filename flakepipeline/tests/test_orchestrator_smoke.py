"""One-minute smoke test of the orchestrator loop (ROADMAP 近-4), with the five stages
replaced by fakes so it needs neither torch, nor Kimi, nor the 4 GB of tiles.

What it pins down:
  * plan -> run -> cache: the second run hits the cache everywhere and the
    plan-vs-execution check passes;
  * TOOL USE: a named segment self-check failure ("填充色可能判错") triggers one
    parameter-repair retry, recorded as two attempts and decision=accept;
  * ESCALATE -> HUMAN -> OVERRIDE: a split vote lands in REVIEW/review_queue.json,
    a recorded human verdict is applied on the next run by geometric identity, the
    agent's original verdict stays next to it, and the escalation is gone;
  * --replay rebuilds verdicts without a model, flagging unmatched regions;
  * review.py refuses a decision without a note.

Run:  cd MosaicAgent && python3 -m pytest flakepipeline/tests -q
"""
import json
import sys
from pathlib import Path

import pytest

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import orchestrator as O          # noqa: E402
import review as RV               # noqa: E402
import stages                     # noqa: E402
from agents import Result         # noqa: E402


# ------------------------------------------------------------------ fakes
def fake_stitch(ctx):
    out = ctx["work"] / "01_mosaic" / f"{ctx['sample']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mosaic")
    return Result(ok=True, data={"mosaic": str(out)}, evidence="fake stitch")


def make_segment(fail_first=False):
    calls = {"n": 0}

    def fake_segment(ctx, mosaic):
        calls["n"] += 1
        outdir = ctx["work"] / "02_segment"
        outdir.mkdir(parents=True, exist_ok=True)
        mask, valid = outdir / "m.npy", outdir / "v.npy"
        mask.write_bytes(b"mask" + str(ctx.get("filler")).encode())
        valid.write_bytes(b"valid")
        data = {"mask": str(mask), "valid_mask": str(valid), "tiles": 4, "tile": 512,
                "overlap": ctx.get("overlap"), "excluded_non_scan_pct": 6.1}
        if fail_first and calls["n"] == 1:
            return Result(ok=False, confidence=0.5, data=data,
                          evidence="非扫描区 0.10%",
                          escalate=True, escalate_reason="非扫描区只占 0.10%，填充色可能判错")
        return Result(ok=True, confidence=0.95, data=data, evidence="非扫描区 6.1%")
    fake_segment.calls = calls
    return fake_segment


def fake_audit(ctx, mosaic, mask, valid):
    # escalate-only stage between segment and regions; writes the audit json the
    # cache looks for and returns no product key
    outdir = ctx["work"] / "02_segment"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{ctx['sample']}_audit.json"
    out.write_text(json.dumps({"detector": {"alarms": []}, "pilot": {"ran": False}}), encoding="utf-8")
    return Result(ok=True, confidence=0.9,
                  data={"audit": str(out), "n_alarms": 0, "alarms": [],
                        "low_var_crystal_frac": 0.0, "pilot": {"ran": False}},
                  evidence="fake audit")


def make_regions(split=True):
    def fake_regions(ctx, mosaic, mask, valid):
        outdir = ctx["work"] / "03_regions"
        outdir.mkdir(parents=True, exist_ok=True)
        v = {"rank": 1, "label_id": 3, "area_px": 40000, "pct_of_class": 70.2,
             "pct_of_all_crystal": 12.0, "bbox": (1000, 1200, 500, 700),
             "solidity": 0.87, "bbox_fill": 0.6, "mean_rgb": (120, 130, 160),
             "category": "precursor_melt", "exclude": True, "confidence": 0.7,
             "reason": "液滴状", "_ai": True,
             "_tally": {"precursor_melt": 2, "multilayer_crystals": 1} if split
             else {"precursor_melt": 3},
             "_n_votes": 3}
        verdicts = [v]
        log = []
        if ctx.get("human_verdicts"):
            verdicts, log = RV.apply_overrides(ctx["sample"], verdicts,
                                               ctx["human_verdicts"], int(ctx.get("votes", 3)))
        (outdir / f"{ctx['sample']}_exclusions.json").write_text(
            json.dumps({"sample": ctx["sample"], "verdicts": verdicts,
                        "human_overrides": log}, default=list), encoding="utf-8")
        votes_n = int(ctx.get("votes", 3))
        open_v = [x for x in verdicts if not x.get("decided_by")]
        spl = [x for x in open_v if max(x["_tally"].values()) < votes_n]
        return Result(ok=not spl, confidence=0.7 if spl else 1.0,
                      data={"candidates": 1, "verdicts": verdicts, "human_overrides": log,
                            "exclusion_mask": ""},
                      evidence="1 块候选",
                      escalate=bool(spl),
                      escalate_reason="票数分歧: #1(precursor_melt)" if spl else "")
    return fake_regions


def fake_stats(ctx, mask, valid, excl):
    outdir = ctx["work"] / "04_stats"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{ctx['sample']}_stats.json").write_text("{}", encoding="utf-8")
    return Result(ok=True, data={"ratios_pct": {"1L": 71.6, "2L": 15.7, "TL": 12.7},
                                 "ratios_pct_without_region_exclusion": {"1L": 60, "2L": 12, "TL": 28},
                                 "delta_from_region_exclusion_pp": {"1L": 11.6, "2L": 3.7, "TL": -15.3}},
                  evidence="fake stats")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "w.pth").write_bytes(b"weights")
    (tmp_path / "mosaic.png").write_bytes(b"png")
    return {"work_root": str(tmp_path / "_work"), "weights": str(tmp_path / "w.pth"),
            "tile": 512, "overlap": 64, "filler": "white", "votes": 3,
            "exclusion_rule": "test",
            "samples": [{"name": "S1", "mosaic_in": str(tmp_path / "mosaic.png")}]}


def wire(monkeypatch, segment=None, regions=None, audit=None):
    monkeypatch.setattr(stages, "stage_stitch", fake_stitch)
    monkeypatch.setattr(stages, "stage_segment", segment or make_segment())
    monkeypatch.setattr(stages, "stage_audit", audit or fake_audit)
    monkeypatch.setattr(stages, "stage_regions", regions or make_regions(split=False))
    monkeypatch.setattr(stages, "stage_stats", fake_stats)


# ------------------------------------------------------------------ tests
def test_plan_run_then_cache(cfg, monkeypatch):
    wire(monkeypatch)
    samples = cfg.pop("samples")
    o = O.Orchestrator(cfg)
    plan = o.plan(samples)
    acts = [l["action"] for l in plan["samples"][0]["stages"]]
    assert acts[0] == "run" and all(a.startswith("after") for a in acts[1:])
    assert (Path(cfg["work_root"]) / "plan.json").exists()

    rep = o.run(samples)
    assert rep["samples"][0]["actions"] == {s: "run" for s in O.Orchestrator.STAGES}
    man = json.loads(Path(rep["samples"][0]["manifest"]).read_text(encoding="utf-8"))
    assert man["stages"]["segment"]["decision"] == "accept"
    assert len(man["stages"]["segment"]["attempts"]) == 1

    rep2 = O.Orchestrator(cfg).run(samples)
    assert rep2["samples"][0]["actions"] == {s: "cache" for s in O.Orchestrator.STAGES}
    plan2 = rep2["plan"]["samples"][0]["stages"]
    assert all(l["action"] == "cache" for l in plan2)
    chk = {c["check"]: c for c in rep2["consistency"]}
    assert chk["Execution matched the plan (cache hits as planned; repairs listed)"]["pass"]


def test_tool_repairs_named_failure(cfg, monkeypatch):
    seg = make_segment(fail_first=True)
    wire(monkeypatch, segment=seg)
    samples = cfg.pop("samples")
    rep = O.Orchestrator(cfg).run(samples)
    assert seg.calls["n"] == 2
    man = json.loads(Path(rep["samples"][0]["manifest"]).read_text(encoding="utf-8"))
    st = man["stages"]["segment"]
    assert st["decision"] == "accept" and len(st["attempts"]) == 2
    assert st["attempts"][1]["tool"] == "retry_filler"
    assert st["attempts"][1]["params"]["filler"] == "black"
    assert rep["samples"][0]["actions"]["segment"] == "repair"
    assert rep["samples"][0]["escalations"] == []


def test_no_repair_switch_escalates(cfg, monkeypatch):
    seg = make_segment(fail_first=True)
    wire(monkeypatch, segment=seg)
    samples = cfg.pop("samples")
    cfg["no_repair"] = True
    rep = O.Orchestrator(cfg).run(samples)
    assert seg.calls["n"] == 1
    assert [e["stage"] for e in rep["samples"][0]["escalations"]] == ["segment"]
    q = json.loads(RV.queue_path(Path(cfg["work_root"])).read_text(encoding="utf-8"))
    assert q and q[0]["stage"] == "segment"


def test_escalation_then_human_override(cfg, monkeypatch):
    wire(monkeypatch, regions=make_regions(split=True))
    samples = cfg.pop("samples")
    work = Path(cfg["work_root"])
    rep = O.Orchestrator(cfg).run(samples)
    esc = rep["samples"][0]["escalations"]
    assert esc and esc[0]["stage"] == "regions" and esc[0]["regions"][0]["rank"] == 1
    q = json.loads(RV.queue_path(work).read_text(encoding="utf-8"))
    key = q[0]["regions"][0]["region_key"]
    assert key["sample"] == "S1" and key["area_px"] == 40000

    # the human keeps the region (faceted edges), slightly different centroid
    hv = [{"region_key": {**key, "cx": key["cx"] + 3}, "category": "multilayer_crystals",
           "exclude": False, "decided_by": "Yong Xie", "note": "有 60 度晶棱", "at": "now"}]
    RV.verdicts_path(work).parent.mkdir(parents=True, exist_ok=True)
    RV.verdicts_path(work).write_text(json.dumps(hv), encoding="utf-8")

    rep2 = O.Orchestrator(cfg).run(samples)
    s = rep2["samples"][0]
    assert s["actions"]["regions"] == "run"          # human hash changed the key
    assert s["escalations"] == []
    v = s["regions"][0]
    assert v["exclude"] is False and v["decided_by"] == "Yong Xie"
    assert v["agent_category"] == "precursor_melt" and v["agent_exclude"] is True
    assert v["override_of_unanimous"] is False       # 2/3, not unanimous
    assert len(s["human_overrides"]) == 1
    chk = {c["check"]: c for c in rep2["consistency"]}
    assert chk["Human overrides are recorded, and none reverses a unanimous vote silently"]["pass"]
    assert chk["No escalations are awaiting a human"]["pass"]


def test_override_of_unanimous_is_flagged():
    v = [{"rank": 1, "area_px": 100, "bbox": (0, 10, 0, 10), "category": "precursor_melt",
          "exclude": True, "_tally": {"precursor_melt": 3}}]
    hv = [{"region_key": RV.region_key("S", v[0]), "exclude": False,
           "category": "thick_flake", "decided_by": "me", "note": "n"}]
    out, log = RV.apply_overrides("S", v, hv, votes=3)
    assert out[0]["override_of_unanimous"] is True and log[0]["changed"]


def test_replay_matches_by_geometry():
    class C:
        def __init__(self, r, a, b):
            self.d = {"rank": r, "area_px": a, "bbox": b, "label_id": r}
        def to_dict(self):
            return dict(self.d)
    cands = [C(1, 40000, (1000, 1200, 500, 700)), C(2, 900, (10, 40, 10, 40))]
    recorded = [{"rank": 7, "area_px": 41000, "bbox": (1002, 1198, 503, 699),
                 "category": "precursor_melt", "exclude": True, "confidence": 0.8,
                 "reason": "r", "_tally": {"precursor_melt": 3}, "_ai": True}]
    out = RV.replay_verdicts("S", cands, recorded)
    assert out[0]["_replayed"] and out[0]["exclude"] and out[0]["_replay_source_rank"] == 7
    assert out[1]["_replay_unmatched"] and out[1]["exclude"] is False


def test_review_cli_requires_note(tmp_path, monkeypatch, capsys):
    work = tmp_path / "_work"
    d = work / "S1" / "03_regions"
    d.mkdir(parents=True)
    (d / "S1_exclusions.json").write_text(json.dumps({"verdicts": [
        {"rank": 1, "area_px": 100, "bbox": (0, 10, 0, 10), "category": "precursor_melt",
         "exclude": True, "_tally": {"precursor_melt": 3}}]}), encoding="utf-8")
    argv = ["review.py", "--work", str(work), "decide", "--sample", "S1", "--rank", "1",
            "--keep", "--by", "me", "--note", "  "]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        RV.main()
    argv[-1] = "有晶棱"
    monkeypatch.setattr(sys, "argv", argv)
    RV.main()
    hv = json.loads(RV.verdicts_path(work).read_text(encoding="utf-8"))
    assert hv[0]["exclude"] is False and hv[0]["note"] == "有晶棱"
    assert hv[0]["agent_at_decision"]["category"] == "precursor_melt"
