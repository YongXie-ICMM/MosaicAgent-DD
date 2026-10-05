"""The segmentation auditor (ROADMAP 中-4) on synthetic data, without torch or Kimi.

  * the free detector catches the 2026-08-29 failure class: a flat white rectangle
    whose mask says thick layer is alarmed -- exactly that region, nothing else --
    and the stage escalates;
  * the darkness criterion (audit_pilot, 2026-09-02) catches the noisy near-black band
    beyond the wafer edge labelled 1L that the variance detector cannot see; it is a
    named parameter that enters the cache key, and a clean mosaic still raises nothing;
  * a textured mosaic with a plausible mask raises nothing and passes;
  * the window sampler is deterministic, non-overlapping and stratified;
  * the VLM pilot's aggregation is exercised with an injected fake pool, so no key and
    no network are needed: it escalates on evidence (flagged fraction over the
    threshold, or any unanimous systematic_error window) and reports split votes as
    information only; with --no-ai the pilot is skipped and the detector still runs;
  * the orchestrator runs five stages with the REAL stage_audit between fake segment
    and fake regions, records it under the segmentation_auditor agent, and the mask
    reaches the downstream stages untouched.

Run:  cd MosaicAgent && python3 -m pytest flakepipeline/tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import orchestrator as O          # noqa: E402
import stages                     # noqa: E402
from agents import REGISTRY, Result   # noqa: E402

H = W = 600
RECT = (0, 200, 0, 200)           # (x0, x1, y0, y1): a white corner, like the real bug


def _textured(seed=0, std=8.0):
    """Substrate-like photo: mid grey with pixel noise and some crystal-ish blobs."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), 150, np.float64)
    base += rng.normal(0, std, (H, W, 3))
    yy, xx = np.mgrid[0:H, 0:W]
    for cy, cx, r, col in ((300, 300, 60, (110, 140, 190)), (450, 150, 40, (90, 110, 170)),
                           (150, 450, 50, (130, 160, 200))):
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        base[blob] = np.array(col, np.float64) + rng.normal(0, std, (int(blob.sum()), 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def _mask_for(photo):
    """Mask that follows the blobs: 1L over the substrate, 2L / TL on the blobs."""
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.full((H, W), 2, np.uint8)                      # monolayer film everywhere
    m[(yy - 300) ** 2 + (xx - 300) ** 2 < 60 ** 2] = 1    # 2L
    m[(yy - 450) ** 2 + (xx - 150) ** 2 < 40 ** 2] = 3    # TL
    m[(yy - 150) ** 2 + (xx - 450) ** 2 < 50 ** 2] = 1    # 2L
    m[:40, 250:350] = 0                                   # some bare substrate
    return m


def _write(tmp_path, photo, mask, valid=None):
    valid = np.ones(mask.shape, bool) if valid is None else valid
    mp = tmp_path / "mosaic.png"
    Image.fromarray(photo).save(mp)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "valid.npy", valid)
    return mp, tmp_path / "mask.npy", tmp_path / "valid.npy"


def _ctx(tmp_path, **kw):
    ctx = {"sample": "T", "work": tmp_path / "_work", "votes": 3, "seed": 0,
           "audit_scale_div": 1, "audit_window": 32}
    ctx.update(kw)
    ctx["work"].mkdir(exist_ok=True)
    return ctx


# ------------------------------------------------------------------ detector
def test_white_corner_labelled_tl_is_alarmed(tmp_path):
    photo = _textured()
    x0, x1, y0, y1 = RECT
    photo[y0:y1, x0:x1] = 255                          # flat white, no structure at all
    mask = _mask_for(photo)
    mask[y0:y1, x0:x1] = 3                             # ...and the segmenter said TL
    mask[(np.mgrid[0:H, 0:W][0] - 450) ** 2 + (np.mgrid[0:H, 0:W][1] - 150) ** 2 < 40 ** 2] = 1
    paths = _write(tmp_path, photo, mask)
    ctx = _ctx(tmp_path)
    res = stages.stage_audit(ctx, *paths)

    assert res.escalate and not res.ok and res.confidence == 0.5
    assert res.data["n_alarms"] == 1 and len(res.data["alarms"]) == 1
    a = res.data["alarms"][0]
    # exactly that region: the alarm sits inside the white rectangle and covers most
    # of it (the sliding window erodes ~half a window at the textured boundary)
    bx0, bx1, by0, by1 = a["bbox"]
    assert x0 <= bx0 and bx1 <= x1 and y0 <= by0 and by1 <= y1
    assert a["area_px"] >= 0.6 * (x1 - x0) * (y1 - y0)
    assert a["dominant_class"] == "TL" and a["class_composition"]["TL"] == 100.0
    assert a["pct_of_class"] > 60                      # TL lives almost only there
    assert a["mean_gray"] > 250 and a["mean_var"] < 1.0
    assert res.data["low_var_frac_by_class"]["TL"] > 0.6
    assert res.data["low_var_frac_by_class"]["1L"] == 0.0
    assert "1 处报警" in res.evidence and "无纹理或近黑区域被判成晶体" in res.escalate_reason
    assert a["low_var_frac"] == 1.0 and a["dark_frac"] == 0.0
    # written record, parameters included, pilot off by default
    doc = json.loads(Path(res.data["audit"]).read_text())
    assert doc["detector"]["n_alarms"] == 1 and doc["detector"]["alarms"][0]["rank"] == 1
    assert doc["parameters"]["audit_var_thr"] == stages.AUDIT_DEFAULTS["audit_var_thr"]
    assert doc["parameters"]["audit_dark_gray"] == 60.0
    assert doc["pilot"]["ran"] is False and doc["escalate"] is True
    # the mask on disk is untouched: the auditor only escalates
    assert np.array_equal(np.load(paths[1]), mask)


def test_white_corner_outside_valid_is_not_alarmed(tmp_path):
    """Once the segment stage has excluded the filler geometrically the auditor must
    stay silent about it -- that is the normal, fixed pipeline."""
    photo = _textured()
    x0, x1, y0, y1 = RECT
    photo[y0:y1, x0:x1] = 255
    mask = _mask_for(photo)
    mask[y0:y1, x0:x1] = 0
    valid = np.ones((H, W), bool)
    valid[y0:y1, x0:x1] = False
    paths = _write(tmp_path, photo, mask, valid)
    res = stages.stage_audit(_ctx(tmp_path), *paths)
    assert res.ok and not res.escalate and res.data["n_alarms"] == 0


def test_clean_synthetic_passes(tmp_path):
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    res = stages.stage_audit(_ctx(tmp_path), *paths)
    assert res.ok and not res.escalate and res.confidence == 0.9
    assert res.data["n_alarms"] == 0 and res.data["alarms"] == []
    assert res.data["low_var_crystal_frac"] == 0.0 and res.data["dark_crystal_frac"] == 0.0
    assert res.escalate_reason == ""
    assert Path(res.data["audit"]).exists()


def test_dark_band_labelled_1l_is_alarmed(tmp_path):
    """The S05mg case: a noisy near-black band beyond the wafer edge (variance far
    above var_thr, so the variance detector misses it) that the U-Net labelled 1L."""
    rng = np.random.default_rng(1)
    photo = _textured()
    band = 80
    photo[:band] = np.clip(20 + rng.normal(0, 8, (band, W, 3)), 0, 255).astype(np.uint8)
    mask = _mask_for(photo)
    mask[:band] = 2                                     # ...and the segmenter said 1L
    paths = _write(tmp_path, photo, mask)
    res = stages.stage_audit(_ctx(tmp_path), *paths)
    assert res.escalate and not res.ok and res.data["n_alarms"] == 1
    a = res.data["alarms"][0]
    assert a["dominant_class"] == "1L" and a["class_composition"]["1L"] == 100.0
    assert a["mean_gray"] < 60 and a["mean_var"] > stages.AUDIT_DEFAULTS["audit_var_thr"]
    assert a["dark_frac"] > 0.95 and a["low_var_frac"] == 0.0     # darkness fired, variance did not
    assert a["bbox"][3] <= band and a["area_px"] >= 0.9 * band * W
    assert res.data["low_var_crystal_frac"] == 0.0 and res.data["dark_crystal_frac"] > 0.1
    assert res.data["dark_frac_by_class"]["1L"] > 0.1 and res.data["dark_frac_by_class"]["TL"] == 0.0
    assert "近黑区域被判成晶体" in res.escalate_reason and f"平均灰度 {a['mean_gray']}" in res.escalate_reason
    # the threshold is a named parameter: switching it off silences exactly this alarm,
    # and it is in the recorded parameters (hence in the orchestrator's cache key)
    res0 = stages.stage_audit(_ctx(tmp_path, audit_dark_gray=0), *paths)
    assert res0.ok and res0.data["n_alarms"] == 0
    doc = json.loads(Path(res0.data["audit"]).read_text())
    assert doc["parameters"]["audit_dark_gray"] == 0.0
    assert stages.audit_params({})["audit_dark_gray"] == 60.0
    assert "audit_dark_gray" in stages.AUDIT_DEFAULTS


def test_min_px_and_scale_div_are_honoured(tmp_path):
    photo = _textured()
    x0, x1, y0, y1 = RECT
    photo[y0:y1, x0:x1] = 255
    mask = _mask_for(photo)
    mask[y0:y1, x0:x1] = 3
    paths = _write(tmp_path, photo, mask)
    # too small a minimum area threshold -> still one alarm at scale_div 2
    res = stages.stage_audit(_ctx(tmp_path, audit_scale_div=2, audit_window=16), *paths)
    assert res.data["n_alarms"] == 1 and res.data["alarms"][0]["area_px"] > 10000
    # an absurdly large minimum silences it, but the fraction still reports the truth
    res = stages.stage_audit(_ctx(tmp_path, audit_min_px=10 ** 7), *paths)
    assert res.data["n_alarms"] == 0 and res.data["low_var_frac_by_class"]["TL"] > 0.6
    # scale_div is honoured as the fallback name
    P = stages.audit_params({"scale_div": 8})
    assert P["audit_scale_div"] == 8
    assert stages.audit_params({"scale_div": 8, "audit_scale_div": 2})["audit_scale_div"] == 2


# ------------------------------------------------------------------ sampler
def test_sampler_is_deterministic_stratified_and_non_overlapping():
    mask = np.zeros((1024, 1024), np.uint8)
    mask[100:500, 600:1000] = 2                      # crystal-rich top-right
    valid = np.ones((1024, 1024), bool)
    valid[:, :128] = False                           # a non-scanned strip on the left
    a = stages.sample_audit_windows(mask, valid, 4, 256, seed=7)
    b = stages.sample_audit_windows(mask, valid, 4, 256, seed=7)
    assert a == b and len(a) == 4
    # one window per spatial stratum (2 x 2 for n = 4)
    assert sorted(w["stratum"] for w in a) == [0, 1, 2, 3]
    # non-overlapping, inside the valid area
    for i, w in enumerate(a):
        x0, x1, y0, y1 = w["bbox"]
        assert valid[y0:y1, x0:x1].all() and w["valid_frac"] == 1.0
        for v in a[i + 1:]:
            assert not stages._bbox_overlap(w["bbox"], v["bbox"])
    # the crystal-rich stratum picks a crystal-rich window
    tr = [w for w in a if w["stratum"] == 1][0]
    assert tr["crystal_frac"] > 0.5
    # asking for more than exists is capped; nothing is invented
    assert len(stages.sample_audit_windows(mask, valid, 50, 256, seed=0)) == 12
    assert stages.sample_audit_windows(mask, valid, 0, 256) == []


# ------------------------------------------------------------------ pilot
class _FakeClient:
    available = True


class _FakePool:
    """Stands in for kimi_agents.AgentPool: calls build() so the crop code runs, then
    returns canned answers in the shape AgentPool.map produces."""
    def __init__(self, answers):
        self.client = _FakeClient()
        self.answers = answers
        self.seen = []

    def map(self, role, items, build, key="category", use_votes=True, **kw):
        assert role.name == "segmentation_auditor" and key == "verdict"
        out = []
        for i, it in enumerate(items):
            prompt, images = build(it)
            assert len(images) == 2 and images[0].shape == images[1].shape
            assert images[0].shape[0] == 768 and images[0].ndim == 3
            self.seen.append(prompt)
            out.append(self.answers[i % len(self.answers)])
        return out


def _ans(verdict, tally, conf=0.8, ec="none"):
    return {"verdict": verdict, "error_class": ec, "confidence": conf, "reason": "r",
            "_tally": tally, "_n_votes": sum(tally.values())}


def test_pilot_split_votes_are_information_only(tmp_path):
    """S15mg in the audit_pilot experiment: two 2-1 splits with nothing real behind
    them must not escalate (at temperature 1, 8/45 windows split and none was an
    error). They are still counted and written down."""
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    ctx = _ctx(tmp_path, audit_vlm_windows=4, audit_vlm_window_px=200)
    pool = _FakePool([_ans("consistent", {"consistent": 2, "systematic_error": 1}, 0.6),
                      _ans("consistent", {"consistent": 2, "boundary_error": 1}, 0.7),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 3}, 0.9)])
    res = stages.stage_audit(ctx, *paths, pool=pool)
    p = res.data["pilot"]
    assert p["ran"] and p["sampled"] == 4 and p["answered"] == 4
    assert p["flagged"] == 0 and p["flagged_frac"] == 0.0
    assert p["split"] == 2 and p["unanimous_systematic"] == 0 and p["lost_votes"] == 0
    assert p["verdict_counts"] == {"consistent": 4, "systematic_error": 0,
                                   "boundary_error": 0, "unsure": 0}
    assert res.ok and not res.escalate and res.escalate_reason == ""
    assert "分歧 2（仅作信息）" in res.evidence
    assert res.confidence == round((0.6 + 0.7 + 0.9 + 0.9) / 4, 3)
    assert len(pool.seen) == 4 and "分割各类占比" in pool.seen[0]
    doc = json.loads(Path(res.data["audit"]).read_text())
    assert doc["pilot"]["results"][0]["split"] is True
    assert doc["pilot"]["results"][2]["unanimous"] is True and doc["escalate"] is False


def test_pilot_unanimous_systematic_error_escalates_below_flag_frac(tmp_path):
    """S05mg P02 (3/3 systematic_error, conf 0.96): one unanimous window is evidence
    even when the flagged fraction stays under audit_flag_frac."""
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    ctx = _ctx(tmp_path, audit_vlm_windows=6, audit_vlm_window_px=200)
    pool = _FakePool([_ans("systematic_error", {"systematic_error": 3}, 0.96, "1L"),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 3}, 0.9)])
    res = stages.stage_audit(ctx, *paths, pool=pool)
    p = res.data["pilot"]
    assert p["sampled"] == 6 and p["flagged"] == 1 and p["flagged_frac"] < 0.2
    assert p["unanimous_systematic"] == 1 and p["split"] == 0
    assert res.escalate and not res.ok
    assert "全票判 systematic_error" in res.escalate_reason
    assert "判分割有误" not in res.escalate_reason and "票数分歧" not in res.escalate_reason
    assert res.data["n_alarms"] == 0            # the detector itself was quiet
    doc = json.loads(Path(res.data["audit"]).read_text())
    assert doc["pilot"]["results"][0]["error_class"] == "1L"
    assert doc["pilot"]["results"][0]["unanimous"] is True


def test_pilot_flag_fraction_escalates_without_unanimity(tmp_path):
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    ctx = _ctx(tmp_path, audit_vlm_windows=4, audit_vlm_window_px=200)
    pool = _FakePool([_ans("systematic_error", {"systematic_error": 2, "consistent": 1}, 0.8, "TL"),
                      _ans("boundary_error", {"boundary_error": 2, "consistent": 1}, 0.7),
                      _ans("consistent", {"consistent": 3}, 0.9),
                      _ans("consistent", {"consistent": 2}, 0.9)])     # one vote lost
    res = stages.stage_audit(ctx, *paths, pool=pool)
    p = res.data["pilot"]
    assert p["flagged"] == 2 and p["flagged_frac"] == 0.5 and p["split"] == 2
    assert p["unanimous_systematic"] == 0 and p["lost_votes"] == 1
    assert res.escalate and "2/4 个窗口判分割有误" in res.escalate_reason
    assert "全票" not in res.escalate_reason
    doc = json.loads(Path(res.data["audit"]).read_text())
    assert doc["pilot"]["results"][3]["lost_votes"] == 1
    assert doc["pilot"]["results"][3]["split"] is False      # a lost vote is not dissent


def test_pilot_unanimous_consistent_passes_and_none_answers_escalate(tmp_path):
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    ctx = _ctx(tmp_path, audit_vlm_windows=3, audit_vlm_window_px=200)
    res = stages.stage_audit(ctx, *paths,
                             pool=_FakePool([_ans("consistent", {"consistent": 3}, 0.85)]))
    assert res.ok and not res.escalate and res.confidence == 0.85
    assert "VLM 试点 3/3" in res.evidence
    # the model never answered: that is escalated, not silently passed
    res = stages.stage_audit(ctx, *paths, pool=_FakePool([None]))
    assert res.escalate and "无有效返回" in res.escalate_reason
    assert res.confidence == 0.9 and res.data["pilot"]["answered"] == 0


def test_pilot_skipped_with_no_ai(tmp_path):
    photo = _textured()
    mask = _mask_for(photo)
    paths = _write(tmp_path, photo, mask)
    res = stages.stage_audit(_ctx(tmp_path, audit_vlm_windows=5, no_ai=True), *paths)
    assert res.ok and not res.escalate and res.data["pilot"]["ran"] is False
    assert "no-ai" in res.data["pilot"]["reason"] and "未运行" in res.evidence


# ------------------------------------------------------------------ orchestrator
def test_orchestrator_runs_five_stages_with_real_audit(tmp_path, monkeypatch):
    photo = _textured()
    mask = _mask_for(photo)
    mosaic = tmp_path / "in.png"
    Image.fromarray(photo).save(mosaic)
    (tmp_path / "w.pth").write_bytes(b"weights")

    def fake_segment(ctx, mosaic_path):
        outdir = ctx["work"] / "02_segment"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "mask.npy", mask)
        np.save(outdir / "valid.npy", np.ones(mask.shape, bool))
        return Result(ok=True, confidence=0.95, evidence="fake segment",
                      data={"mask": str(outdir / "mask.npy"),
                            "valid_mask": str(outdir / "valid.npy"),
                            "tiles": 1, "tile": 512, "overlap": 64,
                            "excluded_non_scan_pct": 0.0})

    seen = {}

    def fake_regions(ctx, mosaic_path, mask_path, valid_path):
        seen["mask_hash"] = np.load(mask_path).tobytes()
        (ctx["work"] / "03_regions").mkdir(parents=True, exist_ok=True)
        (ctx["work"] / "03_regions" / f"{ctx['sample']}_exclusions.json").write_text("{}")
        return Result(ok=True, data={"candidates": 0, "verdicts": [], "exclusion_mask": ""},
                      evidence="fake regions")

    def fake_stats(ctx, mask_path, valid_path, excl):
        (ctx["work"] / "04_stats").mkdir(parents=True, exist_ok=True)
        (ctx["work"] / "04_stats" / f"{ctx['sample']}_stats.json").write_text("{}")
        return Result(ok=True, data={"ratios_pct": {"1L": 80, "2L": 15, "TL": 5},
                                     "delta_from_region_exclusion_pp": {"1L": 0, "2L": 0, "TL": 0}},
                      evidence="fake stats")

    monkeypatch.setattr(stages, "stage_segment", fake_segment)
    monkeypatch.setattr(stages, "stage_regions", fake_regions)
    monkeypatch.setattr(stages, "stage_stats", fake_stats)

    cfg = {"work_root": str(tmp_path / "_work"), "weights": str(tmp_path / "w.pth"),
           "votes": 3, "audit_scale_div": 1, "audit_window": 32}
    samples = [{"name": "S1", "mosaic_in": str(mosaic)}]
    o = O.Orchestrator(cfg)
    plan = o.plan(samples)
    assert [l["stage"] for l in plan["samples"][0]["stages"]] == \
        ["stitch", "segment", "audit", "regions", "stats"]
    assert plan["samples"][0]["stages"][2]["agent"] == "segmentation_auditor"

    rep = o.run(samples)
    s = rep["samples"][0]
    assert O.Orchestrator.STAGES == ["stitch", "segment", "audit", "regions", "stats"]
    assert s["actions"] == {st: "run" for st in O.Orchestrator.STAGES}
    assert s["escalations"] == []
    man = json.loads(Path(s["manifest"]).read_text())
    au = man["stages"]["audit"]
    assert au["agent"] == "segmentation_auditor" and au["agent_kind"] == "hybrid"
    assert REGISTRY["segmentation_auditor"].kind == "hybrid"
    assert "待实现" not in REGISTRY["segmentation_auditor"].judgment
    assert au["decision"] == "accept" and au["result"]["n_alarms"] == 0
    assert Path(au["outputs"][0]).name == "S1_audit.json" and Path(au["outputs"][0]).exists()
    assert seen["mask_hash"] == mask.tobytes()          # downstream saw the same mask

    # second run: everything is cached, the audit included
    rep2 = O.Orchestrator(cfg).run(samples)
    assert rep2["samples"][0]["actions"] == {st: "cache" for st in O.Orchestrator.STAGES}
    # changing an audit parameter re-runs only the audit
    rep3 = O.Orchestrator({**cfg, "audit_var_thr": 5.0}).run(samples)
    acts = rep3["samples"][0]["actions"]
    assert acts["audit"] == "run" and acts["segment"] == "run"   # config_hash changed
    assert Path(rep3["samples"][0]["manifest"]).exists()


def test_orchestrator_escalates_on_alarm(tmp_path, monkeypatch):
    photo = _textured()
    x0, x1, y0, y1 = RECT
    photo[y0:y1, x0:x1] = 255
    mask = _mask_for(photo)
    mask[y0:y1, x0:x1] = 3
    mosaic = tmp_path / "in.png"
    Image.fromarray(photo).save(mosaic)
    (tmp_path / "w.pth").write_bytes(b"weights")

    def fake_segment(ctx, mosaic_path):
        outdir = ctx["work"] / "02_segment"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "mask.npy", mask)
        np.save(outdir / "valid.npy", np.ones(mask.shape, bool))
        return Result(ok=True, confidence=0.95, evidence="fake segment",
                      data={"mask": str(outdir / "mask.npy"),
                            "valid_mask": str(outdir / "valid.npy"), "tiles": 1,
                            "tile": 512, "overlap": 64, "excluded_non_scan_pct": 0.0})

    def fake_regions(ctx, *a):
        (ctx["work"] / "03_regions").mkdir(parents=True, exist_ok=True)
        (ctx["work"] / "03_regions" / f"{ctx['sample']}_exclusions.json").write_text("{}")
        return Result(ok=True, data={"candidates": 0, "verdicts": [], "exclusion_mask": ""})

    def fake_stats(ctx, *a):
        (ctx["work"] / "04_stats").mkdir(parents=True, exist_ok=True)
        (ctx["work"] / "04_stats" / f"{ctx['sample']}_stats.json").write_text("{}")
        return Result(ok=True, data={"ratios_pct": {}, "delta_from_region_exclusion_pp": {}})

    monkeypatch.setattr(stages, "stage_segment", fake_segment)
    monkeypatch.setattr(stages, "stage_regions", fake_regions)
    monkeypatch.setattr(stages, "stage_stats", fake_stats)
    cfg = {"work_root": str(tmp_path / "_work"), "weights": str(tmp_path / "w.pth"),
           "votes": 3, "audit_scale_div": 1, "audit_window": 32}
    rep = O.Orchestrator(cfg).run([{"name": "S1", "mosaic_in": str(mosaic)}])
    esc = rep["samples"][0]["escalations"]
    assert [e["stage"] for e in esc] == ["audit"]
    assert esc[0]["alarms"][0]["dominant_class"] == "TL"
    assert esc[0]["alarms"][0]["bbox"][1] <= x1 and esc[0]["alarms"][0]["bbox"][3] <= y1
    q = json.loads((Path(cfg["work_root"]) / "REVIEW" / "review_queue.json").read_text())
    assert q[0]["stage"] == "audit" and q[0]["alarms"][0]["area_px"] > 20000
    chk = {c["check"]: c for c in rep["consistency"]}
    assert not chk["No escalations are awaiting a human"]["pass"]
