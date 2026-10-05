"""Closed-form focus gate on the stitcher's tile triage (tiles.py + run_stitch.step_qc),
after experiment E (triage_prefilter, 2026-09-02).

Everything here is offline: Kimi is a fake pool that records what it was asked and hands
back scripted verdicts; no key, no network. What it pins down:

  * the gate's thresholds are the same numbers as flakepipeline/focus_metrics.FocusConfig
    (the 2026-09-02 calibration on the 920 Figure 3a tiles);
  * tiles.flag_tiles: reference = p75 of focus_score over tiles with contrast >= 4,
    score < 0.4 x ref -> "defocus", 0.4-0.6 x ref -> "focus_band"; the old MAD-z defocus
    flag on the 1/8 metric is gone; small-file / low-contrast / clipped stay; the whole-
    batch-soft warning (ref < 10) and the no-reference / NaN cases are reported, and the
    function is idempotent (main() and step_qc() both call it);
  * tiles.build_cache scores the decoded FULL frame with focus_metrics.focus_metrics on
    both branches (fresh decode and old cache) and persists the numbers in
    cache_dir/focus_index.json, so a cached run has them without touching the raw files;
  * run_stitch.step_qc three-way gate: "defocus" -> closed-form drop verdict without a
    model call (category defocus, keep False, confidence 0.8, reason with score and ref,
    _ai False); "focus_band" or any other flag -> ROLE_TILE_INSPECTOR with decision_key
    "keep" and score/ref/ratio in the prompt; unflagged -> never sent; ties -> kept and
    listed; state.json records which verdicts were closed-form; with no AI exactly the
    defocus-flagged tiles are dropped.

Run:  cd MosaicAgent && python3 -m pytest flakepipeline/tests -q
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tiles as T                              # noqa: E402
import run_stitch as RS                        # noqa: E402
from flakepipeline import focus_metrics as A  # noqa: E402


# ------------------------------------------------------------------ helpers
def _tile(i, focus_score, contrast=20.0, nbytes=4_800_000, std=20.0, focus=30.0,
          clip_frac=0.0, key=None, col=18, direction="down"):
    return T.Tile(col=col, col_idx=0, direction=direction, order=i, key=float(key if key is not None else i + 1),
                  name=f"{i + 1:04d}.png", zip_path="", inner="", nominal_row=i, nbytes=nbytes,
                  mean=120.0, std=std, focus=focus, clip_frac=clip_frac,
                  focus_score=focus_score, contrast=contrast)


def _texture(seed, h=180, w=320):
    """Blocky random shapes with sharp edges (a stand-in for triangles on a substrate)."""
    rng = np.random.default_rng(seed)
    low = rng.normal(size=(h // 12 + 1, w // 12 + 1))
    low = np.kron(low, np.ones((12, 12)))[:h, :w]
    img = np.where(low > 0.3, 170.0, 120.0).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _blurred(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


# ------------------------------------------------------------------ thresholds
def test_gate_thresholds_match_autoscan_config():
    cfg = A.FocusConfig()
    assert T.FOCUS_RATIO_DEFOCUS == cfg.ratio == 0.4
    assert T.FOCUS_MIN_CONTRAST == cfg.min_contrast == 4.0
    assert T.FOCUS_REF_PERCENTILE == cfg.ref_percentile == 75.0
    assert T.FOCUS_MIN_REF_TILES == cfg.min_ref_tiles == 3
    assert T.FOCUS_REF_WARN_BELOW == cfg.ref_warn_below == 10.0
    assert T.FOCUS_RATIO_BAND == 0.6              # the band top has no FocusConfig twin
    assert "defocus" in T.FOCUS_FLAGS and "focus_band" in T.FOCUS_FLAGS


# ------------------------------------------------------------------ flag_tiles
def test_flag_tiles_ratio_rule_on_full_frame_score():
    # 25 textured tiles: 20 sharp ones at 20..39 and 5 soft ones below 20, so the sorted
    # p75 (index 18 of 25) is exactly 33.0 -> thr 0.4x = 13.2, band top 0.6x = 19.8.
    # The reference is over ALL textured tiles, the soft ones included (that is why it
    # is the p75 and not the median).
    ts = [_tile(i, 20.0 + i) for i in range(20)]
    ts += [_tile(20, 5.0),                       # 0.15 x ref -> defocus
           _tile(21, 13.1),                      # 0.397 x ref -> defocus (just under)
           _tile(22, 13.3),                      # 0.403 x ref -> focus_band (just over)
           _tile(23, 19.7),                      # 0.597 x ref -> focus_band
           _tile(24, 19.9)]                      # 0.603 x ref -> nothing
    # the 1/8 metric must NOT drive the defocus flag any more: give the sharpest tile a
    # ridiculously low 1/8 Laplacian variance and it still gets no focus flag
    ts[19].focus = 0.01
    qc = T.flag_tiles(ts)
    assert qc["focus_ref"] == pytest.approx(33.0) and qc["focus_ref_n"] == 25
    assert [t.name for t in qc["defocus"]] == ["0021.png", "0022.png"]
    assert [t.name for t in qc["focus_band"]] == ["0023.png", "0024.png"]
    assert ts[24].flags == [] and ts[19].flags == []
    assert all("defocus" not in t.flags for t in ts[:20])
    assert qc["warnings"] == [] and qc["low_texture"] == []
    assert {t.tid for t in qc["flagged"]} == {t.tid for t in qc["defocus"] + qc["focus_band"]}


def test_flag_tiles_keeps_statistical_flags_and_is_idempotent():
    ts = [_tile(i, 20.0) for i in range(30)]
    ts[3].nbytes = 1_000_000                     # far below the 4.8 MB crowd -> small-file
    ts[4].std = 1.0                              # -> low-contrast (MAD-z on std)
    ts[5].clip_frac = 0.5                        # -> clipped
    ts[6] = _tile(6, 20.0, key=6.5)              # -> irregular-index
    ts[7].flags.append("conflict")               # not ours: must survive re-flagging
    qc = T.flag_tiles(ts)
    assert ts[3].flags == ["small-file"] and ts[4].flags == ["low-contrast"]
    assert ts[5].flags == ["clipped"] and ts[6].flags == ["irregular-index"]
    assert ts[7].flags == ["conflict"]
    assert {t.name for t in qc["flagged"]} == {"0004.png", "0005.png", "0006.png", "0007.png", "0008.png"}
    # main() and step_qc() both call flag_tiles: the second call must not double the labels
    T.flag_tiles(ts)
    assert ts[3].flags == ["small-file"] and ts[7].flags == ["conflict"]


def test_flag_tiles_reference_excludes_low_contrast_and_warns_when_batch_is_soft():
    # low-contrast frames (empty shutter frames) never enter the reference pool and are
    # never called defocused, however low their score
    ts = [_tile(i, 20.0) for i in range(10)] + [_tile(10, 0.5, contrast=1.0)]
    qc = T.flag_tiles(ts)
    assert qc["focus_ref"] == pytest.approx(20.0) and qc["focus_ref_n"] == 10
    assert ts[10].flags == ["low_texture"]       # nominated as an empty frame, not as defocus
    assert qc["low_texture"] == [ts[10]] and qc["defocus"] == []
    # whole batch soft: reference below 10 -> the warning ported from check_focus_folder
    ts = [_tile(i, 6.0 + 0.1 * i) for i in range(10)]
    qc = T.flag_tiles(ts)
    assert qc["focus_ref"] < T.FOCUS_REF_WARN_BELOW
    assert any("整批" in w and "低于 10" in w for w in qc["warnings"]), qc["warnings"]
    assert qc["defocus"] == []                   # relative rule is silent on a uniformly soft batch


def test_flag_tiles_without_reference_or_scores_disables_focus_flags_with_warning():
    # fewer than 3 textured tiles: no reference, no focus flags, one explicit warning
    ts = [_tile(0, 20.0), _tile(1, 1.0), _tile(2, 20.0, contrast=2.0)]
    qc = T.flag_tiles(ts)
    assert qc["focus_ref"] is None and qc["defocus"] == [] and qc["focus_band"] == []
    assert ts[1].flags == [] and ts[2].flags == ["low_texture"]
    assert any("没有对焦参考值" in w for w in qc["warnings"])
    # NaN scores (old cache, backend unavailable): those tiles are skipped and reported
    ts = [_tile(i, 20.0) for i in range(6)] + [_tile(6, float("nan")), _tile(7, 2.0)]
    qc = T.flag_tiles(ts)
    assert qc["focus_ref"] == pytest.approx(20.0)
    assert [t.name for t in qc["defocus"]] == ["0008.png"] and ts[6].flags == []
    assert any("1/8 张瓦片没有全帧对焦分数" in w for w in qc["warnings"]), qc["warnings"]


# ------------------------------------------------------------------ build_cache
def test_build_cache_scores_full_frame_and_persists_index(tmp_path):
    sharp = _texture(0)
    blurred = _blurred(sharp, 6.0)
    flat = np.full_like(sharp, 128)
    d = tmp_path / "data" / "L01_down"
    d.mkdir(parents=True)
    frames = {"0001.png": sharp, "0002.png": blurred, "0003.png": flat}
    for n, im in frames.items():
        assert cv2.imwrite(str(d / n), im)
    ts = [T.Tile(col=1, col_idx=0, direction="down", order=i, key=float(i + 1), name=n,
                 zip_path=str(d), inner=n, nominal_row=i, nbytes=(d / n).stat().st_size)
          for i, n in enumerate(frames)]
    cache = tmp_path / "cache"

    # 1. fresh decode: scores come from the full frame and equal autoscan's own numbers
    info = T.build_cache(ts, str(cache), scale_div=4, workers=2)
    assert info["cached"] == 3 and info["rescored"] == 0 and info["total"] == 3
    assert info["focus_backend"] is True
    for t, im in zip(ts, frames.values()):
        m = A.focus_metrics(im, A.FocusConfig())
        assert t.focus_score == pytest.approx(m["score"], rel=1e-6)
        assert t.contrast == pytest.approx(m["contrast"], rel=1e-6)
        assert math.isfinite(t.focus) and math.isfinite(t.mean)
        assert T.load_small(str(cache), t).shape == (180 // 4, 320 // 4, 3)
    assert ts[0].focus_score > 3 * ts[1].focus_score > 0     # sharp >> blurred
    assert ts[2].contrast < T.FOCUS_MIN_CONTRAST             # the flat frame has no texture
    idx = json.loads((cache / T.FOCUS_INDEX).read_text(encoding="utf-8"))
    keys = [str(Path(T.cache_path(str(cache), t)).relative_to(cache)) for t in ts]
    assert set(idx) == set(keys) == {"L01_down/00001.00.png", "L01_down/00002.00.png",
                                     "L01_down/00003.00.png"}
    for k, t in zip(keys, ts):
        e = idx[k]
        assert e["focus_score"] == pytest.approx(t.focus_score) and e["src_w"] == 320
        assert {"mean", "std", "focus", "clip_frac", "contrast"} <= set(e)

    # 2. cached run with the index: nothing is decoded -- prove it by removing the raw
    #    files; the numbers still come back and nothing is flagged unreadable
    first = [(t.focus_score, t.contrast, t.focus, t.mean, t.std, t.clip_frac) for t in ts]
    for n in frames:
        (d / n).unlink()
    ts2 = [T.Tile(**{**t.__dict__, "flags": []}) for t in ts]
    for t in ts2:
        t.focus_score = t.contrast = t.focus = float("nan")
    info = T.build_cache(ts2, str(cache), scale_div=4, workers=2)
    assert info["cached"] == 0 and info["rescored"] == 0
    assert [(t.focus_score, t.contrast, t.focus, t.mean, t.std, t.clip_frac) for t in ts2] == \
        pytest.approx(first)
    assert all(t.flags == [] for t in ts2)

    # 3. old cache without an index (pre-2026-09-02 work dirs): the full frame is scored
    #    again from the raw file, without rewriting the cached small image
    for n, im in frames.items():
        assert cv2.imwrite(str(d / n), im)
    (cache / T.FOCUS_INDEX).unlink()
    small_path = Path(T.cache_path(str(cache), ts[0]))
    small_before = small_path.read_bytes()
    ts3 = [T.Tile(**{**t.__dict__, "flags": []}) for t in ts]
    for t in ts3:
        t.focus_score = t.contrast = float("nan")
    info = T.build_cache(ts3, str(cache), scale_div=4, workers=2)
    assert info["cached"] == 0 and info["rescored"] == 3
    assert [t.focus_score for t in ts3] == pytest.approx([f[0] for f in first])
    assert [t.contrast for t in ts3] == pytest.approx([f[1] for f in first])
    assert small_path.read_bytes() == small_before
    assert (cache / T.FOCUS_INDEX).exists()

    # 4. the gate on these three: the flat frame is low_texture (excluded from the
    #    reference); with only two textured tiles there is no reference -> no defocus,
    #    explicit warning
    qc = T.flag_tiles(ts3)
    assert qc["focus_ref"] is None and any("没有对焦参考值" in w for w in qc["warnings"])
    assert ts3[2].flags == ["low_texture"] and qc["defocus"] == []


def test_full_frame_focus_degrades_to_nan_when_backend_missing(monkeypatch):
    monkeypatch.setattr(T, "_FOCUS_BACKEND", False)
    m = T.full_frame_focus(np.zeros((8, 12, 3), np.uint8))
    assert math.isnan(m["focus_score"]) and math.isnan(m["contrast"])
    monkeypatch.setattr(T, "_FOCUS_BACKEND", None)
    assert T._focus_backend() and math.isfinite(T.full_frame_focus(_texture(1))["focus_score"])


# ------------------------------------------------------------------ step_qc
class FakeClient:
    def __init__(self, available=True):
        self.available = available
        self.model, self.usage = "fake-kimi", "no calls"

    def chat(self, *a, **k):
        raise AssertionError("Kimi must not be called in tests")


class FakePool:
    """Records every map() call; verdicts are scripted per tile id (None = tie)."""

    def __init__(self, verdicts, votes=3, available=True):
        self.client = FakeClient(available)
        self.votes = votes
        self.verdicts = verdicts
        self.maps = []

    def map(self, role, items, build, key="category", use_votes=True, verbose=True,
            decision_key=None):
        prompts = [build(t)[0] for t in items]
        self.maps.append({"role": role.name, "tids": [t.tid for t in items], "key": key,
                          "decision_key": decision_key, "prompts": prompts})
        return [self.verdicts.get(t.tid) for t in items]

    def vote(self, *a, **k):
        raise AssertionError("step_qc must go through map()")


def _scan_like_tiles():
    """30 sharp textured tiles at score 40 (so the p75 stays 40.0 whatever the 7 case
    tiles below do: thr 16.0, band top 24.0) plus the cases of the gate."""
    ts = [_tile(i, 40.0) for i in range(30)]
    ts.append(_tile(30, 5.0))                                  # defocus (0.125 x ref)
    ts.append(_tile(31, 12.0))                                 # defocus (0.30 x ref)
    ts.append(_tile(32, 12.0, nbytes=1_000_000))               # defocus AND small-file -> still closed-form
    ts.append(_tile(33, 17.0))                                 # focus_band (0.425 x ref)
    ts.append(_tile(34, 22.0))                                 # focus_band (0.55 x ref)
    ts.append(_tile(35, 40.0, nbytes=1_000_000))               # sharp but small-file -> model
    ts.append(_tile(36, 40.0, clip_frac=0.6))                  # clipped -> model
    return ts


DEFOCUS = ("0031.png", "0032.png", "0033.png")
TO_MODEL = ("0034.png", "0035.png", "0036.png", "0037.png")


@pytest.fixture
def qc_env(monkeypatch, tmp_path):
    monkeypatch.setattr(RS.T, "load_small", lambda cache_dir, t: np.zeros((27, 48, 3), np.uint8))
    return tmp_path


def test_step_qc_three_way_gate(qc_env):
    ts = _scan_like_tiles()
    tids = {t.name: t.tid for t in ts}
    verdicts = {
        tids["0034.png"]: {"category": "defocus", "keep": False, "confidence": 0.9, "reason": "糊",
                           "_tally": {"defocus": 3}, "_n_votes": 3, "_decision_tally": {"false": 3},
                           "_split": False},
        tids["0035.png"]: {"category": "normal", "keep": True, "confidence": 0.7, "reason": "清",
                           "_tally": {"normal": 2, "defocus": 1}, "_n_votes": 3,
                           "_decision_tally": {"true": 2, "false": 1}, "_split": True},
        tids["0036.png"]: {"category": "sample_edge", "keep": True, "confidence": 0.95, "reason": "边",
                           "_tally": {"sample_edge": 3}, "_n_votes": 3, "_split": False},
        # 0037.png (clipped): a tie -> None from map()
    }
    pool = FakePool(verdicts, votes=3)
    st = RS.State(qc_env / "state.json")
    drop = RS.step_qc(ts, str(qc_env / "cache"), pool, st, votes=3)

    # exactly one map() call, only the non-defocus flagged tiles, keep-majority requested
    assert len(pool.maps) == 1
    m = pool.maps[0]
    assert m["role"] == "tile_inspector" and m["key"] == "category" and m["decision_key"] == "keep"
    assert m["tids"] == [tids[n] for n in TO_MODEL]
    # the prompt carries score, ref and ratio next to the old statistics and the labels
    ref = 40.0
    assert float(np.percentile([t.focus_score for t in ts], 75)) == ref
    p = m["prompts"][0]
    assert "score = 17.00" in p and "ref = 40.00" in p and "score/ref = 0.42" in p
    assert "focus_band" in p and "Laplacian" in p and "文件大小" in p
    assert "small-file" in m["prompts"][2] and "clipped" in m["prompts"][3]

    # closed-form verdicts for the three defocus tiles, no model involved
    q = st.d["qc"]
    cf = [tids[n] for n in DEFOCUS]
    assert q["closed_form"] == sorted(cf)
    for tid in cf:
        v = q["verdicts"][tid]
        assert v["category"] == "defocus" and v["keep"] is False and v["confidence"] == 0.8
        assert v["_ai"] is False and "闭式" in v["reason"]
        assert "40.00" in v["reason"] and "0.4" in v["reason"]
    assert "5.00" in q["verdicts"][tids["0031.png"]]["reason"]
    assert "比值 0.12" in q["verdicts"][tids["0031.png"]]["reason"]
    # model verdicts are marked _ai True; the tie is kept and listed, the split is listed
    assert q["verdicts"][tids["0034.png"]]["_ai"] is True
    assert q["no_verdict"] == [tids["0037.png"]] and q["split"] == [tids["0035.png"]]
    assert tids["0037.png"] not in q["verdicts"]
    # drop = closed-form defocus + the model's one drop; sharp unflagged tiles never appear
    assert drop == set(cf) | {tids["0034.png"]}
    assert q["drop"] == sorted(drop)
    for t in ts[:30]:
        assert t.tid not in q["verdicts"] and t.tid not in m["tids"]
    # the guardrail numbers land in state.json
    f = q["focus"]
    assert f["ref"] == pytest.approx(ref) and f["ref_n"] == 37
    assert f["n_defocus"] == 3 and f["n_focus_band"] == 2 and f["n_low_texture"] == 0
    assert f["n_nominated"] == 7 and f["n_to_model"] == 4 and f["n_calls"] == 12 and f["votes"] == 3
    assert f["thresholds"]["ratio_defocus"] == 0.4 and f["thresholds"]["ratio_band"] == 0.6
    assert st.done("qc")
    saved = json.loads((qc_env / "state.json").read_text(encoding="utf-8"))
    assert saved["qc"]["closed_form"] == sorted(cf)


def test_step_qc_no_ai_drops_exactly_the_defocus_tiles(qc_env):
    ts = _scan_like_tiles()
    tids = {t.name: t.tid for t in ts}
    st = RS.State(qc_env / "state.json")
    drop = RS.step_qc(ts, str(qc_env / "cache"), None, st, votes=3)
    cf = {tids[n] for n in DEFOCUS}
    assert drop == cf
    q = st.d["qc"]
    assert q["closed_form"] == sorted(cf)
    assert all(q["verdicts"][tid]["_ai"] is False for tid in cf)
    # the other nominated tiles are kept, not silently: they are listed for a human
    assert q["no_verdict"] == sorted(tids[n] for n in TO_MODEL)
    assert q["focus"]["n_calls"] == 0
    # a pool whose client has no key behaves the same as no pool
    pool = FakePool({}, available=False)
    st2 = RS.State(qc_env / "state2.json")
    assert RS.step_qc(ts, str(qc_env / "cache"), pool, st2, votes=3) == cf
    assert pool.maps == []


def test_step_qc_without_reference_sends_flagged_tiles_and_drops_nothing_closed_form(qc_env):
    # two textured tiles only: no reference -> no defocus flag -> nothing is closed-form;
    # the empty frames (low_texture) and the small-file nomination still go to the model
    ts = [_tile(0, 20.0), _tile(1, 20.0)]
    ts += [_tile(2 + i, 0.5, contrast=1.0) for i in range(4)]        # empty frames
    ts.append(_tile(6, float("nan"), nbytes=1_000_000))              # no score, small-file
    pool = FakePool({ts[2].tid: {"category": "blank_frame", "keep": False, "confidence": 0.9,
                                 "reason": "空"}})
    st = RS.State(qc_env / "state.json")
    drop = RS.step_qc(ts, str(qc_env / "cache"), pool, st, votes=3)
    q = st.d["qc"]
    assert q["closed_form"] == [] and drop == {ts[2].tid}
    assert q["verdicts"][ts[2].tid]["_ai"] is True
    m = pool.maps[0]
    assert m["tids"] == [t.tid for t in ts[2:]]
    assert "low_texture" in m["prompts"][0] and "没有参考值" in m["prompts"][0]
    assert "全帧对焦分数不可用" in m["prompts"][4] and "small-file" in m["prompts"][4]
    ws = q["focus"]["warnings"]
    assert any("没有对焦参考值" in w for w in ws) and any("没有全帧对焦分数" in w for w in ws)
    assert q["focus"]["n_low_texture"] == 4 and q["focus"]["ref"] is None
