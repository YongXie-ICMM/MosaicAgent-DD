"""AutoSpectra after experiments_20260902/autospectra_kimi: the amplitude veto, the
per-substrate reference handed to the assigner, the model gate and the escalation
logic under the shared AgentPool contract (_tally / _n_votes / _split / None on a
tie). No network: the Kimi client and pool are faked."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import autospectra as AS          # noqa: E402
import region_agents as RA        # noqa: E402


def synth(EA, EB, aA=0.6, aB=0.5, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(1.55, 2.5, 1300)
    y = (0.15 + 0.25 * (x - 1.55) + AS.gauss(x, aA, EA, 0.03) + AS.gauss(x, aB, EB, 0.035)
         + rng.normal(0, noise, x.size))
    return x, y


def spectrum(sid, substrate, split_meV, aA, truth=None, seed=0):
    x, y = synth(1.850, 1.850 + split_meV / 1000, aA=aA, aB=0.8 * aA, seed=seed)
    return {"id": sid, "substrate": substrate, "label": truth, "x": x, "y": y}


# two substrates, threshold 150 meV, amplitude threshold 0.4; "gap" has a clean
# 1L/bilayer amplitude gap (SiO2-like), "nogap" has none (sapphire-like)
REF = {
    "split_threshold_meV": 148.0,                       # global: must never reach the assigner
    "split_threshold_by_substrate": {"gap": 150.0, "nogap": 150.0},
    "class_split_medians_meV": {"1L": 139.0, "3R": 139.0, "2H": 157.0},
    "class_split_medians_by_substrate": {"gap": {"1L": 140.0, "3R": 140.0, "2H": 160.0},
                                         "nogap": {"1L": 140.0, "3R": 141.0, "2H": 161.0}},
    "amp_1L_2L": {"gap": 0.4, "nogap": 0.4},
    "amp_gap": {"gap": {"ok": True, "max_1L_ratio": 0.6, "min_bilayer_ratio": 1.3},
                "nogap": {"ok": False, "max_1L_ratio": 1.02, "min_bilayer_ratio": 0.66}},
    "amp_ratio_class_medians_by_substrate": {"gap": {"1L": 0.5, "3R": 1.8, "2H": 2.0},
                                             "nogap": {"1L": 0.91, "3R": 1.1, "2H": 1.03}},
}


# ------------------------------------------------------------------ calibrate: the amplitude gap
def test_calibrate_records_amplitude_gap_only_where_classes_separate():
    def fits(split, amps):
        return [{"split_meV": split, "aA": a} for a in amps]
    groups = {
        # SiO2-like: 1L 0.2-0.25, 3R 0.6-0.65, 2H 0.6 -> gap
        ("gap", "1L"): fits(140, [0.20, 0.25]), ("gap", "3R"): fits(140, [0.60, 0.65]),
        ("gap", "2H"): fits(160, [0.60]),
        # sapphire-like: 1L up to 0.45, bilayers down to 0.30 -> no gap
        ("nogap", "1L"): fits(140, [0.30, 0.45]), ("nogap", "3R"): fits(141, [0.35, 0.50]),
        ("nogap", "2H"): fits(161, [0.30]),
    }
    ref = AS.calibrate(groups)
    assert ref["amp_gap"]["gap"]["ok"] is True
    assert ref["amp_gap"]["gap"]["max_1L_ratio"] < ref["amp_gap"]["gap"]["min_bilayer_ratio"]
    assert ref["amp_gap"]["nogap"]["ok"] is False
    assert ref["amp_gap"]["nogap"]["max_1L_ratio"] > ref["amp_gap"]["nogap"]["min_bilayer_ratio"]
    # the per-substrate class median ratios go into the reference table for the assigner
    med = ref["amp_ratio_class_medians_by_substrate"]
    assert set(med["gap"]) == {"1L", "3R", "2H"} and med["gap"]["1L"] < 1 < med["gap"]["3R"]
    assert ref["split_threshold_by_substrate"]["gap"] == 150.0
    # the reference stays JSON-serialisable (ref.json)
    json.dumps(ref)


# ------------------------------------------------------------------ rule: the amplitude veto
def test_amplitude_veto_relabels_2h_to_1l_only_on_gap_substrate():
    f = AS.fit_ab(*synth(1.850, 2.010, aA=0.2, aB=0.16))       # 2H by splitting, monolayer amplitude
    assert f and f["aA"] / 0.4 < 0.8
    r = AS.rule_assign(f, "gap", REF)
    assert r["label"] == "1L" and r["stacking"] == "2H" and r["vetoed"] is True
    assert "幅度否决" in r["reason"] and r["flag"] == ""
    assert r["confidence"] == 1.0                                 # ratio well below the ambiguity edge
    # same numbers on a substrate without a gap: no veto, no flag on the 2H side
    r2 = AS.rule_assign(f, "nogap", REF)
    assert r2["label"] == "2H" and r2["vetoed"] is False and r2["flag"] == ""
    # a substrate with a threshold but no amp_gap record is treated as no gap
    ref_old = {k: v for k, v in REF.items() if k != "amp_gap"}
    assert AS.rule_assign(f, "gap", ref_old)["label"] == "2H"
    # a real bilayer amplitude keeps 2H
    f2 = AS.fit_ab(*synth(1.850, 2.010, aA=0.6))
    r3 = AS.rule_assign(f2, "gap", REF)
    assert r3["label"] == "2H" and r3["vetoed"] is False and r3["amp_ratio"] > 1.2


def test_amplitude_veto_ambiguous_band_flags_instead_of_relabelling():
    f = AS.fit_ab(*synth(1.850, 2.010, aA=0.36, aB=0.3))        # ratio ~0.9: between the classes
    assert f and 0.8 < f["aA"] / 0.4 < 1.0
    r = AS.rule_assign(f, "gap", REF)
    assert r["label"] == "2H" and r["vetoed"] is False
    assert r["flag"] == "amp_veto_ambiguous" and "交人工" in r["flag_reason"]
    assert r["confidence"] < 1.0


def test_nogap_substrate_flags_undecidable_1l_3r():
    f = AS.fit_ab(*synth(1.850, 1.990, aA=0.4, aB=0.32))         # ratio ~1.0 on the sapphire-like substrate
    r = AS.rule_assign(f, "nogap", REF)
    assert r["label"] in ("1L", "3R") and r["flag"] == "amp_nogap_ambiguous"
    # the same ratio on the gap substrate is a plain decision
    assert AS.rule_assign(f, "gap", REF)["flag"] == ""
    # far from the band: no flag even without a gap
    f2 = AS.fit_ab(*synth(1.850, 1.990, aA=0.2, aB=0.16))
    assert AS.rule_assign(f2, "nogap", REF)["label"] == "1L"
    assert AS.rule_assign(f2, "nogap", REF)["flag"] == ""


def test_ref_for_prefers_the_substrate_values():
    R = AS.ref_for(REF, "gap")
    assert R["split_threshold_meV"] == 150.0 and R["class_split_medians_meV"]["2H"] == 160.0
    assert R["amp_threshold_1L_below_2L_above"] == 0.4 and R["amp_gap_ok"] is True
    # unknown substrate falls back to the global numbers and no amplitude threshold
    R2 = AS.ref_for(REF, "other")
    assert R2["split_threshold_meV"] == 148.0 and R2["amp_threshold_1L_below_2L_above"] is None
    assert R2["amp_gap_ok"] is False


# ------------------------------------------------------------------ a fake Kimi (no network)
class FakeRole:
    def __init__(self, name, system, schema_hint, max_tokens=2048):
        self.name, self.system, self.schema_hint, self.max_tokens = name, system, schema_hint, max_tokens


class FakeClient:
    available = True

    def __init__(self, cache_dir=None, **kw):
        self.cache_dir = cache_dir


class FakePool:
    """Replays canned answers keyed on (role, spectrum id); records every prompt."""
    answers: dict = {}
    calls: list = []

    def __init__(self, client, workers=6, votes=3):
        self.votes = votes

    def map(self, role, items, build, key="category", use_votes=True, verbose=True, decision_key=None):
        out = []
        for item in items:
            s, _ = item
            prompt, images = build(item)
            FakePool.calls.append({"role": role.name, "id": s["id"], "prompt": prompt, "key": key})
            out.append(FakePool.answers.get((role.name, s["id"])))
        return out


FAKE_KA = types.SimpleNamespace(Role=FakeRole, KimiClient=FakeClient, AgentPool=FakePool)


@pytest.fixture
def fake_kimi(monkeypatch):
    FakePool.answers = {}
    FakePool.calls = []
    monkeypatch.setattr(RA, "_import_kimi", lambda: (FAKE_KA, FP.parent))
    return FakePool


def qc(verdict, tally=None, n=3):
    return {"verdict": verdict, "confidence": 0.9, "reason": "r", "_tally": tally or {verdict: n}, "_n_votes": n}


def asg(label, tally=None, n=3, split=None):
    a = {"label": label, "confidence": 0.9, "reason": "r", "_tally": tally or {label: n}, "_n_votes": n}
    if split is not None:
        a["_split"] = split
    return a


def test_roles_budget_and_sapphire_statement():
    q, a = AS._roles(FAKE_KA)
    assert q.max_tokens == 16384 and a.max_tokens == 16384
    assert "蓝宝石" in a.system and "amp_threshold_1L_below_2L_above" in a.system
    assert "amp_1L_2L_this_substrate" not in a.system


# ------------------------------------------------------------------ analyze(): gate + escalation
def test_gate_models_only_near_threshold_and_never_overrides(tmp_path, fake_kimi):
    spectra = [
        spectrum("far2H", "gap", 160, 0.6, "2H", seed=1),         # margin +10: rule only
        spectrum("far1L", "gap", 140, 0.2, "1L", seed=2),         # margin -10: rule only
        spectrum("near_ok", "gap", 152, 0.6, "2H", seed=3),       # margin +2: modelled, agrees
        spectrum("near_dis", "gap", 148, 0.6, "3R", seed=4),      # margin -2: model says 1L -> escalate, label stays
        spectrum("near_lost", "gap", 152, 0.6, "2H", seed=5),     # one vote lost to HTTP: 投票不足
        spectrum("near_qcsplit", "gap", 152, 0.6, "2H", seed=6),  # QC split, majority fit_ok: no escalation
        spectrum("near_qcbad", "gap", 152, 0.6, "2H", seed=7),    # QC majority peak_missing: escalate
        spectrum("near_tie", "gap", 152, 0.6, "2H", seed=8),      # assigner tie -> None: escalate
        spectrum("near_split", "gap", 152, 0.6, "2H", seed=9),    # real 2-1 split: 票数分歧
        spectrum("near_unc", "gap", 152, 0.6, "2H", seed=10),     # model uncertain: escalate
        spectrum("veto_amb", "gap", 160, 0.36, "2H", seed=11),    # amplitude ambiguous: not modelled, escalated
        spectrum("nogap_amb", "nogap", 140, 0.4, "3R", seed=12),  # sapphire-like: straight to the human
    ]
    fake_kimi.answers = {
        ("spectrum_qc", "near_ok"): qc("fit_ok"), ("stacking_assigner", "near_ok"): asg("2H"),
        ("spectrum_qc", "near_dis"): qc("fit_ok"), ("stacking_assigner", "near_dis"): asg("1L"),
        ("spectrum_qc", "near_lost"): qc("fit_ok", {"fit_ok": 2}, n=2),
        ("stacking_assigner", "near_lost"): asg("2H", {"2H": 2}, n=2),
        ("spectrum_qc", "near_qcsplit"): qc("fit_ok", {"fit_ok": 2, "noisy": 1}),
        ("stacking_assigner", "near_qcsplit"): asg("2H"),
        ("spectrum_qc", "near_qcbad"): qc("peak_missing", {"peak_missing": 2, "fit_ok": 1}),
        ("stacking_assigner", "near_qcbad"): asg("2H"),
        ("spectrum_qc", "near_tie"): qc("fit_ok"), ("stacking_assigner", "near_tie"): None,
        ("spectrum_qc", "near_split"): qc("fit_ok"),
        ("stacking_assigner", "near_split"): asg("2H", {"2H": 2, "3R": 1}),
        ("spectrum_qc", "near_unc"): qc("fit_ok"), ("stacking_assigner", "near_unc"): asg("uncertain"),
    }
    out = AS.analyze(spectra, REF, tmp_path / "w", no_ai=False, votes=3, escalate_to_queue=False, model_qc=True)
    res = {r["id"]: r for r in out["results"]}
    assert out["ai"] is True and out["fitted"] == 12

    # gate bookkeeping in the results
    modelled = sorted({c["id"] for c in fake_kimi.calls})
    near = sorted(i for i in res if i.startswith("near_"))
    assert modelled == near
    assert out["gate"]["meV"] == 4.0 and out["gate"]["n_modelled"] == len(near)
    assert out["gate"]["model_calls_expected"] == len(near) * 3 * 2
    assert out["gate"]["nogap_policy"] == "escalate"
    for i in ("far2H", "far1L", "veto_amb", "nogap_amb"):
        assert res[i]["_ai"] is False and res[i]["qc"] is None and res[i]["llm"] is None
    assert "未调用模型" in res["far2H"]["note"] and res["far2H"]["escalate"] is False
    assert res["far1L"]["escalate"] is False
    for i in near:
        assert res[i]["_ai"] is True and res[i]["gated"] is True

    # the assigner never overrides the rule; disagreement escalates
    assert res["near_ok"]["label"] == "2H" and res["near_ok"]["escalate"] is False
    assert res["near_dis"]["label"] == "3R" and res["near_dis"]["escalate"] is True
    assert "模型判 1L，规则判 3R" in res["near_dis"]["escalate_reason"]
    # a vote lost to HTTP is reported as 投票不足, not as disagreement
    assert res["near_lost"]["escalate"] is True
    assert "投票不足 2/3" in res["near_lost"]["escalate_reason"]
    assert "票数分歧" not in res["near_lost"]["escalate_reason"]
    # QC split with a fit_ok majority does not escalate
    assert res["near_qcsplit"]["escalate"] is False
    # QC non-ok majority does
    assert res["near_qcbad"]["escalate"] is True and "peak_missing" in res["near_qcbad"]["escalate_reason"]
    # tie (None under the shared contract) and a real split both escalate, with distinct wording
    assert "无有效多数" in res["near_tie"]["escalate_reason"]
    assert "堆垛票数分歧" in res["near_split"]["escalate_reason"] and res["near_split"]["label"] == "2H"
    assert "uncertain" in res["near_unc"]["escalate_reason"]
    # rule-level flags reach the queue without a model call
    assert res["veto_amb"]["label"] == "2H" and res["veto_amb"]["escalate"] is True
    assert "模糊带" in res["veto_amb"]["escalate_reason"]
    assert res["nogap_amb"]["escalate"] is True and "无间隙" in res["nogap_amb"]["escalate_reason"]
    assert out["amp_veto"] == {"applied": {}, "ambiguous": {"gap": 1}}
    esc_ids = sorted(e["spectrum"] for e in out["escalations"])
    assert esc_ids == sorted(["near_dis", "near_lost", "near_qcbad", "near_tie", "near_split",
                              "near_unc", "veto_amb", "nogap_amb"])


def test_assigner_sees_the_substrate_reference_not_the_global(tmp_path, fake_kimi):
    spectra = [spectrum("near", "gap", 152, 0.6, "2H", seed=1)]
    fake_kimi.answers = {("spectrum_qc", "near"): qc("fit_ok"), ("stacking_assigner", "near"): asg("2H")}
    AS.analyze(spectra, REF, tmp_path / "w", no_ai=False, votes=3, escalate_to_queue=False, model_qc=True)
    prompts = [c["prompt"] for c in fake_kimi.calls if c["role"] == "stacking_assigner"]
    assert len(prompts) == 1
    table = json.loads(prompts[0].split("本衬底参考表：", 1)[1].split("。这片", 1)[0])
    assert table["split_threshold_meV_this_substrate"] == 150.0          # not the global 148
    assert table["class_split_medians_meV_this_substrate"] == {"1L": 140.0, "3R": 140.0, "2H": 160.0}
    assert table["amp_threshold_1L_below_2L_above"] == 0.4
    assert table["amp_ratio_class_medians_this_substrate"] == {"1L": 0.5, "3R": 1.8, "2H": 2.0}
    assert table["amp_discriminative_this_substrate"] is True
    assert table["substrate"] == "gap" and table["amp_ratio_this_spectrum"] > 1
    assert "amp_1L_2L_this_substrate" not in prompts[0] and "148" not in prompts[0]
    # the key the pool tallies on is the label field
    assert [c["key"] for c in fake_kimi.calls] == ["verdict", "label"]


def test_split_reference_between_substrates_is_per_substrate(tmp_path, fake_kimi):
    spectra = [spectrum("s_gap", "gap", 152, 0.6, "2H", seed=1),
               spectrum("s_nogap", "nogap", 148, 0.4, "3R", seed=2)]   # margin -2, ratio ~1: both gated
    fake_kimi.answers = {("spectrum_qc", "s_gap"): qc("fit_ok"), ("stacking_assigner", "s_gap"): asg("2H"),
                         ("spectrum_qc", "s_nogap"): qc("fit_ok"), ("stacking_assigner", "s_nogap"): asg("3R")}
    out = AS.analyze(spectra, REF, tmp_path / "w", no_ai=False, votes=3, escalate_to_queue=False)
    by_id = {c["id"]: c["prompt"] for c in fake_kimi.calls if c["role"] == "stacking_assigner"}
    assert '"amp_discriminative_this_substrate": true' in by_id["s_gap"]
    assert '"amp_discriminative_this_substrate": false' in by_id["s_nogap"]
    assert '"2H": 161.0' in by_id["s_nogap"] and '"2H": 160.0' in by_id["s_gap"]
    res = {r["id"]: r for r in out["results"]}
    # near-threshold on the no-gap substrate: modelled because of the margin, and
    # still escalated for the undecidable 1L/3R (default policy)
    assert res["s_nogap"]["_ai"] is True and res["s_nogap"]["escalate"] is True
    assert "无间隙" in res["s_nogap"]["escalate_reason"]


def test_nogap_policy_model_sends_ambiguous_to_the_models(tmp_path, fake_kimi):
    spectra = [spectrum("amb", "nogap", 140, 0.4, "3R", seed=1)]     # margin -10, ratio ~1.0
    fake_kimi.answers = {("spectrum_qc", "amb"): qc("fit_ok"), ("stacking_assigner", "amb"): asg("3R")}
    out = AS.analyze(spectra, REF, tmp_path / "w", no_ai=False, votes=3, escalate_to_queue=False,
                     nogap_policy="model")
    r = out["results"][0]
    assert r["_ai"] is True and r["gated"] is True and "无间隙" in r["gate_reason"]
    assert r["escalate"] is False                     # the model agreed: no escalation under this policy
    assert out["gate"]["nogap_policy"] == "model"
    # the default policy on the same spectrum: 0 calls, straight to the queue
    fake_kimi.calls = []
    out2 = AS.analyze(spectra, REF, tmp_path / "w2", no_ai=False, votes=3, escalate_to_queue=False)
    assert fake_kimi.calls == [] and out2["results"][0]["escalate"] is True
    with pytest.raises(ValueError):
        AS.analyze(spectra, REF, tmp_path / "w3", no_ai=True, nogap_policy="nope")


def test_gate_width_is_a_parameter(tmp_path, fake_kimi):
    spectra = [spectrum("m6", "gap", 156, 0.6, "2H", seed=1)]          # margin +6
    fake_kimi.answers = {("spectrum_qc", "m6"): qc("fit_ok"), ("stacking_assigner", "m6"): asg("2H")}
    out = AS.analyze(spectra, REF, tmp_path / "a", no_ai=False, escalate_to_queue=False, model_qc=True)
    assert fake_kimi.calls == [] and out["results"][0]["_ai"] is False
    out = AS.analyze(spectra, REF, tmp_path / "b", no_ai=False, escalate_to_queue=False, gate_meV=8.0, model_qc=True)
    assert len(fake_kimi.calls) == 2 and out["results"][0]["_ai"] is True and out["gate"]["meV"] == 8.0


def test_no_ai_escalates_what_the_models_would_have_seen(tmp_path):
    spectra = [
        spectrum("far2H", "gap", 160, 0.6, "2H", seed=1),
        spectrum("near", "gap", 152, 0.6, "2H", seed=2),
        spectrum("vetoed", "gap", 160, 0.2, "1L", seed=3),        # veto applied: relabelled, no escalation
        spectrum("veto_amb", "gap", 160, 0.36, "2H", seed=4),
        spectrum("nogap_amb", "nogap", 140, 0.4, "3R", seed=5),
    ]
    out = AS.analyze(spectra, REF, tmp_path / "w", no_ai=True, escalate_to_queue=False)
    res = {r["id"]: r for r in out["results"]}
    assert out["ai"] is False and all(r["_ai"] is False for r in out["results"])
    assert res["vetoed"]["label"] == "1L" and res["vetoed"]["escalate"] is False
    assert out["amp_veto"]["applied"] == {"gap": 1}
    assert sorted(e["spectrum"] for e in out["escalations"]) == ["near", "nogap_amb", "veto_amb"]
    assert "不到 4 meV" in res["near"]["escalate_reason"]
    assert out["agreement_with_labels"] == 1.0
    assert out["gate"]["n_modelled"] == 0 and out["gate"]["n_gated"] == 1
    r = AS.as_result(out)
    assert r.escalate and "3 条谱需要人看" in r.escalate_reason and "仅规则" in r.evidence
