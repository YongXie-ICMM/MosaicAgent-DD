"""AutoSpectra without a model: the closed-form fit recovers synthetic peaks, the
rule assigns from the splitting, calibration sets the thresholds, and an unfit
spectrum is escalated to the review queue."""
import json
import sys
from pathlib import Path

import numpy as np

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import autospectra as AS          # noqa: E402
import review as RV               # noqa: E402


def synth(EA, EB, aA=0.6, aB=0.5, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(1.55, 2.5, 1300)
    y = (0.15 + 0.25 * (x - 1.55) + AS.gauss(x, aA, EA, 0.03) + AS.gauss(x, aB, EB, 0.035)
         + rng.normal(0, noise, x.size))
    return x, y


def test_fit_recovers_peaks():
    x, y = synth(1.850, 2.010)
    f = AS.fit_ab(x, y)
    assert f and abs(f["EA"] - 1.850) < 0.002 and abs(f["EB"] - 2.010) < 0.002
    assert f["snr"] > 20 and 155 < f["split_meV"] < 165


def test_unfit_on_noise():
    rng = np.random.default_rng(1)
    x = np.linspace(1.55, 2.5, 1300)
    y = 0.2 + rng.normal(0, 0.05, x.size)
    assert AS.fit_ab(x, y) is None


def test_rule_and_calibration():
    ref = {"split_threshold_meV": 150.0, "amp_1L_2L": {"S": 0.4}}
    f2h = AS.fit_ab(*synth(1.850, 2.010, aA=0.6))
    f3r = AS.fit_ab(*synth(1.850, 1.990, aA=0.6))
    f1l = AS.fit_ab(*synth(1.850, 1.990, aA=0.25, aB=0.2))
    assert AS.rule_assign(f2h, "S", ref)["label"] == "2H"
    assert AS.rule_assign(f3r, "S", ref)["label"] == "3R"
    assert AS.rule_assign(f1l, "S", ref)["label"] == "1L"
    assert AS.rule_assign(f1l, "other", ref)["label"] == "3R-or-1L"
    cal = AS.calibrate({("S", "2H"): [f2h], ("S", "3R"): [f3r], ("S", "1L"): [f1l]})
    assert 148 < cal["split_threshold_meV"] < 152
    assert 0.25 < cal["amp_1L_2L"]["S"] < 0.6


def test_analyze_no_ai_escalates_unfit(tmp_path):
    rng = np.random.default_rng(2)
    x = np.linspace(1.55, 2.5, 1300)
    spectra = [
        {"id": "a", "substrate": "S", "label": "2H", "x": x, "y": synth(1.850, 2.010)[1]},
        {"id": "b", "substrate": "S", "label": "3R", "x": x, "y": synth(1.850, 1.990, seed=3)[1]},
        {"id": "c", "substrate": "S", "label": "1L", "x": x, "y": synth(1.850, 1.990, aA=0.25, aB=0.2, seed=4)[1]},
        {"id": "noise", "substrate": "S", "label": "1L", "x": x, "y": 0.2 + rng.normal(0, 0.05, x.size)},
    ]
    ref = {"split_threshold_meV": 150.0, "amp_1L_2L": {"S": 0.4}}
    out = AS.analyze(spectra, ref, tmp_path / "w", no_ai=True)
    assert out["fitted"] == 3 and out["ai"] is False
    assert out["agreement_with_labels"] == 1.0
    assert [e["spectrum"] for e in out["escalations"]] == ["noise"]
    q = json.loads(RV.queue_path(tmp_path / "w").read_text())
    assert q[0]["stage"] == "autospectra" and q[0]["spectrum"] == "noise"
    res = AS.as_result(out)
    assert res.escalate and "1 条谱需要人看" in res.escalate_reason
    assert (tmp_path / "w" / "autospectra_results.json").exists()
