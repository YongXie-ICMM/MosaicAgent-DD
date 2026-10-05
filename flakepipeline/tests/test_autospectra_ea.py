"""The E_A rule for substrates without an amplitude gap (2026-09-05 real-data run):
calibrate() records whether the A-exciton energy separates 1L from 3R; rule_assign()
uses it only where it does, escalates at the threshold, and flags a conflict with an
unambiguous amplitude. Model QC is off by default and halves the expected calls."""
import sys
from pathlib import Path

import numpy as np

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import autospectra as AS          # noqa: E402


def fit(split, aA, EA, aB=None):
    return dict(EA=EA, EB=EA + split / 1000, wA=0.03, wB=0.03, aA=aA, aB=aB or aA,
                split_meV=float(split), snr=8.0)


def sapphire_like():
    # amplitudes overlap (no gap), E_A separates with a 17 meV gap: 1L >= 1.874, 3R <= 1.857
    groups = {
        ("sap", "1L"): [fit(140, a, e) for a, e in zip([0.30, 0.36, 0.40], [1.874, 1.877, 1.880])],
        ("sap", "3R"): [fit(141, a, e) for a, e in zip([0.32, 0.38, 0.45], [1.850, 1.853, 1.857])],
        ("sap", "2H"): [fit(161, 0.35, 1.860)],
        # SiO2-like: amplitude gap, E_A ranges overlap
        ("sio2", "1L"): [fit(140, a, e) for a, e in zip([0.20, 0.24], [1.790, 1.810])],
        ("sio2", "3R"): [fit(141, a, e) for a, e in zip([0.50, 0.60], [1.797, 1.819])],
        ("sio2", "2H"): [fit(155, 0.55, 1.815)],
    }
    return AS.calibrate(groups)


def test_calibrate_records_ea_gap_only_where_it_exists():
    ref = sapphire_like()
    sap, sio2 = ref["ea_gap"]["sap"], ref["ea_gap"]["sio2"]
    assert sap["ok"] is True and abs(sap["gap_meV"] - 17.0) < 0.1
    assert abs(sap["threshold_eV"] - 1.8655) < 1e-4
    assert sio2["ok"] is False and sio2["gap_meV"] < 0
    assert ref["amp_gap"]["sap"]["ok"] is False and ref["amp_gap"]["sio2"]["ok"] is True


def test_ea_decides_on_nogap_substrate_and_amplitude_on_gap_substrate():
    ref = sapphire_like()
    # amplitude ratio ~1 (undecidable by amplitude), E_A says monolayer
    r = AS.rule_assign(fit(140, 0.36, 1.876), "sap", ref)
    assert r["label"] == "1L" and r["decided_by"] == "E_A" and r["flag"] == ""
    assert r["ea_margin_meV"] > 8
    # same amplitude, E_A says bilayer
    r = AS.rule_assign(fit(141, 0.36, 1.852), "sap", ref)
    assert r["label"] == "3R" and r["decided_by"] == "E_A" and r["flag"] == ""
    # SiO2: amplitude decides, E_A ignored even when it would say otherwise
    r = AS.rule_assign(fit(140, 0.21, 1.819), "sio2", ref)
    assert r["label"] == "1L" and r["decided_by"] == "amplitude" and r["ea_margin_meV"] is None


def test_ea_rule_escalates_at_threshold_and_on_conflict():
    ref = sapphire_like()
    thr = ref["ea_gap"]["sap"]["threshold_eV"]
    r = AS.rule_assign(fit(140, 0.36, thr + 0.001), "sap", ref)
    assert r["flag"] == "ea_ambiguous" and AS._rule_flag_reasons(r, False, "escalate")
    # E_A says 1L but the amplitude is far above the band (unambiguous bilayer amplitude)
    r = AS.rule_assign(fit(140, 0.80, 1.879), "sap", ref)
    assert r["label"] == "1L" and r["flag"] == "ea_amp_conflict"
    assert AS._rule_flag_reasons(r, True, "model")      # the conflict is never silenced by the model
    # 2H by splitting is untouched by the E_A rule
    r = AS.rule_assign(fit(161, 0.36, 1.879), "sap", ref)
    assert r["label"] == "2H" and r["decided_by"] == "splitting"


def test_no_ea_gap_record_keeps_the_old_nogap_behaviour():
    ref = sapphire_like()
    ref.pop("ea_gap")
    r = AS.rule_assign(fit(140, 0.36, 1.876), "sap", ref)
    assert r["decided_by"] == "amplitude" and r["flag"] == "amp_nogap_ambiguous"


def test_model_qc_off_by_default_halves_expected_calls(tmp_path):
    from test_autospectra import synth
    x, y = synth(1.88, 2.03)
    out = AS.analyze([{"id": "a", "substrate": "S", "x": x, "y": y}], AS.DEFAULT_REF,
                     tmp_path, no_ai=True, escalate_to_queue=False)
    assert out["gate"]["model_qc"] is False
    assert out["gate"]["model_calls_expected"] == out["gate"]["n_modelled"] * out["votes"]


def test_uncalibrated_substrate_is_flagged_not_guessed():
    ref = sapphire_like()
    r = AS.rule_assign(fit(155, 0.36, 1.86), "276nm", ref)          # a typo for a calibrated substrate
    assert r["flag"] == "substrate_uncalibrated" and r["confidence"] == 0.0
    assert AS._rule_flag_reasons(r, True, "model")                  # never silenced by the model path
    # the default ref has no per-substrate table: nothing to be "missing" from, no flag
    r = AS.rule_assign(fit(155, 0.36, 1.86), "anything", AS.DEFAULT_REF)
    assert r["flag"] == ""
