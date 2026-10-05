#!/usr/bin/env python3
"""Optical-spectrum fitting, calibrated rules and optional model cross-checks.

A deterministic two-Gaussian model with a quadratic background estimates exciton
features. Supplied substrate-specific calibration rules produce provisional
labels. Optional model review is limited to gated cases and does not overwrite
the numerical rule's label; unresolved cases are recorded for human review.
A reference label supplied by the user is not automatically physical ground truth.

The source package supplies analysis code, not an independently validated
stacking reference or completed model-comparison experiment. Use trusted pickle
inputs only, or import CSV/Excel exports through spectra_ingest.py."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from agents import Result            # noqa: E402
import review as RV                  # noqa: E402

# ------------------------------------------------------------------ 1. closed-form fit
NLO, NHI = 1.72, 2.14                # fit window, eV (fit_ab.py)


def gauss(x, a, c, w):
    return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))


def model2(x, a1, c1, w1, a2, c2, w2, q0, q1, q2):
    return gauss(x, a1, c1, w1) + gauss(x, a2, c2, w2) + q0 + q1 * x + q2 * x * x


def fit_ab(x, y):
    """A- and B-exciton energies from one DR spectrum. Returns None when the fit
    fails its own guards (amplitude below 2.5x residual noise, or a centre railed
    against a bound) -- the same guards as fit_ab.py."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > NLO) & (x < NHI)
    xf, yf = x[m], y[m]
    if len(xf) < 150:
        return None
    ys = savgol_filter(yf, 21, 3)
    amp = ys.max() - ys.min()
    p0 = [0.3 * amp, 1.86, 0.03, 0.3 * amp, 2.00, 0.035, ys.min(), 0, 0]
    lo = [0.001 * amp, 1.76, 0.012, 0.001 * amp, 1.90, 0.012, -np.inf, -np.inf, -np.inf]
    hi = [5 * amp, 1.92, 0.07, 5 * amp, 2.10, 0.08, np.inf, np.inf, np.inf]
    try:
        popt, _ = curve_fit(model2, xf, yf, p0=p0, bounds=(lo, hi), maxfev=30000)
    except Exception:
        return None
    a1, c1, w1, a2, c2, w2 = popt[:6]
    noise = float(np.std(yf - model2(xf, *popt)))
    if a1 < 2.5 * noise or a2 < 2.5 * noise:
        return None
    if min(c1 - 1.76, 1.92 - c1) < 0.008 or min(c2 - 1.90, 2.10 - c2) < 0.008:
        return None
    return dict(EA=float(c1), EB=float(c2), wA=float(w1), wB=float(w2),
                aA=float(a1), aB=float(a2), noise=noise,
                split_meV=float(1000 * (c2 - c1)), snr=float(min(a1, a2) / noise),
                popt=[float(v) for v in popt])


# ------------------------------------------------------------------ 2. the rule
DEFAULT_REF = {
    # B-A splitting: ~160 meV for 2H bilayers, ~140 meV for 3R bilayers and
    # monolayers (paper, Figure 4e). The threshold is the midpoint; --calibrate
    # replaces it with the midpoint of the class means in the labelled data.
    "split_threshold_meV": 150.0,
    # A-peak amplitude that separates a monolayer from a bilayer, per substrate.
    # Bilayers are roughly twice as bright; the value is the geometric mean of the
    # 1L and 3R medians from --calibrate. None = layer number not decided by DR.
    "amp_1L_2L": {},
    # Per substrate: does the labelled data show a clean amplitude gap between
    # monolayers and bilayers (max 1L ratio < min bilayer ratio)? Only then may the
    # amplitude veto a 2H splitting. Filled by --calibrate; absent = no gap (safe).
    "amp_gap": {},
    # Per substrate: does the A-exciton energy itself separate 1L from 3R (bilayers are
    # red-shifted by interlayer coupling)? Found on the 2026-09-05 real-data run: on
    # sapphire 1L >= 1.8737 eV and 3R <= 1.8564 eV (17 meV gap, class widths ~3 meV),
    # while on both SiO2 substrates the ranges overlap. Only where the calibration
    # shows such a gap may E_A decide 1L/3R. Filled by --calibrate; absent = no gap.
    "ea_gap": {},
    "notes": "defaults; run --calibrate on labelled spectra to fit these",
}

# amplitude ratio = aA / amp_1L_2L[substrate]
# A 2H-by-splitting spectrum on a gap substrate with ratio < 1 is a veto candidate;
# above AMP_VETO_AMBIGUOUS the amplitude evidence is too weak to relabel silently,
# so the spectrum is escalated instead (labelled data: 276 nm 1L max 0.59, bilayer
# min 1.26; 67 nm 0.67 vs 1.10 -- nothing labelled falls inside the band).
AMP_VETO_AMBIGUOUS = 0.8
# On a substrate without an amplitude gap (sapphire: 1L 0.67-1.02, 3R 0.81-1.22,
# 2H 0.66-1.10) a 1L/3R decision inside this band is undecidable from DR alone.
AMP_NOGAP_BAND = (0.8, 1.25)
# E_A rule (no-amplitude-gap substrates only): the calibrated 1L-3R gap must be at
# least this wide to be used, and a decision closer than EA_AMBIGUOUS_MEV to the
# threshold (about one class width) is escalated instead of assigned.
EA_GAP_MIN_MEV = 8.0
EA_AMBIGUOUS_MEV = 3.0
# model gate: spectra with |B-A - threshold| below this go to the models
DEFAULT_GATE_MEV = 4.0
NOGAP_POLICIES = ("escalate", "model")


def ref_for(ref: dict, substrate: str) -> dict:
    """The per-substrate reference the rule uses. build_asg() hands the assigner
    exactly this, so the model can never be judged against a different threshold
    than the rule (the global 148.8 vs the 276 nm 140.8 bug of experiment B)."""
    thr = float((ref.get("split_threshold_by_substrate") or {}).get(
        substrate, ref.get("split_threshold_meV", 150.0)))
    medians = ((ref.get("class_split_medians_by_substrate") or {}).get(substrate)
               or ref.get("class_split_medians_meV") or {})
    amp_thr = (ref.get("amp_1L_2L") or {}).get(substrate)
    gap = (ref.get("amp_gap") or {}).get(substrate) or {}
    ea = (ref.get("ea_gap") or {}).get(substrate) or {}
    by_sub = ref.get("split_threshold_by_substrate") or {}
    # A calibrated ref lists every substrate it was fitted on. A substrate missing
    # from that list gets the global fallback threshold above, which is a guess:
    # rule_assign() flags it so the whole batch goes to a human (2026-09-05 review:
    # a misspelt substrate name used to be judged 2H silently).
    calibrated = (substrate in by_sub) if by_sub else True
    return {"split_threshold_meV": thr,
            "calibrated": calibrated,
            "ea_gap_ok": bool(ea.get("ok", False)),
            "ea_gap": ea,
            "class_split_medians_meV": medians,
            "amp_threshold_1L_below_2L_above": amp_thr,
            "amp_gap_ok": bool(gap.get("ok", False)),
            "amp_gap": gap,
            "amp_ratio_class_medians": (ref.get("amp_ratio_class_medians_by_substrate") or {}).get(substrate) or {}}


def rule_assign(fit: dict, substrate: str, ref: dict) -> dict:
    """Closed-form assignment. Stacking from the B-A splitting; layer number from
    the A amplitude when a calibrated threshold exists for this substrate; on a
    substrate with a calibrated 1L/bilayer amplitude gap, a monolayer amplitude
    vetoes a 2H splitting (2H -> 1L). Sets `flag` when the decision needs a human:
      amp_veto_ambiguous  -- splitting says 2H, amplitude is between the classes
      amp_nogap_ambiguous -- no amplitude gap on this substrate, 1L/3R undecidable
      ea_ambiguous        -- E_A decides 1L/3R on this substrate but sits at the threshold
      ea_amp_conflict     -- E_A and an unambiguous amplitude point to different labels
      substrate_uncalibrated -- the ref was calibrated, but not on this substrate
    """
    # the substrate shifts A and B by different amounts (on 276 nm SiO2/Si the 2H
    # splitting is only ~148 meV), so a per-substrate threshold wins when calibrated
    R = ref_for(ref, substrate)
    thr = R["split_threshold_meV"]
    split = fit["split_meV"]
    margin = split - thr
    stacking = "2H" if split >= thr else "3R-or-1L"
    amp_thr = R["amp_threshold_1L_below_2L_above"]
    ratio = None if not amp_thr else float(fit["aA"] / amp_thr)
    # confidence: how far the splitting sits from the threshold, saturating at 15 meV
    conf = float(min(1.0, abs(margin) / 15.0))
    reason = f"B-A = {split:.0f} meV ({'>=' if margin >= 0 else '<'} {thr:.0f})"
    flag, flag_reason, vetoed = "", "", False
    decided_by, ea_margin = "splitting", None
    if stacking == "2H":
        label = "2H"
        if ratio is not None and R["amp_gap_ok"] and ratio < 1.0:
            # amplitude-based confidence: 1 at the ambiguity edge and below, 0 at the threshold
            amp_conf = float(min(1.0, max(0.0, (1.0 - ratio) / (1.0 - AMP_VETO_AMBIGUOUS))))
            if ratio > AMP_VETO_AMBIGUOUS:
                flag = "amp_veto_ambiguous"
                flag_reason = (f"B-A 说 2H，但 A 幅度 {fit['aA']:.3f} 只有 1L/2L 阈值 {amp_thr:.3f} 的 "
                               f"{ratio:.2f} 倍，落在 {AMP_VETO_AMBIGUOUS:g}-1 的模糊带内，不静默改标，交人工")
                reason += f"; 幅度比 {ratio:.2f} 模糊（不否决）"
                conf = min(conf, amp_conf)
            else:
                label, vetoed, conf = "1L", True, amp_conf
                reason += (f"; 幅度否决：A 幅度 {fit['aA']:.3f} 仅为 1L/2L 阈值 {amp_thr:.3f} 的 "
                           f"{ratio:.2f} 倍（单层幅度），2H -> 1L")
        elif ratio is not None:
            reason += f"; 幅度比 {ratio:.2f}" + ("" if R["amp_gap_ok"] else "（本衬底幅度不判层数）")
    elif amp_thr is None:
        label = "3R-or-1L"
    else:
        label = "1L" if fit["aA"] < amp_thr else "3R"
        decided_by = "amplitude"
        reason += f"; A amplitude {fit['aA']:.3f} vs 1L/2L threshold {amp_thr:.3f} (ratio {ratio:.2f})"
        if not R["amp_gap_ok"] and R["ea_gap_ok"]:
            # No amplitude gap on this substrate, but the calibration shows the A-exciton
            # energy separates 1L from 3R. Closed form, zero calls. In-sample on the 135;
            # the absolute E_A drifts with strain and doping between batches, so the
            # threshold belongs to the calibration set and the margin is reported.
            ea_thr = float(R["ea_gap"]["threshold_eV"])
            ea_margin = (float(fit["EA"]) - ea_thr) * 1e3
            amp_label, label, decided_by = label, ("1L" if ea_margin >= 0 else "3R"), "E_A"
            half_gap = max(EA_AMBIGUOUS_MEV, float(R["ea_gap"]["gap_meV"]) / 2)
            conf = min(conf, float(min(1.0, abs(ea_margin) / half_gap)))
            reason += (f"; E_A {fit['EA']:.4f} eV {'>=' if ea_margin >= 0 else '<'} 1L/3R 阈值 {ea_thr:.4f}"
                       f"（{ea_margin:+.1f} meV；本衬底幅度无间隙，靠 E_A 分 1L/3R）")
            if abs(ea_margin) < EA_AMBIGUOUS_MEV:
                flag = "ea_ambiguous"
                flag_reason = f"E_A 距 1L/3R 阈值只有 {abs(ea_margin):.1f} meV（< {EA_AMBIGUOUS_MEV:g}），交人工"
            elif amp_label != label and not (AMP_NOGAP_BAND[0] < ratio < AMP_NOGAP_BAND[1]):
                flag = "ea_amp_conflict"
                flag_reason = (f"E_A 判 {label}，但 A 幅度比 {ratio:.2f} 在模糊带外、指向 {amp_label}，交人工")
        elif not R["amp_gap_ok"] and AMP_NOGAP_BAND[0] < ratio < AMP_NOGAP_BAND[1]:
            flag = "amp_nogap_ambiguous"
            flag_reason = (f"本衬底 1L 与双层的 A 幅度无间隙，幅度比 {ratio:.2f} 在 "
                           f"{AMP_NOGAP_BAND[0]:g}-{AMP_NOGAP_BAND[1]:g} 内，1L/3R 由 DR 判不了")
    if not R["calibrated"]:
        # overrides any other flag: nothing below is trustworthy on this substrate
        flag = "substrate_uncalibrated"
        flag_reason = (f"衬底 {substrate!r} 不在标定表里（标定过的：{sorted(ref.get('split_threshold_by_substrate') or {})}），"
                       f"用的是全局阈值 {thr:.1f} meV，整批交人工；先对该衬底跑 --calibrate")
        conf = 0.0
    return {"label": label, "stacking": stacking, "confidence": round(conf, 3),
            "reason": reason, "margin_meV": round(margin, 1),
            "amp_ratio": None if ratio is None else round(ratio, 3),
            "amp_gap_ok": R["amp_gap_ok"], "vetoed": vetoed,
            "decided_by": decided_by,
            "ea_margin_meV": None if ea_margin is None else round(ea_margin, 1),
            "flag": flag, "flag_reason": flag_reason}


def calibrate(groups: dict) -> dict:
    """Fit the rule's thresholds from labelled groups {(substrate, label): [fits]}."""
    ref = json.loads(json.dumps(DEFAULT_REF))
    sp = {lab: [] for lab in ("1L", "3R", "2H")}
    sp_sub: dict = {}
    amps: dict = {}
    eas: dict = {}
    for (sub, lab), fits in groups.items():
        for f in fits:
            if f:
                sp.setdefault(lab, []).append(f["split_meV"])
                sp_sub.setdefault(sub, {}).setdefault(lab, []).append(f["split_meV"])
                amps.setdefault(sub, {}).setdefault(lab, []).append(f["aA"])
                if "EA" in f:
                    eas.setdefault(sub, {}).setdefault(lab, []).append(f["EA"])

    def midpoint(d):
        lo = np.median(d.get("1L", []) + d.get("3R", [])) if (d.get("1L") or d.get("3R")) else 140.0
        hi = np.median(d["2H"]) if d.get("2H") else 160.0
        return round(float((lo + hi) / 2), 1)

    ref["split_threshold_meV"] = midpoint(sp)
    ref["split_threshold_by_substrate"] = {sub: midpoint(d) for sub, d in sp_sub.items()}
    ref["class_split_medians_meV"] = {k: round(float(np.median(v)), 1) for k, v in sp.items() if v}
    ref["class_split_medians_by_substrate"] = {
        sub: {k: round(float(np.median(v)), 1) for k, v in d.items()} for sub, d in sp_sub.items()}
    ref["amp_ratio_class_medians_by_substrate"] = {}
    for sub, d in amps.items():
        if d.get("1L") and d.get("3R"):
            thr = float(np.sqrt(np.median(d["1L"]) * np.median(d["3R"])))
            ref["amp_1L_2L"][sub] = round(thr, 4)
            ratios = {lab: [a / thr for a in v] for lab, v in d.items()}
            one = max(ratios["1L"])
            bilayer = ratios["3R"] + ratios.get("2H", [])
            two = min(bilayer)
            # the veto is allowed only where monolayers and bilayers do not overlap
            ref["amp_gap"][sub] = {"ok": bool(one < two),
                                   "max_1L_ratio": round(one, 3), "min_bilayer_ratio": round(two, 3),
                                   "n_1L": len(ratios["1L"]), "n_bilayer": len(bilayer)}
            ref["amp_ratio_class_medians_by_substrate"][sub] = {
                lab: round(float(np.median(v)), 2) for lab, v in ratios.items()}
    ref["ea_gap"] = {}
    for sub, d in eas.items():
        if d.get("1L") and d.get("3R"):
            lo1, hi3 = float(min(d["1L"])), float(max(d["3R"]))
            gap = (lo1 - hi3) * 1e3                   # physical direction only: 1L above 3R
            ref["ea_gap"][sub] = {"ok": bool(gap >= EA_GAP_MIN_MEV), "gap_meV": round(gap, 1),
                                  "threshold_eV": round((lo1 + hi3) / 2, 4),
                                  "min_1L_eV": round(lo1, 4), "max_3R_eV": round(hi3, 4),
                                  "median_1L_eV": round(float(np.median(d["1L"])), 4),
                                  "median_3R_eV": round(float(np.median(d["3R"])), 4),
                                  "n_1L": len(d["1L"]), "n_3R": len(d["3R"])}
    ref["notes"] = f"calibrated {time.strftime('%Y-%m-%d')} from {sum(len(v) for v in sp.values())} labelled fits"
    return ref


# ------------------------------------------------------------------ 3. the agents
def _roles(KA):
    qc = KA.Role(
        name="spectrum_qc",
        system=(
            "你是二维材料光谱分析员。你看到的是一条 MoS2 的微分反射（DR）谱：灰点是数据，"
            "蓝线是双高斯 + 二次背景的拟合，两条竖虚线是拟合给出的 A、B 激子位置。\n"
            "你只回答一个问题：**这次拟合给出的 A、B 峰位可不可信？**\n"
            "  - fit_ok：两个峰都被拟合线抓住，虚线落在肉眼可见的峰上。\n"
            "  - bad_background：峰位大致对，但背景线明显偏离数据（例如高能端翘起被当成峰）。\n"
            "  - peak_missing：数据里看不出两个峰（只有一个，或全是噪声），拟合硬凑出来的。\n"
            "  - shifted：肉眼看峰在别处，虚线偏了 20 meV 以上。\n"
            "  - noisy：噪声太大，峰位误差会很大。\n"
            "宁可判 noisy 也不要把噪声上的拟合判成 fit_ok。只输出 JSON。"),
        schema_hint='{"verdict": "fit_ok|bad_background|peak_missing|shifted|noisy", '
                    '"confidence": 0.0-1.0, "reason": "一句话，中文"}',
        # kimi-k3 spends its budget on chain of thought: 8/23 completions were cut
        # at 4096 and re-sent (experiments_20260902/autospectra_kimi)
        max_tokens=16384)
    asg = KA.Role(
        name="stacking_assigner",
        system=(
            "你是二维材料光谱分析员。给你一条 MoS2 DR 谱拟合出的数字和一张**本衬底**的参考表，"
            "判断这片是 2H 双层、3R 双层还是单层 1L。\n"
            "物理依据：\n"
            "  - B 与 A 激子的能量差（B−A）：2H 双层比 3R 双层和单层大 10–20 meV——"
            "这是区分 2H 与其他的主要依据。阈值和各类中位数随衬底变化很大"
            "（276 nm SiO2 上 2H 的 B−A 只有约 147 meV），所以只能和参考表里本衬底的 "
            "split_threshold_meV_this_substrate 与 class_split_medians_meV_this_substrate 比，"
            "不要用别的衬底或记忆里的数。\n"
            "  - A 峰幅度：在 SiO2 衬底上双层的 ΔR/R 幅度约为单层的两倍（同一衬底上比较）——"
            "这是区分 3R 与 1L 的依据。参考表里的 amp_threshold_1L_below_2L_above 是**判界阈值**"
            "（1L 与 3R 中位数的几何平均），不是单层的幅度；幅度比 = A 峰幅度 / 该阈值，"
            "各类的典型值见 amp_ratio_class_medians_this_substrate（SiO2 上 1L 约 0.5、双层约 1.5–2）。\n"
            "  - 在蓝宝石（Sapphire）上幅度**不判层数**：标注谱三类的幅度比中位数约 0.91 / 1.10 / 1.03，"
            "范围完全重叠。参考表 amp_discriminative_this_substrate 为 false 时不要用幅度分 3R 与 1L，"
            "B−A 又不能定 2H 的话就判 uncertain。\n"
            "  - 衬底会整体平移能量，所以要和**同一衬底**的参考值比。\n"
            "证据不足就判 uncertain，不要硬猜。只输出 JSON。"),
        schema_hint='{"label": "2H|3R|1L|uncertain", "confidence": 0.0-1.0, '
                    '"reason": "一句话，中文，只说决定性证据"}',
        max_tokens=16384)
    return qc, asg


def render_fit(x, y, fit) -> "np.ndarray":
    """The picture the QC agent looks at: data, fit, A/B guides. BGR uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=110)
    m = (x > 1.6) & (x < 2.4)
    ax.plot(x[m], y[m], ".", color="0.55", ms=2.5, label="data")
    if fit:
        xx = np.linspace(NLO, NHI, 300)
        ax.plot(xx, model2(xx, *fit["popt"]), color="#2f77ad", lw=1.8, label="fit")
        for e, lab in ((fit["EA"], "A"), (fit["EB"], "B")):
            ax.axvline(e, color="0.35", ls="--", lw=1)
            ax.text(e, ax.get_ylim()[1], lab, ha="center", va="bottom")
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel("ΔR/R")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[:, :, [2, 1, 0]].copy()


# ------------------------------------------------------------------ 4. the pipeline
def gate_decision(rule: dict, gate_meV: float, nogap_policy: str) -> tuple[bool, str]:
    """Which spectra the models see. Returns (to_model, why)."""
    if rule["label"] == "3R-or-1L":
        return True, "规则无本衬底幅度阈值，分不开 3R/1L"
    if abs(rule["margin_meV"]) < gate_meV:
        return True, f"|B−A − 阈值| = {abs(rule['margin_meV']):.1f} meV < {gate_meV:g} meV"
    if rule["flag"] == "amp_nogap_ambiguous" and nogap_policy == "model":
        return True, rule["flag_reason"]
    return False, ""


def _vote_state(ans: dict | None, votes: int) -> tuple[int, bool, dict]:
    """(answers received, split, tally) of a pool.map() result under the shared
    AgentPool contract: `_n_votes` counts the answers that came back (an HTTP
    failure loses one), `_split` is set when a decision majority was not unanimous;
    without it a split is a majority short of the answers received."""
    if not ans:
        return 0, False, {}
    n = int(ans.get("_n_votes", 1) or 1)
    tally = ans.get("_tally") or {}
    split = ans.get("_split")
    if split is None:
        split = bool(tally) and max(tally.values()) < n
    return n, bool(split), tally


def _rule_flag_reasons(rule: dict, modelled: bool, nogap_policy: str) -> list:
    """Escalation reasons the rule itself raises (no model involved)."""
    if rule["flag"] == "amp_veto_ambiguous":
        return [rule["flag_reason"]]
    if rule["flag"] == "amp_nogap_ambiguous" and not (modelled and nogap_policy == "model"):
        return [rule["flag_reason"]]
    if rule["flag"] in ("ea_ambiguous", "ea_amp_conflict", "substrate_uncalibrated"):
        return [rule["flag_reason"]]
    return []


def analyze(spectra: list, ref: dict, work: Path, no_ai=False, votes=3, workers=6,
            escalate_to_queue=True, gate_meV: float = DEFAULT_GATE_MEV,
            nogap_policy: str = "escalate", model_qc: bool = False) -> dict:
    """Analyze spectra with a supplied calibration reference and optional model review.

    Each input provides id, substrate, x, y and optionally a reference label whose
    independent provenance must be checked by the caller. model_qc adds vision-model
    fit review for gated cases and is off by default. Returns per-spectrum results,
    a summary and unresolved cases for the review queue."""
    if nogap_policy not in NOGAP_POLICIES:
        raise ValueError(f"nogap_policy must be one of {NOGAP_POLICIES}, got {nogap_policy!r}")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pool = None
    if not no_ai:
        try:
            import region_agents as RA
            KA, _ = RA._import_kimi()
            client = KA.KimiClient(cache_dir=str(work / "_kimi_cache"))
            if client.available:
                pool = KA.AgentPool(client, workers=workers, votes=votes)
                qc_role, asg_role = _roles(KA)
        except Exception as e:
            print(f"  [warn] Kimi 不可用（{e}），全部按规则判定")

    # ---- fit (closed form) + rule
    results = []
    for s in spectra:
        fit = fit_ab(s["x"], s["y"])
        r = {"id": s["id"], "substrate": s["substrate"], "truth": s.get("label"),
             "fit": None if fit is None else {k: v for k, v in fit.items() if k != "popt"},
             "_popt": None if fit is None else fit["popt"]}
        if fit is None:
            r.update(label="unfit", rule=None, qc=None, llm=None, escalate=True,
                     escalate_reason="拟合未通过自检（幅度低于 2.5× 噪声或峰位贴界）", _ai=False,
                     gated=False, gate_reason="", note="")
        else:
            r["rule"] = rule_assign(fit, s["substrate"], ref)
            r["label"] = r["rule"]["label"]
            r["gated"], r["gate_reason"] = gate_decision(r["rule"], gate_meV, nogap_policy)
        results.append(r)

    # ---- gate: the models see only what the rule could not settle
    fitted = [(s, r) for s, r in zip(spectra, results) if r["fit"]]
    to_model = [(s, r) for s, r in fitted if r["gated"]] if pool is not None else []
    for s, r in fitted:
        if pool is not None and r["gated"]:
            continue
        r.update(qc=None, llm=None, _ai=False)
        rule = r["rule"]
        if pool is not None:
            r["note"] = (f"规则判定距阈值 {abs(rule['margin_meV']):.1f} meV ≥ {gate_meV:g} meV，未调用模型"
                         if not r["gated"] else "")
            reasons = _rule_flag_reasons(rule, modelled=False, nogap_policy=nogap_policy)
        else:
            # no key: the spectra the models would have seen go to the human instead
            r["note"] = "无模型，仅规则"
            reasons = []
            if rule["label"] == "3R-or-1L":
                reasons.append("规则无本衬底幅度阈值，分不开 3R/1L（无模型，交人工）")
            elif abs(rule["margin_meV"]) < gate_meV:
                reasons.append(f"B−A 距阈值不到 {gate_meV:g} meV，规则判定不稳（无模型，交人工）")
            reasons += _rule_flag_reasons(rule, modelled=False, nogap_policy="escalate")
        r["escalate"] = bool(reasons)
        r["escalate_reason"] = "; ".join(reasons)

    # ---- QC + cross-check (models), N votes each, gated spectra only
    if pool is not None and to_model:
        def build_qc(item):
            s, r = item
            fit = {**r["fit"], "popt": r["_popt"]}
            img = render_fit(np.asarray(s["x"]), np.asarray(s["y"]), fit)
            return (f"衬底 {s['substrate']}；拟合给出 E_A = {fit['EA']:.3f} eV，E_B = {fit['EB']:.3f} eV，"
                    f"信噪比 {fit['snr']:.1f}。这次拟合可信吗？"), [img]

        def build_asg(item):
            s, r = item
            f = r["fit"]
            R = ref_for(ref, s["substrate"])          # the reference the rule used, nothing else
            ref_txt = json.dumps({
                "substrate": s["substrate"],
                "split_threshold_meV_this_substrate": R["split_threshold_meV"],
                "class_split_medians_meV_this_substrate": R["class_split_medians_meV"],
                "amp_threshold_1L_below_2L_above": R["amp_threshold_1L_below_2L_above"],
                "amp_ratio_this_spectrum": r["rule"]["amp_ratio"],
                "amp_ratio_class_medians_this_substrate": R["amp_ratio_class_medians"],
                "amp_discriminative_this_substrate": R["amp_gap_ok"],
            }, ensure_ascii=False)
            return (f"衬底 {s['substrate']}。E_A = {f['EA']:.4f} eV，E_B = {f['EB']:.4f} eV，"
                    f"B−A = {f['split_meV']:.1f} meV，A 峰幅度 {f['aA']:.4f}，B 峰幅度 {f['aB']:.4f}，"
                    f"信噪比 {f['snr']:.1f}。本衬底参考表：{ref_txt}。这片是 2H、3R 还是 1L？"), None

        qc = (pool.map(qc_role, to_model, build_qc, key="verdict") if model_qc
              else [None] * len(to_model))
        asg = pool.map(asg_role, to_model, build_asg, key="label")
        for (s, r), q, a in zip(to_model, qc, asg):
            r["qc"] = q
            r["llm"] = a
            r["_ai"] = True
            r["note"] = ""
            rule = r["rule"]
            rl = rule["label"]
            reasons = []
            # -- fit QC: a non-ok majority escalates; a split whose majority is fit_ok
            #    does not (it changed no label in experiment B); a short vote is a
            #    network failure, reported as such and never as "disagreement"
            qn, qsplit, qtally = _vote_state(q, votes)
            if not model_qc:
                pass                                   # closed-form QC ran upstream (spectra_ingest)
            elif q is None:
                reasons.append("拟合质检无有效多数（平票或无返回）")
            else:
                if q.get("verdict") != "fit_ok":
                    reasons.append(f"拟合质检 {q.get('verdict')}" + (f"（票 {qtally}）" if qsplit else ""))
                if qn < votes:
                    reasons.append(f"质检投票不足 {qn}/{votes}（请求失败，非模型分歧）")
            # -- stacking cross-check: an escalation source, never a label fixer
            an, asplit, atally = _vote_state(a, votes)
            if a is None:
                reasons.append("堆垛判定无有效多数（平票或无返回）")
            else:
                ml = a.get("label")
                if ml == "uncertain":
                    reasons.append("模型判 uncertain")
                elif rl == "3R-or-1L":
                    reasons.append(f"规则无本衬底幅度阈值分不开 3R/1L，模型判 {ml}")
                elif ml != rl:
                    reasons.append(f"模型判 {ml}，规则判 {rl}")
                if an < votes:
                    reasons.append(f"堆垛投票不足 {an}/{votes}（请求失败，非模型分歧）")
                elif asplit:
                    reasons.append(f"堆垛票数分歧 {atally}")
            reasons += _rule_flag_reasons(rule, modelled=True, nogap_policy=nogap_policy)
            r["escalate"] = bool(reasons)
            r["escalate_reason"] = "; ".join(reasons)

    # ---- statistics
    summary = {}
    for r in results:
        key = f"{r['substrate']} | {r.get('truth') or '?'}"
        d = summary.setdefault(key, {"n": 0, "fitted": 0, "labels": {}, "EA": [], "split": []})
        d["n"] += 1
        if r["fit"]:
            d["fitted"] += 1
            d["EA"].append(r["fit"]["EA"]); d["split"].append(r["fit"]["split_meV"])
        d["labels"][r["label"]] = d["labels"].get(r["label"], 0) + 1
    for d in summary.values():
        d["EA_mean"] = round(float(np.mean(d["EA"])), 4) if d["EA"] else None
        d["split_mean_meV"] = round(float(np.mean(d["split"])), 1) if d["split"] else None
        del d["EA"], d["split"]
    truth = [(r["truth"], r["label"]) for r in results if r.get("truth") and r["fit"]]
    agreement = None
    if truth:
        ok = sum(1 for t, l in truth if l == t or (l == "3R-or-1L" and t in ("3R", "1L")))
        agreement = round(ok / len(truth), 3)
    veto = {"applied": {}, "ambiguous": {}}
    for s, r in fitted:
        if r["rule"]["vetoed"]:
            veto["applied"][s["substrate"]] = veto["applied"].get(s["substrate"], 0) + 1
        if r["rule"]["flag"] == "amp_veto_ambiguous":
            veto["ambiguous"][s["substrate"]] = veto["ambiguous"].get(s["substrate"], 0) + 1
    n_gated = sum(1 for _, r in fitted if r["gated"])
    n_modelled = len(to_model)
    esc = [{"stage": "autospectra", "sample": r["substrate"], "spectrum": r["id"],
            "reason": r["escalate_reason"], "confidence": (r["rule"] or {}).get("confidence", 0)}
           for r in results if r.get("escalate")]
    if escalate_to_queue:
        for e in esc:
            RV.push_queue(work, e)
    out = {"ref": ref, "n": len(results), "fitted": sum(1 for r in results if r["fit"]),
           "ai": pool is not None, "votes": votes, "agreement_with_labels": agreement,
           "gate": {"meV": gate_meV, "nogap_policy": nogap_policy, "nogap_band": list(AMP_NOGAP_BAND),
                    "veto_ambiguous_above": AMP_VETO_AMBIGUOUS,
                    "n_gated": n_gated, "n_modelled": n_modelled,
                    "model_qc": model_qc,
                    "model_calls_expected": n_modelled * votes * (2 if model_qc else 1)},
           "amp_veto": veto,
           "results": [{k: v for k, v in r.items() if k != "_popt"} for r in results],
           "summary": summary, "escalations": esc,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (work / "autospectra_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def as_result(out: dict) -> Result:
    """The stage contract, for the orchestrator."""
    esc = out["escalations"]
    conf = 1.0 if out["ai"] else 0.5
    gate = out.get("gate") or {}
    return Result(ok=not esc, confidence=conf,
                  data={k: v for k, v in out.items() if k != "results"},
                  evidence=(f"{out['fitted']}/{out['n']} 条谱拟合通过；"
                            + (f"与标注一致 {out['agreement_with_labels']*100:.0f}%；" if out["agreement_with_labels"] is not None else "")
                            + (f"门内 {gate.get('n_modelled', 0)} 条经模型质检 + 交叉判（门宽 {gate.get('meV')} meV）"
                               if out["ai"] else "无模型，仅规则")),
                  escalate=bool(esc),
                  escalate_reason=f"{len(esc)} 条谱需要人看" if esc else "")


# ------------------------------------------------------------------ 5. data + CLI
def load_pkl(path: Path, limit: int | None = None) -> list:
    """Spectra/curves2.pkl: {(substrate, label): [{x, ypos, file, ...}]}."""
    curves = pickle.load(open(path, "rb"))
    spectra = []
    for (sub, lab), glist in curves.items():
        for g in glist:
            spectra.append({"id": str(g.get("file", len(spectra))), "substrate": sub,
                            "label": lab, "x": np.asarray(g["x"], float),
                            "y": np.asarray(g["ypos"], float)})
    return spectra[:limit] if limit else spectra


def main():
    ap = argparse.ArgumentParser(description="AutoSpectra: agent-supervised DR analysis")
    ap.add_argument("--pkl", required=True, help="curves2.pkl (Spectra/)")
    ap.add_argument("--work", default="_work/autospectra")
    ap.add_argument("--ref", default=None, help="ref.json from --calibrate (default: <work>/ref.json if present)")
    ap.add_argument("--calibrate", action="store_true", help="fit the rule thresholds from the labelled groups and exit")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gate-mev", type=float, default=DEFAULT_GATE_MEV,
                    help="model gate: only spectra with |B-A - threshold| below this see the models")
    ap.add_argument("--model-qc", action="store_true",
                    help="also ask the vision model whether each gated fit looks right (off by default, see analyze())")
    ap.add_argument("--nogap-policy", choices=NOGAP_POLICIES, default="escalate",
                    help="1L/3R decisions on a substrate without an amplitude gap (sapphire), ratio in "
                         f"{AMP_NOGAP_BAND}: escalate to the human directly (default) or ask the models first")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    spectra = load_pkl(Path(a.pkl), a.limit)

    if a.calibrate:
        groups: dict = {}
        for s in spectra:
            groups.setdefault((s["substrate"], s["label"]), []).append(fit_ab(s["x"], s["y"]))
        ref = calibrate(groups)
        (work / "ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2))
        print(json.dumps(ref, ensure_ascii=False, indent=2))
        return

    rp = Path(a.ref) if a.ref else work / "ref.json"
    ref = json.loads(rp.read_text()) if rp.exists() else DEFAULT_REF
    out = analyze(spectra, ref, work, no_ai=a.no_ai, votes=a.votes,
                  gate_meV=a.gate_mev, nogap_policy=a.nogap_policy, model_qc=a.model_qc)
    print(f"\n{out['fitted']}/{out['n']} 条谱拟合通过"
          + (f"；与标注一致 {out['agreement_with_labels']*100:.1f}%" if out["agreement_with_labels"] is not None else ""))
    print(f"{'substrate | label':<26}{'n':>4}{'fit':>5}{'E_A':>9}{'B-A':>8}  labels")
    for k, d in out["summary"].items():
        print(f"{k:<26}{d['n']:>4}{d['fitted']:>5}{d['EA_mean'] or 0:>9.4f}{d['split_mean_meV'] or 0:>8.1f}  {d['labels']}")
    g = out["gate"]
    print(f"\n幅度否决 {out['amp_veto']['applied'] or '无'}；模糊 {out['amp_veto']['ambiguous'] or '无'}"
          f"\n门宽 {g['meV']:g} meV：门内 {g['n_gated']} 条"
          + (f"，模型看了 {g['n_modelled']} 条（约 {g['model_calls_expected']} 次调用）" if out["ai"] else "（无模型，交人工）"))
    print(f"上报 {len(out['escalations'])} 条 -> {RV.queue_path(work)}\n结果 {work / 'autospectra_results.json'}")


if __name__ == "__main__":
    main()
