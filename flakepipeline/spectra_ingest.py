#!/usr/bin/env python3
"""Read spectrometer exports and compute explicit optical-spectrum quality checks.

Excel/CSV loaders locate the supported differential-reflectance column group,
clean numeric rows and preserve the chosen signal convention. Deterministic
checks report insufficient samples, clipping, fit quality and spectral-shape
agreement. Optional downstream AutoSpectra analysis uses a supplied calibration
reference; substrate identity must be provided explicitly.

Default thresholds are configurable historical settings, not validation on a
new instrument, substrate or batch. These checks do not verify spatial pairing,
prove that the optical spot lies inside the intended object, or establish a new
physical stacking reference. No external model is called by the loader itself.

    python3 spectra_ingest.py --dir /path/to/workbooks --substrate Sapphire --qc-only"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import autospectra as AS  # noqa: E402  (fit_ab, calibrate, analyze, ref_for)

# ---------------------------------------------------------------- constants
# Cleaning window (process_dr.py: ELO, EHI). The fit window is narrower
# (autospectra.NLO/NHI = 1.72-2.14 eV); a spectrum must cover it entirely.
ELO, EHI = 1.55, 2.50
# Orientation: make the C-exciton feature positive (process_dr.py main()).
C_WIN, BASE_WIN = (2.15, 2.40), (1.60, 1.70)
MIN_POINTS = 300                       # process_dr.py: shorter traces were skipped ("SHORT")

# The substrates are 276 nm and 67 nm SiO2/Si (measured oxide, the paper's numbers).
# "285nm" / "70nm" are only the nominal wafer names used in the raw data folders;
# they are accepted as aliases so a folder name can be passed through, nothing more.
SUBSTRATE_ALIAS = {"276nm": "276 nm SiO2/Si", "276": "276 nm SiO2/Si",
                   "285nm": "276 nm SiO2/Si", "285": "276 nm SiO2/Si",
                   "67nm": "67 nm SiO2/Si", "67": "67 nm SiO2/Si",
                   "70nm": "67 nm SiO2/Si", "70": "67 nm SiO2/Si",
                   "sapphire": "Sapphire", "al2o3": "Sapphire"}


def canon_substrate(s: str) -> str:
    key = s.strip().lower().replace(" ", "")
    return SUBSTRATE_ALIAS.get(key, s.strip())


# ---------------------------------------------------------------- loaders (verbatim port)
def clean(x, y):
    """process_dr.clean: finite, inside [ELO, EHI], sorted, rolling-median outlier cut."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > ELO) & (x < EHI)
    x, y = x[m], y[m]
    o = np.argsort(x)
    x, y = x[o], y[o]
    if len(y) > 25:
        med = pd.Series(y).rolling(21, center=True, min_periods=5).median().values
        resid = np.abs(y - med)
        mad = np.nanmedian(resid) + 1e-12
        keep = resid < 8 * mad
        x, y = x[keep], y[keep]
    return x, y


def load_xlsx(f):
    """process_dr.load_xlsx: find the 'X [eV]' header row; prefer the '1-B/A' column
    group (not '-(1-B/A)'); otherwise the last 'X [..]' column. y is the column after x."""
    df = pd.read_excel(f, header=None)
    hdr_row = None
    for i in range(min(3, len(df))):
        if any(str(v).strip().startswith("X [") for v in df.iloc[i].tolist()):
            hdr_row = i
            break
    if hdr_row is None:
        raise ValueError("no X [eV] header")
    xcol = None
    if hdr_row > 0:
        labels = [str(v).strip() for v in df.iloc[hdr_row - 1].tolist()]
        for c, lab in enumerate(labels):
            if lab.upper() == "1-B/A":
                xcol = c
                break
    if xcol is None:
        hdr = [str(v).strip() for v in df.iloc[hdr_row].tolist()]
        xcols = [c for c, v in enumerate(hdr) if v.startswith("X [")]
        xcol = xcols[-1]
    x = pd.to_numeric(df[xcol], errors="coerce").values
    y = pd.to_numeric(df[xcol + 1], errors="coerce").values
    return clean(x, y)


def load_csv(f):
    """process_dr.load_csv: wavelength [nm] in column 0, dR/R in column 3, one header line."""
    df = pd.read_csv(f, header=None, skiprows=1)
    wl = pd.to_numeric(df[0], errors="coerce").values
    dr = pd.to_numeric(df[3], errors="coerce").values
    e = 1239.842 / wl
    return clean(e, dr)


def load_any(f):
    return load_csv(f) if str(f).lower().endswith(".csv") else load_xlsx(f)


def orient(x, y):
    """process_dr main(): sign chosen so the C-exciton band (2.15-2.40 eV) sits above
    the 1.60-1.70 eV baseline. Returns (ypos, sign); sign is None if either window is empty."""
    m_c = (x > C_WIN[0]) & (x < C_WIN[1])
    m_b = (x > BASE_WIN[0]) & (x < BASE_WIN[1])
    if not m_c.any() or not m_b.any():
        return None, None
    sgn = float(np.sign(np.nanmedian(y[m_c]) - np.nanmedian(y[m_b])))
    if sgn == 0.0:
        sgn = 1.0
    return y * sgn, sgn


def workbook_number(path) -> int | None:
    b = unicodedata.normalize("NFC", os.path.basename(str(path)))
    m = re.search(r"工作簿(\d+)", b)
    return int(m.group(1)) if m else None


def list_files(folder: str, skip_reference_spots: bool = True) -> list:
    """process_dr main(): xlsx and csv, recursive; when the same workbook number exists
    as both, keep the xlsx; '-2.' files are the on-monolayer reference spots of a
    bilayer flake and are not spectra to be labelled."""
    xl = sorted(glob.glob(os.path.join(folder, "**", "*.xlsx"), recursive=True))
    cs = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    have = {workbook_number(f) for f in xl if workbook_number(f) is not None}
    cs = [f for f in cs if workbook_number(f) is None or workbook_number(f) not in have]
    files = xl + cs
    if skip_reference_spots:
        files = [f for f in files if "-2." not in os.path.basename(f)]
    return files


# ---------------------------------------------------------------- QC (closed form)
# Thresholds and what they are anchored to (paper set, 135 spectra, 2026-09-05):
#   n_points     >= 300      process_dr skipped shorter traces; the 135 have 1232-1316 points
#   axis         covers [1.72, 2.14] eV fully; otherwise fit_ab has nothing to fit
#   clipping     top 1 % of |ypos| must not be a run of identical values (ADC saturation)
#   fit          fit_ab must succeed; on the 135 it succeeds 135/135
#   snr          fit_ab's min(aA, aB)/noise; the paper's own gate is 2.5 (fit_ab returns
#                None below it, so a reject there shows up as "fit failed"); the 135
#                accepted spectra have p5 = 5.8, minimum 2.9 (sapphire 2H) -> warn below 5
#   shape_cc     Pearson correlation on a 1.7-2.4 eV grid with the best-matching class
#                median of the substrate (fit_ab.py's metric); the paper gated fits at
#                >= 0.90 (SHAPE_CC_MIN) -> reject below it; the 135 accepted spectra span
#                0.939-0.999 (low tail = sapphire 1L) -> warn below 0.93
QC_DEFAULTS = dict(min_points=MIN_POINTS, snr_warn=5.0, snr_reject=2.5,
                   cc_warn=0.93, cc_reject=0.90, clip_frac=0.01)


# Shape metric, verbatim from the paper's fit_ab.py main(): interpolate the oriented
# spectrum onto a fixed 1.7-2.4 eV grid and take the Pearson correlation with the class
# median. No detrending, no normalisation -- Pearson is already offset/scale invariant.
SHAPE_GRID = np.linspace(1.7, 2.4, 400)


def _on_grid(x, ypos, grid):
    return np.interp(grid, x, ypos)


def shape_templates(spectra: list, grid=None) -> tuple[dict, "np.ndarray"]:
    """Build median normalised templates per (substrate, supplied label).

    Use an independently selected calibration collection. Unlabelled spectra are
    pooled under (substrate, None). Keeping labels separate avoids mixing distinct
    spectral classes into one substrate-wide reference."""
    grid = SHAPE_GRID if grid is None else grid
    by = {}
    for s in spectra:
        x = np.asarray(s["x"], float)
        if "ypos" in s:
            yp = np.asarray(s["ypos"], float)
        else:
            yp, _ = orient(x, np.asarray(s["y"], float))
        if yp is None:
            continue
        by.setdefault((s["substrate"], s.get("label")), []).append(_on_grid(x, yp, grid))
    return {k: np.median(np.stack(v), axis=0) for k, v in by.items()}, grid


def templates_for(templates: dict, substrate: str) -> list:
    return [t for (sub, _), t in templates.items() if sub == substrate]


def qc_spectrum(x, ypos, fit, templates=None, grid=None, cfg=None) -> dict:
    """Closed-form quality checks on one oriented spectrum.

    `templates` is the list of this substrate's class templates (templates_for()).
    Returns {"verdict": accept|warn|reject, "flags": [...], "metrics": {...}}.
    A reject means the spectrum should be re-measured now; a warn means it goes
    through but is marked for a human look in the review queue.
    """
    c = {**QC_DEFAULTS, **(cfg or {})}
    flags, metrics = [], {}
    n = int(len(x)); metrics["n_points"] = n
    if n < c["min_points"]:
        flags.append(("reject", f"only {n} points (< {c['min_points']})"))
    lo, hi = (float(x.min()), float(x.max())) if n else (np.nan, np.nan)
    metrics["e_min"], metrics["e_max"] = lo, hi
    if not (lo <= AS.NLO and hi >= AS.NHI):
        flags.append(("reject", f"axis {lo:.2f}-{hi:.2f} eV does not cover the {AS.NLO}-{AS.NHI} eV fit window"))
    if n:
        top = np.sort(np.abs(ypos))[-max(3, int(c["clip_frac"] * n)):]
        metrics["top_identical_frac"] = float(np.mean(np.isclose(top, top.max(), rtol=0, atol=1e-9)))
        if metrics["top_identical_frac"] > 0.5:
            flags.append(("reject", "saturated: the largest |dR/R| values are identical (ADC clipping)"))
    if fit is None:
        flags.append(("reject", "A/B fit failed"))
    else:
        metrics["snr"] = float(fit["snr"])
        metrics["EA"], metrics["EB"] = float(fit["EA"]), float(fit["EB"])
        if fit["snr"] < c["snr_reject"]:
            flags.append(("reject", f"snr {fit['snr']:.1f} < {c['snr_reject']}"))
        elif fit["snr"] < c["snr_warn"]:
            flags.append(("warn", f"snr {fit['snr']:.1f} < {c['snr_warn']}"))
    if templates and n and grid is not None:
        v = _on_grid(x, ypos, grid)
        # best match over the substrate's known classes: the QC does not know the label
        cc = float(max(np.corrcoef(v, t)[0, 1] for t in templates))
        metrics["shape_cc"] = cc
        if cc < c["cc_reject"]:
            flags.append(("reject", f"shape_cc {cc:.3f} < {c['cc_reject']}: does not look like a spectrum of this substrate"))
        elif cc < c["cc_warn"]:
            flags.append(("warn", f"shape_cc {cc:.3f} < {c['cc_warn']}: outside the paper set's range (0.939-0.999)"))
    verdict = "reject" if any(k == "reject" for k, _ in flags) else ("warn" if flags else "accept")
    return {"verdict": verdict, "flags": [m for _, m in flags], "metrics": metrics}


# ---------------------------------------------------------------- ingest
def ingest_dir(folder: str, substrate: str, labels: dict | None = None,
               templates: dict | None = None, grid=None, qc_cfg=None) -> tuple[list, list]:
    """Load every spectrum in `folder`, orient it, fit it, QC it.

    Returns (spectra, qc): `spectra` is the list `autospectra.analyze()` accepts
    ({id, substrate, x, y(=ypos), label?}); `qc` has one record per file, including
    the ones that failed to load (verdict reject) so nothing disappears silently.
    """
    substrate = canon_substrate(substrate)
    spectra, qc = [], []
    tpl = templates_for(templates or {}, substrate)
    for f in list_files(folder):
        rec = {"id": os.path.basename(f), "file": f, "substrate": substrate}
        try:
            x, y = load_any(f)
        except Exception as e:
            rec.update(verdict="reject", flags=[f"load failed: {e!r}"[:120]], metrics={})
            qc.append(rec); continue
        ypos, sgn = orient(x, y)
        if ypos is None:
            rec.update(verdict="reject", flags=["cannot orient: no points in 1.60-1.70 or 2.15-2.40 eV"], metrics={"n_points": int(len(x))})
            qc.append(rec); continue
        fit = AS.fit_ab(x, ypos) if len(x) >= MIN_POINTS else None
        q = qc_spectrum(x, ypos, fit, tpl, grid, qc_cfg)
        rec.update(sign=sgn, **q)
        qc.append(rec)
        if q["verdict"] == "reject":
            continue
        s = {"id": rec["id"], "substrate": substrate, "x": x, "y": ypos,
             "qc_verdict": q["verdict"], "qc_flags": q["flags"]}
        n = workbook_number(f)
        if labels and (rec["id"] in labels or (n is not None and n in labels)):
            s["label"] = labels.get(rec["id"], labels.get(n))
        spectra.append(s)
    return spectra, qc


def load_labels(path: str) -> dict:
    """labels.csv with columns file (or workbook) and label in {1L, 3R, 2H}."""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        lab = str(r["label"]).strip()
        if "file" in df.columns and isinstance(r["file"], str):
            out[os.path.basename(r["file"])] = lab
        if "workbook" in df.columns and pd.notna(r["workbook"]):
            out[int(r["workbook"])] = lab
    return out


def write_qc(qc: list, work: Path, folder: str, substrate: str) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    counts = {k: sum(1 for r in qc if r["verdict"] == k) for k in ("accept", "warn", "reject")}
    remeasure = [{"file": r["id"], "why": r["flags"]} for r in qc if r["verdict"] == "reject"]
    out = {"folder": folder, "substrate": canon_substrate(substrate), "n_files": len(qc),
           "counts": counts, "thresholds": QC_DEFAULTS, "re_measure_now": remeasure,
           "records": qc}
    p = work / "spectra_qc.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    return p


def main():
    ap = argparse.ArgumentParser(description="raw spectrometer folder -> QC -> AutoSpectra")
    ap.add_argument("--dir", required=True, help="folder with 工作簿N.xlsx / *.csv (recursive)")
    ap.add_argument("--substrate", required=True, help='"276 nm SiO2/Si", "67 nm SiO2/Si" or "Sapphire"; short forms 276nm / 67nm / sapphire (folder names 285nm / 70nm also accepted)')
    ap.add_argument("--work", default="_work/spectra_ingest")
    ap.add_argument("--labels", default=None, help="optional labels.csv (file|workbook, label) for ground truth")
    ap.add_argument("--calibrated", default=None, help="ref.json from a previous `autospectra.py --calibrate`; required unless --qc-only or --pkl-calibrate")
    ap.add_argument("--pkl-calibrate", default=None, help="curves2.pkl: calibrate thresholds and shape templates from the paper set")
    ap.add_argument("--qc-only", action="store_true", help="stop after the QC; do not run AutoSpectra")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--model-qc", action="store_true", help="also ask the vision model about each gated fit (off by default)")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="run AutoSpectra on a substrate the ref was not calibrated on; every spectrum is then escalated")
    a = ap.parse_args()

    work = Path(a.work)
    templates, grid, ref = None, None, None
    if a.pkl_calibrate:
        paper = AS.load_pkl(Path(a.pkl_calibrate))
        templates, grid = shape_templates(paper)
        groups = {}          # calibrate() wants {(substrate, label): [fit dicts]}
        for s in paper:
            groups.setdefault((s["substrate"], s["label"]), []).append(AS.fit_ab(s["x"], s["y"]))
        ref = AS.calibrate(groups)
    elif a.calibrated:
        ref = json.loads(Path(a.calibrated).read_text())

    labels = load_labels(a.labels) if a.labels else None
    spectra, qc = ingest_dir(a.dir, a.substrate, labels, templates, grid)
    p = write_qc(qc, work, a.dir, a.substrate)
    c = {k: sum(1 for r in qc if r["verdict"] == k) for k in ("accept", "warn", "reject")}
    print(f"质检: {len(qc)} 个文件 -> 通过 {c['accept']}  警告 {c['warn']}  拒收 {c['reject']}   -> {p}")
    for r in qc:
        if r["verdict"] != "accept":
            print(f"  [{r['verdict']:6s}] {r['id']}: " + "; ".join(r["flags"]))
    if a.qc_only:
        return
    if ref is None:
        sys.exit("AutoSpectra 需要标定：给 --calibrated ref.json 或 --pkl-calibrate curves2.pkl（或加 --qc-only 只做质检）")
    known = sorted(ref.get("split_threshold_by_substrate") or {})
    if known and canon_substrate(a.substrate) not in known and not a.allow_uncalibrated:
        sys.exit(f"衬底 {canon_substrate(a.substrate)!r} 没有标定（标定过的：{known}）。"
                 "先在该衬底的带标签谱上跑 autospectra.py --calibrate；或加 --allow-uncalibrated 让整批进人工复核队列")
    out = AS.analyze(spectra, ref, work, no_ai=a.no_ai, votes=a.votes, model_qc=a.model_qc)
    # QC warnings go through the analysis but must be seen by a human: one review-queue
    # line per substrate, same shape as AutoSpectra's own escalations (review.py list).
    import review as RV
    for s in spectra:
        if s.get("qc_verdict") == "warn":
            RV.push_queue(work, {"stage": "spectra_qc", "sample": canon_substrate(a.substrate),
                                 "spectrum": s["id"], "reason": "; ".join(s["qc_flags"]), "confidence": 0.5})
    (work / "ingest_summary.json").write_text(json.dumps(
        {"folder": a.dir, "substrate": canon_substrate(a.substrate),
         "qc_counts": c, "n_to_autospectra": len(spectra),
         "autospectra": {k: v for k, v in out.items() if k != "spectra"}},
        ensure_ascii=False, indent=2, default=float))
    print(f"AutoSpectra 完成，{len(spectra)} 条进入分析；产物在 {work}")


if __name__ == "__main__":
    main()
