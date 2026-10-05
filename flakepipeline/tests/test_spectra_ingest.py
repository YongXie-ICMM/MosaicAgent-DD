"""spectra_ingest: the verbatim loader port and the closed-form QC, on synthetic files
laid out exactly like the spectrometer exports (no paper data in the repo).

Layout facts these tests pin down (from Spectra/process_dr.py and the real workbooks):
  * xlsx: row 0 = group labels (A, B, 1-B/A, -(1-B/A)), row 1 = per-column headers
    starting "X [eV]"; the loader must take the "1-B/A" group, not the last "X [" column.
  * csv: one header line, wavelength [nm] in column 0, dR/R in column 3.
  * same workbook number as xlsx and csv -> xlsx wins; "-2." files are reference spots.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import autospectra as AS          # noqa: E402
import spectra_ingest as SI       # noqa: E402


def synth(EA=1.88, EB=2.02, aA=0.6, aB=0.5, noise=0.004, seed=0, n=1300, lo=1.50, hi=2.55):
    """A monolayer-like differential spectrum on the instrument's energy range; the
    instrument exports the negative of it (sign flipped), which orient() must undo."""
    rng = np.random.default_rng(seed)
    x = np.linspace(lo, hi, n)
    y = (0.15 + 0.25 * (x - 1.55) + AS.gauss(x, aA, EA, 0.03) + AS.gauss(x, aB, EB, 0.035)
         + 0.9 * AS.gauss(x, 1.0, 2.28, 0.12)                     # C exciton, sets the sign
         + rng.normal(0, noise, x.size))
    return x, y


def write_xlsx(path: Path, x, y):
    """Three column groups; the wanted one is the middle, so 'last X column' would be wrong."""
    n = len(x)
    cols = {}
    labels = ["A", None, "B", None, "1-B/A", None, "-(1-B/A)", None]
    heads = ["X [eV]", "Y [a.u.]"] * 4
    data = [x, 0.5 + 0 * y, x, 0.4 + 0 * y, x, -y, x, y]     # instrument sign is flipped
    for i, (lab, h, d) in enumerate(zip(labels, heads, data)):
        cols[i] = [lab, h] + list(d)
    pd.DataFrame(cols).to_excel(path, header=False, index=False)


def write_csv(path: Path, x, y):
    wl = 1239.842 / x
    df = pd.DataFrame({0: wl, 1: 0 * y, 2: 0 * y, 3: -y})
    with open(path, "w") as f:
        f.write("Wavelength,R_sample,R_ref,dR/R\n")
        df.to_csv(f, header=False, index=False)


# ------------------------------------------------------------------ loaders
def test_xlsx_takes_the_1BA_group_and_cleans(tmp_path):
    x, y = synth()
    write_xlsx(tmp_path / "工作簿3.xlsx", x, y)
    xl, yl = SI.load_xlsx(tmp_path / "工作簿3.xlsx")
    assert xl.min() > SI.ELO and xl.max() < SI.EHI          # clipped to the cleaning window
    xe, ye = SI.clean(x, -y)                                  # same cleaning applied to the truth
    assert np.allclose(xl, xe) and np.allclose(yl, ye)        # the 1-B/A column (flipped), not A or B


def test_csv_converts_nm_to_eV(tmp_path):
    x, y = synth()
    write_csv(tmp_path / "工作簿4.csv", x, y)
    xc, yc = SI.load_csv(tmp_path / "工作簿4.csv")
    xe, ye = SI.clean(1239.842 / (1239.842 / x), -y)          # nm round trip, same cleaning
    assert np.allclose(xc, xe, atol=1e-9) and np.allclose(yc, ye)


def test_orient_makes_C_exciton_positive():
    x, y = synth()
    yp, sgn = SI.orient(x, -y)
    assert sgn == -1.0 and np.allclose(yp, y)
    yp2, sgn2 = SI.orient(x, y)
    assert sgn2 == 1.0 and np.allclose(yp2, y)


def test_list_files_dedup_and_reference_spots(tmp_path):
    x, y = synth()
    write_xlsx(tmp_path / "工作簿1.xlsx", x, y)
    write_csv(tmp_path / "工作簿1.csv", x, y)                  # same number: xlsx wins
    write_csv(tmp_path / "工作簿2.csv", x, y)
    write_xlsx(tmp_path / "工作簿5-2.xlsx", x, y)              # on-monolayer reference spot
    names = [Path(f).name for f in SI.list_files(str(tmp_path))]
    assert names == ["工作簿1.xlsx", "工作簿2.csv"]


# ------------------------------------------------------------------ QC
def _templates():
    x, y = synth()
    yp, _ = SI.orient(x, -y)
    tpl, grid = SI.shape_templates([{"substrate": "Sapphire", "label": "1L", "x": x, "ypos": yp}])
    return SI.templates_for(tpl, "Sapphire"), grid


def test_qc_accepts_a_clean_spectrum():
    x, y = synth(seed=1)
    T, grid = _templates()
    q = SI.qc_spectrum(x, y, AS.fit_ab(x, y), T, grid)
    assert q["verdict"] == "accept", q
    assert q["metrics"]["shape_cc"] > 0.99 and q["metrics"]["snr"] > 5


def test_qc_rejects_short_trace_and_missing_window():
    x, y = synth(n=200)
    q = SI.qc_spectrum(x, y, None)
    assert q["verdict"] == "reject" and any("points" in f for f in q["flags"])
    x, y = synth(lo=1.80, hi=2.55)                             # starts inside the fit window
    q = SI.qc_spectrum(x, y, AS.fit_ab(x, y))
    assert q["verdict"] == "reject" and any("fit window" in f for f in q["flags"])


def test_qc_rejects_saturation():
    x, y = synth()
    y = np.minimum(y, np.percentile(y, 97))                    # ADC clipping: a flat top
    q = SI.qc_spectrum(x, y, AS.fit_ab(x, y))
    assert q["verdict"] == "reject" and any("saturated" in f for f in q["flags"])


def test_qc_flags_noise_and_foreign_shape():
    T, grid = _templates()
    x, y = synth(noise=0.35, seed=3)                           # too noisy: fit fails or snr low
    q = SI.qc_spectrum(x, y, AS.fit_ab(x, y), T, grid)
    assert q["verdict"] == "reject"
    x, y = synth()
    foreign = 0.2 + 0.5 * np.sin(6 * x)                        # not a MoS2 spectrum
    q = SI.qc_spectrum(x, foreign, AS.fit_ab(x, foreign), T, grid)
    assert q["verdict"] == "reject" and any("shape_cc" in f for f in q["flags"])


# ------------------------------------------------------------------ end to end
def test_ingest_dir_end_to_end(tmp_path):
    x, y = synth()
    write_xlsx(tmp_path / "工作簿1.xlsx", x, y)
    write_csv(tmp_path / "工作簿2.csv", x, y)
    (tmp_path / "工作簿3.xlsx").write_bytes(b"not a workbook")   # must not vanish silently
    labels = {1: "1L", "工作簿2.csv": "1L"}
    spectra, qc = SI.ingest_dir(str(tmp_path), "sapphire", labels)
    assert {r["id"]: r["verdict"] for r in qc} == {"工作簿1.xlsx": "accept", "工作簿2.csv": "accept",
                                                  "工作簿3.xlsx": "reject"}
    assert len(spectra) == 2 and all(s["label"] == "1L" for s in spectra)
    assert spectra[0]["substrate"] == "Sapphire"                # alias canonicalised
    assert SI.canon_substrate("276nm") == SI.canon_substrate("285nm") == "276 nm SiO2/Si"
    assert SI.canon_substrate("67nm") == SI.canon_substrate("70nm") == "67 nm SiO2/Si"
    assert np.all(spectra[0]["y"] == SI.orient(*SI.load_xlsx(tmp_path / "工作簿1.xlsx"))[0])
    p = SI.write_qc(qc, tmp_path / "work", str(tmp_path), "sapphire")
    out = json.loads(p.read_text())
    assert out["counts"] == {"accept": 2, "warn": 0, "reject": 1}
    assert out["re_measure_now"][0]["file"] == "工作簿3.xlsx"


def test_ingest_feeds_autospectra_without_a_model(tmp_path):
    x, y = synth()
    write_xlsx(tmp_path / "工作簿1.xlsx", x, y)
    spectra, _ = SI.ingest_dir(str(tmp_path), "Sapphire")
    out = AS.analyze(spectra, AS.DEFAULT_REF, tmp_path / "work", no_ai=True, escalate_to_queue=False)
    assert out["n"] == 1 and out["fitted"] == 1


def test_review_queue_keeps_one_line_per_spectrum(tmp_path):
    """Regression for the 2026-09-05 real-data run: spectrum-level escalations used to fall
    into the stage-level rule and overwrite each other, leaving one line per substrate."""
    import review as RV
    for sid in ("1L/工作簿1.xlsx", "1L/工作簿2.xlsx"):
        RV.push_queue(tmp_path, {"stage": "spectra_qc", "sample": "Sapphire", "spectrum": sid,
                                 "reason": "snr 4.8 < 5.0", "confidence": 0.5})
    RV.push_queue(tmp_path, {"stage": "spectra_qc", "sample": "Sapphire", "spectrum": "1L/工作簿2.xlsx",
                             "reason": "snr 4.9 < 5.0", "confidence": 0.5})       # re-run: replaces, no duplicate
    q = json.loads(RV.queue_path(tmp_path).read_text())
    assert sorted(e["spectrum"] for e in q) == ["1L/工作簿1.xlsx", "1L/工作簿2.xlsx"]
    assert [e["reason"] for e in q if e["spectrum"] == "1L/工作簿2.xlsx"] == ["snr 4.9 < 5.0"]
