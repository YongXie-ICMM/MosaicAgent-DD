"""tools/handover.py on a packed fixture: two runs of one scan, an interruption, a resumed run with a colour step."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "acquisition" / "Auto_Scan"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
import handover as H  # noqa: E402
import pack_handover as ph  # noqa: E402
from handover_fixture import build_scan_dir, NCOLS, NROWS  # noqa: E402

quiet = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    base = tmp_path_factory.mktemp("h")
    scan = build_scan_dir(base)
    folder = ph.pack(scan["scan_dir"], ph.list_sessions(scan["scan_dir"]), base / "out", operator="student", notes="stopped once", log=quiet)
    return {"folder": folder, "scan": scan}


def options(**over):
    base = {"verify": {"ignore": False}, "assemble": {"prefer": "latest"},
            "colour": {"workers": 2, "scale_div": 2, "out_scale": 0.5, "full": False, "no_flat_field": False, "gain_mode": None, "no_ai": True},
            "stitch": {"no_ai": True, "workers": 2, "full": False, "dry_run": True}, "check": {"samples": 4}}
    for k, v in over.items():
        base[k].update(v)
    return base


def test_context_requires_a_handover_manifest(tmp_path):
    with pytest.raises(H.HandoverError, match="handover.json"):
        H.Context(tmp_path, options=options(), log=quiet)


def test_verify_checks_every_file_and_the_runtime(packed):
    ctx = H.Context(packed["folder"], options=options(), log=quiet)
    res = H.run_step(ctx, "verify")
    v = json.loads((ctx.analysis / "verify.json").read_text(encoding="utf-8"))
    assert res["status"] == "done" and v["ok"] is True
    assert v["files_checked"] == v["files_listed"] > 20 and v["missing"] == [] and v["mismatched"] == []
    assert v["scanner_runtime"]["matches_repository"] is True
    assert all(r["tiles_hash_mismatch"] == [] and r["tiles_missing"] == [] for r in v["runs"])
    # the second call is skipped because nothing changed
    assert H.run_step(ctx, "verify")["finished_at_utc"] == res["finished_at_utc"]


def test_verify_detects_a_changed_tile(packed, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(packed["folder"], copy)
    run2 = copy / "runs" / packed["scan"]["stamp2"]
    tile = next(p for p in run2.iterdir() if p.name.startswith("mosaic_r"))
    tile.write_bytes(tile.read_bytes()[:-10] + b"0123456789")
    ctx = H.Context(copy, options=options(), log=quiet)
    with pytest.raises(H.HandoverError, match="does not match"):
        H.run_step(ctx, "verify")
    v = json.loads((ctx.analysis / "verify.json").read_text(encoding="utf-8")) if (ctx.analysis / "verify.json").exists() else None
    assert ctx.status["steps"]["verify"]["status"] == "failed" and v is None
    ctx2 = H.Context(copy, options=options(verify={"ignore": True}), log=quiet)
    H.run_step(ctx2, "verify")
    v = json.loads((ctx2.analysis / "verify.json").read_text(encoding="utf-8"))
    assert v["ok"] is False and v["mismatched"] == [f"runs/{run2.name}/{tile.name}"]
    assert any(r["tiles_hash_mismatch"] == [tile.name] for r in v["runs"])


def test_assemble_merges_runs_by_original_numbers_with_provenance(packed):
    scan = packed["scan"]
    ctx = H.Context(packed["folder"], options=options(), log=quiet)
    H.run_step(ctx, "verify")
    H.run_step(ctx, "assemble")
    a = json.loads((ctx.analysis / "assemble.json").read_text(encoding="utf-8"))
    assert a["grid"]["nx"] == NCOLS and a["grid"]["ny"] == NROWS and a["grid"]["invert_x"] is True
    assert a["complete"] is True and a["tiles"] == NCOLS * NROWS and a["missing_positions"] == []
    assert a["runs_assembled"] == [scan["stamp1"], scan["stamp2"]]
    # the point run 1 missed and run 2 re-shot: no duplicate (run 1 never captured it)
    assert a["duplicates"] == []
    assert a["tiles_by_run"] == {scan["stamp1"]: len(scan["points1"]) - 1, scan["stamp2"]: len(scan["points2"])}
    flat = ctx.analysis / "flat_grid"
    names = sorted(p.name for p in flat.iterdir() if p.name.startswith("mosaic_r"))
    assert len(names) == NCOLS * NROWS
    # tiles are the same bytes as the originals (hard links where possible)
    rec = next(t for t in a["tile_records"] if t["source_run"] == scan["stamp2"])
    src = packed["folder"] / "runs" / scan["stamp2"] / rec["source_filename"]
    assert (flat / rec["filename"]).read_bytes() == src.read_bytes()
    assert sum(a["link_method"].values()) == NCOLS * NROWS
    # derived records carry the provenance and pass the stitcher's acquisition check
    session = json.loads((flat / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "assembled_from_runs" and session["position_trusted"] is False
    assert [r["run"] for r in session["assembled_from"]] == [scan["stamp1"], scan["stamp2"]]
    events = [json.loads(l) for l in (flat / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    caps = [e for e in events if e["event"] == "capture_success"]
    assert len(caps) == NCOLS * NROWS and all(e["assembled_from"]["run"] in (scan["stamp1"], scan["stamp2"]) for e in caps)
    assert a["acquisition_check"]["checked"] and a["acquisition_check"]["checksum_verified"] == NCOLS * NROWS
    seams = a["cross_run_seams"]
    assert seams["checked"] and seams["pairs"] >= NROWS and seams["ok"] is True


def test_assemble_prefers_the_chosen_run_for_duplicates(packed, tmp_path):
    """A position captured by both runs: latest wins by default, earliest on request."""
    import shutil
    scan = packed["scan"]
    copy = tmp_path / "dup"
    shutil.copytree(packed["folder"], copy)
    # make run 1 also contain the tile that run 2 re-shot (copy run 2's file + a capture event into run 1)
    r, c = scan["missing_point"]
    run1, run2 = copy / "runs" / scan["stamp1"], copy / "runs" / scan["stamp2"]
    name = f"mosaic_r{r}_c{c}.png"
    data = (run2 / name).read_bytes()[:-4] + b"\x00\x00\x00\x00"       # different bytes
    (run1 / name).write_bytes(data)
    import hashlib
    ev = {"event": "capture_success", "timestamp_utc": "2026-10-06T06:30:00+00:00", "session_id": "s1", "event_id": "dup1", "sequence": 999,
          "row": r, "col": c, "filename": name, "image_relative_path": name, "image_sha256": hashlib.sha256(data).hexdigest(), "image_bytes": len(data)}
    with (run1 / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    ctx = H.Context(copy, options=options(verify={"ignore": True}), log=quiet)
    H.run_step(ctx, "verify")
    H.run_step(ctx, "assemble")
    a = json.loads((ctx.analysis / "assemble.json").read_text(encoding="utf-8"))
    assert len(a["duplicates"]) == 1 and a["duplicates"][0]["kept"] == scan["stamp2"] and a["duplicates"][0]["identical"] is False
    kept = next(t for t in a["tile_records"] if t["row"] == r and t["col"] == c)
    assert kept["source_run"] == scan["stamp2"]
    ctx2 = H.Context(copy, options=options(verify={"ignore": True}, assemble={"prefer": "earliest"}), log=quiet)
    H.run_step(ctx2, "assemble", force=True)
    a2 = json.loads((ctx2.analysis / "assemble.json").read_text(encoding="utf-8"))
    kept2 = next(t for t in a2["tile_records"] if t["row"] == r and t["col"] == c)
    assert kept2["source_run"] == scan["stamp1"] and a2["duplicates"][0]["kept"] == scan["stamp1"]


def test_incidents_timeline_from_events_and_console(packed):
    scan = packed["scan"]
    ctx = H.Context(packed["folder"], options=options(), log=quiet)
    H.run_step(ctx, "verify")
    H.run_step(ctx, "assemble")
    H.run_step(ctx, "incidents")
    inc = json.loads((ctx.analysis / "incidents.json").read_text(encoding="utf-8"))
    fatal = [i for i in inc["incidents"] if i["event"] == "move_failed"]
    assert len(fatal) == 1 and fatal[0]["run"] == scan["stamp1"] and fatal[0]["point"] == len(scan["points1"]) - 1
    assert fatal[0]["planned"] == len(scan["points1"]) and 0.9 < fatal[0]["fraction_of_run"] < 1
    assert fatal[0]["elapsed_from_first_capture_min"] is not None and fatal[0]["error"].startswith("Y movement failed")
    console = [i for i in inc["incidents"] if i["source"].startswith("launch_")]
    assert console and console[0]["run"] == scan["stamp1"] and "rc=-1" in console[0]["error"]
    assert inc["restart_gaps"] and inc["restart_gaps"][0]["from_run"] == scan["stamp1"]
    assert inc["missing_console_logs"] == []
    assert [r["camera"]["gain"] for r in inc["runs"]] == [21, 27]
    with (ctx.analysis / "incidents.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(r["event"] == "move_failed" for r in rows)
    md = (ctx.analysis / "incidents_zh.md").read_text(encoding="utf-8")
    assert "move_failed" in md and scan["stamp2"] in md


def test_colour_check_and_report_chain_with_dry_run_stitch(packed):
    scan = packed["scan"]
    ctx = H.Context(packed["folder"], options=options(), log=quiet)
    H.run_all(ctx)
    status = json.loads((ctx.analysis / "handover_status.json").read_text(encoding="utf-8"))
    assert [status["steps"][s]["status"] for s in H.STEPS] == ["done"] * len(H.STEPS)
    colour = json.loads((ctx.analysis / "colour.json").read_text(encoding="utf-8"))
    assert colour["skipped"] is False and (ctx.analysis / "colour_matched" / "colour_match_report.json").is_file()
    # the resumed run's column is found as a step and corrected towards the reference run
    steps = [(s["col_a"], s["col_b"]) for s in colour["column_steps"]]
    assert (NCOLS - 2, NCOLS - 1) in steps
    last = np.array(colour["column_level_gain_rgb"][str(NCOLS - 1)])
    ref = np.mean([colour["column_level_gain_rgb"][str(c)] for c in colour["reference_segment"]], axis=0)
    assert np.allclose(last / ref, scan["gain"], atol=0.02)
    stitch = json.loads((ctx.analysis / "stitch.json").read_text(encoding="utf-8"))
    assert stitch["skipped"] is True and "run_stitch.py" in " ".join(stitch["command"]) and "--no-ai" in stitch["command"]
    check = json.loads((ctx.analysis / "colour_check.json").read_text(encoding="utf-8"))
    assert [r["run"] for r in check["runs"]] == [scan["stamp1"], scan["stamp2"]]
    assert all(r["checked"] == 4 and r["original"]["verdict"] and r["corrected"]["verdict"] for r in check["runs"])
    assert check["white_balance_records"][0]["calibrated"] is True
    report = (ctx.analysis / "REPORT_zh.md").read_text(encoding="utf-8")
    assert "FIXTURE_5mg" in report and "move_failed" in report and "跳过：dry run" in report
    # re-running changes nothing: every step is skipped
    before = {s: status["steps"][s]["finished_at_utc"] for s in H.STEPS}
    ctx2 = H.Context(packed["folder"], options=options(), log=quiet)
    H.run_all(ctx2)
    after = {s: ctx2.status["steps"][s]["finished_at_utc"] for s in H.STEPS}
    assert before == after
    # redoing from 'incidents' re-runs the steps after it, not before
    ctx3 = H.Context(packed["folder"], options=options(), log=quiet)
    H.run_all(ctx3, start="incidents")
    assert ctx3.status["steps"]["assemble"]["finished_at_utc"] == before["assemble"]
    assert ctx3.status["steps"]["incidents"]["finished_at_utc"] != before["incidents"]
    assert ctx3.status["steps"]["report"]["finished_at_utc"] != before["report"]


def test_real_stitch_on_the_fixture(packed):
    """The stitcher runs offline on the assembled, colour-matched fixture grid."""
    ctx = H.Context(packed["folder"], options=options(stitch={"dry_run": False}), log=quiet)
    H.run_all(ctx)
    stitch = json.loads((ctx.analysis / "stitch.json").read_text(encoding="utf-8"))
    assert stitch["skipped"] is False and Path(stitch["mosaic"]).is_file() and (ctx.analysis / "mosaic_preview.jpg").is_file()
    assert stitch["tiles_used"] == NCOLS * NROWS and stitch["n_components"] == 1
    report = (ctx.analysis / "REPORT_zh.md").read_text(encoding="utf-8")
    assert "mosaic_preview.jpg" in report


def test_cli_status_and_single_step(packed, capsys):
    assert H.main(["status", str(packed["folder"])]) == 0
    out = capsys.readouterr().out
    assert "verify" in out and "report" in out
    assert H.main(["incidents", str(packed["folder"])]) == 0
