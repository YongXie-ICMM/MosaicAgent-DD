"""pack_handover.py on a fake Auto_Scan tree written with the scanner's own record classes."""
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tests"))
import pack_handover as ph  # noqa: E402
from handover_fixture import build_scan_dir, count_files  # noqa: E402


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    return build_scan_dir(tmp_path_factory.mktemp("scan"))


def test_lists_and_selects_sessions(scan):
    sessions = ph.list_sessions(scan["scan_dir"])
    assert [s["stamp"] for s in sessions] == [scan["stamp1"], scan["stamp2"]]
    assert ph.select_sessions(sessions, today=True) == sessions
    assert ph.select_sessions(sessions, since=date.today() + timedelta(days=1)) == []
    assert [s["stamp"] for s in ph.select_sessions(sessions, names=[scan["stamp2"]])] == [scan["stamp2"]]
    with pytest.raises(ph.PackError, match="Unknown session"):
        ph.select_sessions(sessions, names=["nope"])


def test_run_description_has_grid_counts_camera_and_incidents(scan):
    sessions = {s["stamp"]: s for s in ph.list_sessions(scan["scan_dir"])}
    r1 = ph.describe_run(sessions[scan["stamp1"]])
    assert r1["status"] == "stopped_on_error" and r1["grid"]["nx"] == 5 and r1["grid"]["invert_x"] is True
    assert r1["photos_saved"] == len(scan["points1"]) - 1 == r1["tiles_on_disk"]
    assert r1["cols"] == [0, 3] and r1["camera"]["gain"] == 21
    kinds = [i["event"] for i in r1["incidents"]]
    assert "move_failed" in kinds and "session_finished" in kinds
    failed = next(i for i in r1["incidents"] if i["event"] == "move_failed")
    assert failed["error"].startswith("Y movement failed: rc=-1") and failed["elapsed_min"] is not None
    assert list(failed["verification"]) == ["before", "after"]
    assert r1["event_counts"]["capture_success"] == r1["photos_saved"]
    r2 = ph.describe_run(sessions[scan["stamp2"]])
    assert r2["status"] == "completed" and r2["camera"]["gain"] == 27 and r2["cols"] == [3, 4]


def test_pack_copies_everything_intact_with_manifest(scan, tmp_path):
    out = tmp_path / "out"
    sessions = ph.list_sessions(scan["scan_dir"])
    folder = ph.pack(scan["scan_dir"], sessions, out, operator="student A", notes="stopped once", log=lambda *_: None)
    assert folder.name.startswith("handover_FIXTURE_5mg_")
    manifest = json.loads((folder / "handover.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "scan_handover" and manifest["operator"] == "student A" and manifest["sample_id"] == "FIXTURE_5mg"
    assert manifest["scanner_runtime_check"]["ok"] is True and manifest["scanner_runtime_check"]["files_checked"] == 16
    assert [r["folder"] for r in manifest["runs"]] == [scan["stamp1"], scan["stamp2"]]
    for r in manifest["runs"]:
        assert r["tile_check"]["hash_mismatch"] == [] and r["tile_check"]["recorded_but_missing"] == []
        assert r["tile_check"]["recorded_in_events"] == r["tiles_on_disk"]
    # runs copied intact: same file count, program_snapshot included, session/events byte-identical
    for key in ("run1", "run2"):
        src = scan[key]
        dst = folder / "runs" / src.name
        assert count_files(src) == count_files(dst)
        assert (dst / "program_snapshot" / "acquisition_contract.py").is_file()
        assert (dst / "events.jsonl").read_bytes() == (src / "events.jsonl").read_bytes()
    # companions
    assert len(manifest["console_logs"]) == 2 and manifest["console_logs"][0]["flagged_lines"] >= 1
    assert "rc=-1" in manifest["console_logs"][0]["flagged"][0]["text"]
    assert len(manifest["shared_history"]) == 1 and manifest["shared_history"][0]["status"] == "closed"
    assert (folder / "camera_history" / "camera_connections.jsonl").is_file()
    assert manifest["colour_calibration"][0]["calibrated"] is True and (folder / "colour_calibration" / "latest.json").is_file()
    assert (folder / "scanner" / "delivery_manifest.json").is_file()
    # every packed file listed with a correct hash; handover.json itself not listed
    listed = {f["path"]: f for f in manifest["files"]}
    assert "handover.json" not in listed and "README_zh.txt" not in listed
    for rel, rec in list(listed.items())[:50]:
        p = folder / rel
        assert p.is_file() and p.stat().st_size == rec["bytes"] and ph.sha256_file(p) == rec["sha256"]
    assert manifest["totals"]["files"] == len(listed) == count_files(folder) - 2
    assert "FIXTURE_5mg" in (folder / "README_zh.txt").read_text(encoding="utf-8")
    # a second pack of the same day does not overwrite the first
    again = ph.pack(scan["scan_dir"], sessions, out, log=lambda *_: None)
    assert again != folder and again.name.endswith("_2")


def test_pack_zip_is_stored_and_checksummed(scan, tmp_path):
    out = tmp_path / "out"
    folder = ph.pack(scan["scan_dir"], ph.list_sessions(scan["scan_dir"])[:1], out, make_zip=True, log=lambda *_: None)
    z = folder.with_suffix(".zip")
    assert z.is_file()
    digest, name = folder.with_suffix(".zip.sha256").read_text().split()
    assert name == z.name and digest == ph.sha256_file(z)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert all(n.startswith(folder.name + "/") for n in names)
        assert folder.name + "/handover.json" in names
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())


def test_refuses_when_disk_space_is_short(scan, tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "free_bytes", lambda _p: 0)
    with pytest.raises(ph.PackError, match="free space"):
        ph.pack(scan["scan_dir"], ph.list_sessions(scan["scan_dir"]), tmp_path / "x", log=lambda *_: None)


def test_runtime_check_flags_a_modified_file(scan, tmp_path):
    import shutil
    copy = tmp_path / "scan_copy"
    shutil.copytree(scan["scan_dir"], copy)
    (copy / "gui_theme.py").write_text("# changed\n", encoding="utf-8")
    result = ph.check_runtime(copy)
    assert result["ok"] is False and result["mismatched"] == ["gui_theme.py"]


def test_cli_list_and_unattended_pack(scan, tmp_path, capsys):
    assert ph.main(["--scan-dir", str(scan["scan_dir"]), "--list"]) == 0
    out = capsys.readouterr().out
    assert scan["stamp1"] in out and "sample=FIXTURE_5mg" in out
    code = ph.main(["--scan-dir", str(scan["scan_dir"]), "--today", "--yes", "--out", str(tmp_path / "h")])
    assert code == 0
    folders = list((tmp_path / "h").iterdir())
    assert len(folders) == 1 and (folders[0] / "handover.json").is_file()
