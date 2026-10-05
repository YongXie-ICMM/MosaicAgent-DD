"""Guards added after the 2026-09-02 adversarial review:
 * layout validation: every image file must match; duplicate idx inside a container is an error
 * plan(): an escalate-only stage (audit) does not block downstream cache hits
 * review queue: stage-level lines are replaced per (sample, stage), cleared on a passed
   re-run, and can be resolved by a human with a note
"""
import json
import sys
from pathlib import Path

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP)); sys.path.insert(0, str(FP.parent))

import tiles                      # noqa: E402
import review as RV               # noqa: E402
import orchestrator as O          # noqa: E402
import stages                     # noqa: E402
from test_orchestrator_smoke import fake_stitch, make_segment, fake_audit, make_regions, fake_stats  # noqa: E402


def _dataset(tmp_path):
    for col, names in (("col_01_down", ["0001.png", "0002.png", "0003.png"]),
                       ("col_02_up", ["0001.png", "0002.png", "0002.5.png", "Thumbs.db"])):
        d = tmp_path / col
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"x")
    return tmp_path


def test_layout_rejects_unmatched_image_and_duplicate_idx(tmp_path):
    ds = _dataset(tmp_path)
    base = {"group_regex": r"col_(?P<index>\d+)_(?P<direction>up|down)", "serpentine": True,
            "reverse_direction_token": "up", "major_axis": "column"}
    # a regex that silently drops the re-shoot 0002.5.png: must fail (image files are 100 %)
    rep, _ = tiles._parse_layout(str(ds), {**base, "tile_regex": r"^(?P<idx>\d+)\.png$"})
    assert not rep["ok"] and rep["tiles"]["unmatched_images"] == ["col_02_up/0002.5.png"]
    assert any("图片瓦片" in e for e in rep["errors"])
    # Thumbs.db alone must not fail validation
    assert all("Thumbs" not in e for e in rep["errors"])
    # a regex that collapses 0002 and 0002.5 onto one idx: must fail
    rep, _ = tiles._parse_layout(str(ds), {**base, "tile_regex": r"^(?P<idx>\d+)(?:\.\d+)?\.png$"})
    assert not rep["ok"] and rep["tiles"]["duplicate_idx"]
    assert any("同一个 idx" in e for e in rep["errors"])
    # the right regex passes
    rep, groups = tiles._parse_layout(str(ds), {**base, "tile_regex": r"^(?P<idx>\d+(?:\.\d+)?)\.png$"})
    assert rep["ok"], rep["errors"]
    assert [len(g["entries"]) for g in groups] == [3, 3]


def test_plan_not_blocked_by_escalate_only_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "stage_stitch", fake_stitch)
    monkeypatch.setattr(stages, "stage_segment", make_segment())
    monkeypatch.setattr(stages, "stage_audit", fake_audit)
    monkeypatch.setattr(stages, "stage_regions", make_regions(split=False))
    monkeypatch.setattr(stages, "stage_stats", fake_stats)
    (tmp_path / "w.pth").write_bytes(b"w"); (tmp_path / "m.png").write_bytes(b"p")
    cfg = {"work_root": str(tmp_path / "_work"), "weights": str(tmp_path / "w.pth"),
           "tile": 512, "overlap": 64, "filler": "white", "votes": 3, "exclusion_rule": "t"}
    samples = [{"name": "S1", "mosaic_in": str(tmp_path / "m.png")}]
    O.Orchestrator(cfg).run(samples)
    # a manifest that predates the audit stage: drop its audit record
    mp = tmp_path / "_work" / "S1" / "manifest.json"
    man = json.loads(mp.read_text()); del man["stages"]["audit"]; mp.write_text(json.dumps(man))
    plan = O.Orchestrator(cfg).plan(samples)
    acts = {l["stage"]: l["action"] for l in plan["samples"][0]["stages"]}
    assert acts["audit"] == "run" and acts["regions"] == "cache" and acts["stats"] == "cache"


def test_stage_level_queue_lines(tmp_path):
    w = tmp_path
    RV.push_queue(w, {"stage": "audit", "sample": "S1", "reason": "2 alarms", "alarms": [{"a": 1}]})
    RV.push_queue(w, {"stage": "audit", "sample": "S1", "reason": "1 alarm", "alarms": [{"a": 2}]})
    q = json.loads(RV.queue_path(w).read_text())
    assert len(q) == 1 and q[0]["reason"] == "1 alarm"          # replaced, not duplicated
    RV.push_queue(w, {"stage": "regions", "sample": "S1", "reason": "票数分歧: #1",
                      "regions": [{"rank": 1, "region_key": {"sample": "S1", "cx": 1, "cy": 1, "area_px": 9}}]})
    assert RV.resolve_entry(w, "S1", "audit", "Yong", "looked at the corners, fine") == 1
    q = json.loads(RV.queue_path(w).read_text())
    assert {e["stage"]: e["status"] for e in q} == {"audit": "resolved", "regions": "open"}
    RV.push_queue(w, {"stage": "audit", "sample": "S1", "reason": "3 alarms", "alarms": []})
    RV.clear_queue(w, "S1", "audit")                              # passed re-run
    q = json.loads(RV.queue_path(w).read_text())
    assert [e["stage"] for e in q if e["status"] == "open"] == ["regions"]
