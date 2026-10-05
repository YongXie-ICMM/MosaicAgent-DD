"""Step-1 layout planner on the stitcher's default path (run_stitch.py + tiles.py).

Everything here is closed-form and offline. Kimi is replaced by a fake pool that hands
back a hand-written layout, so no test needs KIMI_API_KEY or the network -- a key IS
resolvable on some machines through a .env file, and the fakes guarantee it is never
used (any real call raises).

What it pins down:
  * regression: the hard-coded Figure-3 naming (a_L<col>_<up|down>.zip, float indices
    from re-shoots, the one zip whose inner folder is named like the zip) still yields
    the same tiles, same order, same fields, and a hand-written layout equivalent to
    that scheme reproduces them field for field;
  * a dataset with a different naming scheme (col_03_up/img_0007.png, as zips or as
    plain folders) is scanned from a layout dict: serpentine directions from the
    container token, float indices, readable downstream via load_full;
  * validation rejects a wrong tile regex / group regex and lists every unmatched
    name, rejects an uncompilable regex and a missing named group, and never falls
    back to the hard-coded rule;
  * run_stitch.main(): the Figure-3 dataset never calls the planner; an unknown
    dataset asks the fake Kimi exactly once, saves work/layout.json with the
    validation report and anomalies, and the second run reuses it without calling
    step_plan; --layout <json> is honoured (and validated); with no key and no cache
    the run stops with a non-zero exit that names --layout; --plan validates and saves.

Run:  cd MosaicAgent && python3 -m pytest flakepipeline/tests -q
"""
import json
import sys
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tiles as T                 # noqa: E402
import run_stitch as RS           # noqa: E402

PNG = cv2.imencode(".png", np.full((8, 12, 3), 128, np.uint8))[1].tobytes()


# ------------------------------------------------------------------ fixtures
def _zip(path, inner_names):
    with zipfile.ZipFile(path, "w") as z:
        for n in inner_names:
            z.writestr(n, PNG if n.lower().endswith(".png") else b"junk")


def _fields(t):
    return (t.col, t.col_idx, t.direction, t.order, t.key, t.name, t.zip_path,
            t.inner, t.nominal_row, t.nbytes)


@pytest.fixture
def fig3_dir(tmp_path):
    """A miniature of the real Figure-3 listing with its traps: a float re-shoot index
    in L19, an extra tile in L20, the inner folder of a_L21_up named like the zip,
    junk inside a zip (Thumbs.db, __MACOSX), junk beside the zips."""
    d = tmp_path / "fig3"
    d.mkdir()
    _zip(d / "a_L18_down.zip", [f"L18_down/{k:04d}.png" for k in range(1, 6)]
         + ["L18_down/Thumbs.db", "__MACOSX/L18_down/._0001.png"])
    _zip(d / "a_L19_up.zip", ["L19_up/0001.png", "L19_up/0002.png", "L19_up/0003.5.png",
                              "L19_up/0003.6.png", "L19_up/0004.png", "L19_up/0005.png"])
    _zip(d / "a_L20_down.zip", [f"L20_down/{k:04d}.png" for k in range(1, 7)])
    _zip(d / "a_L21_up.zip", [f"a_L21_up/{k:04d}.png" for k in range(1, 6)])
    (d / "notes.txt").write_text("not a tile")
    (d / ".DS_Store").write_bytes(b"\x00")
    (d / "_stitch_work").mkdir()
    return d


def _expected_fig3(d):
    """The documented rule written out by hand, not by calling scan_dataset."""
    spec = [(18, "down", "L18_down", ["0001", "0002", "0003", "0004", "0005"]),
            (19, "up", "L19_up", ["0001", "0002", "0003.5", "0003.6", "0004", "0005"]),
            (20, "down", "L20_down", ["0001", "0002", "0003", "0004", "0005", "0006"]),
            (21, "up", "a_L21_up", ["0001", "0002", "0003", "0004", "0005"])]
    out = []
    for col_idx, (col, direction, folder, keys) in enumerate(spec):
        n = len(keys)
        for order, s in enumerate(keys):
            row = order if direction == "down" else n - 1 - order
            out.append((col, col_idx, direction, order, float(s), f"{s}.png",
                        str(d / f"a_L{col}_{direction}.zip"), f"{folder}/{s}.png",
                        row, len(PNG)))
    return out


FIG3_LAYOUT = {
    "group_regex": r"^a_L(?P<index>\d+)_(?P<direction>up|down)\.zip$",
    "tile_regex": r"^(?P<idx>\d+(?:\.\d+)?)\.png$",
    "serpentine": True, "major_axis": "column", "reverse_direction_token": "up",
    "anomalies": ["a_L19_up 有 0003.5/0003.6 补拍", "a_L21_up 内层文件夹叫 a_L21_up/"],
    "confidence": 0.9, "notes": "test",
}

NEW_COLS = {"col_03_up": ["0007", "0008", "0009", "0010", "0011"],
            "col_04_dn": ["0007", "0008", "0009", "0010", "0011"],
            "col_05_up": ["0007", "0008", "0009", "0009.5", "0010"]}

NEW_LAYOUT = {
    "group_regex": r"^col_(?P<index>\d+)_(?P<direction>up|dn)$",
    "tile_regex": r"^img_(?P<idx>\d+(?:\.\d+)?)\.png$",
    "serpentine": True, "major_axis": "column", "reverse_direction_token": "up",
    "anomalies": ["col_05_up 有 img_0009.5.png 补拍"], "confidence": 0.8, "notes": "n",
}


def _new_scheme(d, as_zip):
    for cname, keys in NEW_COLS.items():
        inner = [f"{cname}/img_{k}.png" for k in keys]
        if as_zip:
            _zip(d / f"{cname}.zip", inner)
        else:
            for n in inner:
                p = d / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(PNG)


@pytest.fixture(params=["zip", "folder"])
def scheme(request):
    """How the new-scheme dataset is packaged: one zip per column, or plain folders."""
    return request.param


@pytest.fixture
def new_dir(tmp_path, scheme):
    d = tmp_path / "new"
    d.mkdir()
    _new_scheme(d, as_zip=scheme == "zip")
    return d


# ------------------------------------------------------------------ tiles.py
def test_hardcoded_scan_unchanged(fig3_dir):
    ts = T.scan_dataset(str(fig3_dir))
    assert [_fields(t) for t in ts] == _expected_fig3(fig3_dir)
    assert [t.tid for t in ts][:2] == ["L18_down/0001.png", "L18_down/0002.png"]
    assert all(t.flags == [] and np.isnan(t.focus) for t in ts)
    assert T.scan_dataset(str(fig3_dir), layout=None) == ts


def test_layout_equivalent_to_hardcoded(fig3_dir):
    rep = T.validate_layout(str(fig3_dir), FIG3_LAYOUT)
    assert rep["ok"] and rep["errors"] == [], rep
    assert rep["containers"]["total"] == 4 and rep["containers"]["matched"] == 4
    assert rep["containers"]["ignored"] == [".DS_Store", "_stitch_work", "notes.txt"]
    # Thumbs.db is a real unmatched file (1/23 stays above 95%); ._ files are skipped
    assert rep["tiles"]["unmatched"] == ["a_L18_down.zip/L18_down/Thumbs.db"]
    assert rep["tiles"]["skipped"] == ["a_L18_down.zip/__MACOSX/L18_down/._0001.png"]
    assert [(c["index"], c["direction"], c["n_tiles"]) for c in rep["columns"]] == \
        [(18, "down", 5), (19, "up", 6), (20, "down", 6), (21, "up", 5)]
    ts = T.scan_dataset(str(fig3_dir), layout=FIG3_LAYOUT)
    assert [_fields(t) for t in ts] == [_fields(t) for t in T.scan_dataset(str(fig3_dir))]


def test_layout_scan_new_scheme(new_dir, scheme):
    rep = T.validate_layout(str(new_dir), NEW_LAYOUT)
    assert rep["ok"] and rep["errors"] == [], rep
    # the first container (col_03_up) carries the reverse token: filenames cannot settle
    # whether "up" really is the reverse direction, so this is a warning, never an error
    assert len(rep["warnings"]) == 1 and "第一个容器（index 最小）被判为反向" in rep["warnings"][0]
    assert "col_03_up" in rep["warnings"][0]
    assert [(c["index"], c["direction"], c["n_tiles"]) for c in rep["columns"]] == \
        [(3, "up", 5), (4, "down", 5), (5, "up", 5)]
    ts = T.scan_dataset(str(new_dir), layout=NEW_LAYOUT)
    assert len(ts) == 15
    assert [(t.col, t.col_idx, t.direction) for t in ts[::5]] == \
        [(3, 0, "up"), (4, 1, "down"), (5, 2, "up")]
    c3 = [t for t in ts if t.col == 3]
    assert [t.order for t in c3] == [0, 1, 2, 3, 4]
    assert [t.nominal_row for t in c3] == [4, 3, 2, 1, 0]          # reversed column
    assert [t.nominal_row for t in ts if t.col == 4] == [0, 1, 2, 3, 4]
    assert [t.key for t in ts if t.col == 5] == [7.0, 8.0, 9.0, 9.5, 10.0]
    assert ts[0].tid == "L3_up/img_0007.png" and ts[0].nbytes == len(PNG)
    # zip_path is the container (the zip file, or the column folder) and inner is the
    # path *inside* it -- the pair _read_raw joins, so it must stay readable either way
    for t in ts:
        cname = f"col_{t.col:02d}_{'up' if t.direction == 'up' else 'dn'}"
        if scheme == "zip":
            assert (t.zip_path, t.inner) == (str(new_dir / f"{cname}.zip"), f"{cname}/{t.name}")
        else:
            assert (t.zip_path, t.inner) == (str(new_dir / cname), t.name)
        assert T._read_raw(t.zip_path, t.inner) == PNG
    # zip or plain folder, the container must be readable downstream
    im = T.load_full(ts[-1])
    assert im is not None and im.shape == (8, 12, 3)


def test_wrong_tile_regex_rejected_with_every_name(new_dir):
    bad = {**NEW_LAYOUT, "tile_regex": r"^img_(?P<idx>\d{3})\.png$"}
    rep = T.validate_layout(str(new_dir), bad)
    assert not rep["ok"]
    assert rep["tiles"]["total"] == 15 and rep["tiles"]["matched"] == 0
    assert len(rep["tiles"]["unmatched"]) == 15
    assert any("tile_regex" in e for e in rep["errors"])
    with pytest.raises(T.LayoutError) as ei:
        T.scan_dataset(str(new_dir), layout=bad)
    msg = str(ei.value)
    assert "tile_regex" in msg
    for n in ("col_03_up/img_0007.png", "col_04_dn/img_0011.png", "col_05_up/img_0009.5.png"):
        assert n in msg


def test_wrong_group_regex_rejected(new_dir):
    bad = {**NEW_LAYOUT, "group_regex": r"^col_(?P<index>\d+)_(?P<direction>up)$"}
    rep = T.validate_layout(str(new_dir), bad)
    assert not rep["ok"]
    assert rep["containers"]["matched"] == 2 and len(rep["containers"]["unmatched"]) == 1
    assert rep["containers"]["unmatched"][0].startswith("col_04_dn")
    with pytest.raises(T.LayoutError) as ei:
        T.scan_dataset(str(new_dir), layout=bad)
    assert "col_04_dn" in str(ei.value) and "group_regex" in str(ei.value)


def test_bad_regex_missing_group_missing_field(new_dir):
    rep = T.validate_layout(str(new_dir), {**NEW_LAYOUT, "tile_regex": "^img_(?P<idx>["})
    assert not rep["ok"] and any("不是合法正则" in e for e in rep["errors"])
    rep = T.validate_layout(str(new_dir), {**NEW_LAYOUT,
                                           "group_regex": r"^col_(?P<n>\d+)_(?P<direction>up|dn)$"})
    assert not rep["ok"] and any("(?P<index>" in e for e in rep["errors"])
    with pytest.raises(T.LayoutError) as ei:
        T.scan_dataset(str(new_dir), layout={"group_regex": NEW_LAYOUT["group_regex"]})
    assert "tile_regex" in str(ei.value)


def test_uneven_column_count_is_a_warning_not_a_rejection(tmp_path):
    d = tmp_path / "u"
    d.mkdir()
    _new_scheme(d, as_zip=True)
    _zip(d / "col_06_dn.zip", ["col_06_dn/img_0007.png"])
    rep = T.validate_layout(str(d), NEW_LAYOUT)
    assert rep["ok"] and rep["count_outliers"] and "col_06_dn" in rep["count_outliers"][0]
    assert any("各列张数" in w for w in rep["warnings"])
    assert len(T.scan_dataset(str(d), layout=NEW_LAYOUT)) == 16


def test_no_direction_group_uses_parity_with_warning(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    _new_scheme(d, as_zip=True)
    lay = {**NEW_LAYOUT, "group_regex": r"^col_(?P<index>\d+)_"}
    rep = T.validate_layout(str(d), lay)
    assert rep["ok"] and any("direction" in w for w in rep["warnings"])
    assert [c["direction"] for c in rep["columns"]] == ["down", "up", "down"]


# ------------------------------------------------------------------ run_stitch.py
class _StopAfterScan(Exception):
    """Raised by the build_cache stub so main() ends right after step 1."""


class FakeClient:
    def __init__(self, available):
        self.available = available
        self.model, self.base, self.usage = "fake-kimi", "http://fake", "no calls"

    def chat(self, *a, **k):
        raise AssertionError("Kimi must not be called in tests")


class FakePool:
    calls: list = []
    prompts: list = []
    answer: dict | None = None
    answers: list | None = None      # a queue of answers, one per ask(); wins over `answer`

    def __init__(self, client, workers=1, votes=1):
        self.client = client

    def ask(self, role, prompt, images=None, voter=None):
        FakePool.calls.append(role.name)
        FakePool.prompts.append(prompt)
        if FakePool.answers is not None:
            if not FakePool.answers:
                raise AssertionError("planner asked more times than answers were queued")
            a = FakePool.answers.pop(0)
            return dict(a) if isinstance(a, dict) else a
        if FakePool.answer is None:
            raise AssertionError("planner asked when it must not be")
        return dict(FakePool.answer)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Wire run_stitch to the fakes and stop it right after the scan."""
    captured = {}
    # These 12x8 synthetic tiles have explicit test geometry. Production runs
    # must no longer silently apply the historical 3840x2160 calibration to them.
    profile_path = tmp_path / "fixture_stitch_profile.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1, "name": "layout-test-12x8", "image_size": [12, 8],
        "nominal_vectors": [0, 6, 9, 0], "geometry_source": "synthetic-layout-test",
        "scale_div": 1, "out_scale": 1,
    }), encoding="utf-8")

    def stop_after_scan(tiles, cache_dir, **kw):
        captured["tiles"] = tiles
        raise _StopAfterScan

    monkeypatch.setattr(RS.T, "build_cache", stop_after_scan)
    FakePool.calls, FakePool.prompts, FakePool.answer, FakePool.answers = [], [], None, None

    def wire(available, answer=None, answers=None):
        FakePool.answer = answer
        FakePool.answers = list(answers) if answers is not None else None
        monkeypatch.setattr(RS.KA, "KimiClient", lambda cache_dir=None, **k: FakeClient(available))
        monkeypatch.setattr(RS.KA, "AgentPool", FakePool)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_stitch.py", "--stitch-profile", str(profile_path), *argv])
        RS.main()

    return types.SimpleNamespace(wire=wire, run=run, captured=captured,
                                 work=tmp_path / "work")


def test_main_hardcoded_dataset_never_calls_planner(fig3_dir, harness):
    harness.wire(available=True, answer=None)
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(fig3_dir), "--work", str(harness.work))
    assert FakePool.calls == [] and not (harness.work / "layout.json").exists()
    assert [_fields(t) for t in harness.captured["tiles"]] == _expected_fig3(fig3_dir)


def test_main_unknown_dataset_asks_once_then_reuses_cache(new_dir, harness, monkeypatch):
    harness.wire(available=True, answer=NEW_LAYOUT)
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == ["grid_planner"]
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["layout"] == NEW_LAYOUT and rec["source"] == "kimi" and rec["model"] == "fake-kimi"
    assert rec["validation"]["ok"] and rec["validation"]["tiles"]["matched"] == 15
    assert rec["anomalies"] == NEW_LAYOUT["anomalies"]
    first = [_fields(t) for t in harness.captured["tiles"]]
    assert len(first) == 15 and first[0][:3] == (3, 0, "up")

    # second run: the cached layout is reused, the planner is never entered
    def boom(*a, **k):
        raise AssertionError("step_plan must not run when work/layout.json exists")
    monkeypatch.setattr(RS, "step_plan", boom)
    harness.wire(available=True, answer=None)
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == ["grid_planner"]
    assert [_fields(t) for t in harness.captured["tiles"]] == first


def test_main_no_key_no_cache_stops_and_names_layout_flag(new_dir, harness, capsys):
    harness.wire(available=False)
    with pytest.raises(SystemExit) as ei:
        harness.run("--data", str(new_dir), "--work", str(harness.work), "--no-ai")
    assert ei.value.code != 0
    out = capsys.readouterr().out
    assert "--layout" in out and "group_regex" in out
    assert not (harness.work / "layout.json").exists()


def test_main_manual_layout_flag_validated(new_dir, harness, tmp_path, capsys):
    f = tmp_path / "my_layout.json"
    f.write_text(json.dumps(NEW_LAYOUT), encoding="utf-8")
    harness.wire(available=False)
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(new_dir), "--work", str(harness.work), "--no-ai",
                    "--layout", str(f))
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["source"] == f"file:{f}" and rec["validation"]["ok"]
    assert len(harness.captured["tiles"]) == 15

    # a wrong manual layout is rejected: non-zero exit, names listed, nothing cached
    f.write_text(json.dumps({**NEW_LAYOUT, "tile_regex": r"^img_(?P<idx>\d{3})\.png$"}),
                 encoding="utf-8")
    work2 = tmp_path / "work2"
    with pytest.raises(SystemExit) as ei:
        harness.run("--data", str(new_dir), "--work", str(work2), "--no-ai", "--layout", str(f))
    assert ei.value.code != 0
    out = capsys.readouterr().out
    assert "col_03_up/img_0007.png" in out and "col_05_up/img_0009.5.png" in out
    assert not (work2 / "layout.json").exists()
    # Full-run preflight rejects a bad layout without creating any work files.
    # --plan still saves layout_rejected.json, covered separately below.
    assert not work2.exists()


def test_main_stale_cached_layout_is_rejected_not_replaced(new_dir, harness, capsys):
    harness.work.mkdir()
    (harness.work / "layout.json").write_text(json.dumps(
        {"layout": {**NEW_LAYOUT, "group_regex": r"^L(?P<index>\d+)$"}, "source": "kimi"}),
        encoding="utf-8")
    harness.wire(available=True, answer=NEW_LAYOUT)   # a fresh answer is available...
    with pytest.raises(SystemExit):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == []                        # ...but must not be fetched silently
    out = capsys.readouterr().out
    assert "layout.json" in out and "col_03_up" in out


def test_plan_flag_validates_and_saves(new_dir, harness):
    harness.wire(available=True, answer=NEW_LAYOUT)
    harness.run("--data", str(new_dir), "--work", str(harness.work), "--plan")
    assert FakePool.calls == ["grid_planner"]
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["validation"]["ok"] and rec["layout"]["tile_regex"] == NEW_LAYOUT["tile_regex"]

    harness.wire(available=True,
                 answer={**NEW_LAYOUT, "group_regex": r"^col_(?P<index>\d+)_(?P<direction>up)$"})
    work2 = harness.work.parent / "work_plan2"
    with pytest.raises(SystemExit):
        harness.run("--data", str(new_dir), "--work", str(work2), "--plan")
    assert (work2 / "layout_rejected.json").exists() and not (work2 / "layout.json").exists()


def test_auto_layout_flag_forces_planner_on_hardcoded_dataset(fig3_dir, harness):
    harness.wire(available=True, answer=FIG3_LAYOUT)
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(fig3_dir), "--work", str(harness.work), "--auto-layout")
    assert FakePool.calls == ["grid_planner"]
    assert [_fields(t) for t in harness.captured["tiles"]] == _expected_fig3(fig3_dir)


# ------------------------------------------------------------------ listing digest (experiment D, change 2)
def test_listing_digest_sorted_min_max_and_no_dot_inner_dir(tmp_path):
    """`如 A .. B` must be min..max, not filesystem order; plain folders must not report
    an inner directory `.` (the relpath quirk ds_D r1 puzzled over)."""
    d = tmp_path / "ds"
    (d / "col_00_down").mkdir(parents=True)
    # write in scrambled order so filesystem order != numeric order
    for k in (7, 12, 1, 3, 10, 2, 5, 11, 4, 9, 6, 8):
        (d / "col_00_down" / f"img_{k:04d}.png").write_bytes(PNG)
    text = RS.build_listing(str(d))
    assert "如 img_0001.png .. img_0012.png" in text
    assert "主流式样 img_####.png x12" in text
    assert "内层目录" not in text
    # a nested sub-folder IS reported as an inner directory
    (d / "col_01_up" / "sub").mkdir(parents=True)
    (d / "col_01_up" / "sub" / "img_0001.png").write_bytes(PNG)
    text = RS.build_listing(str(d))
    assert "col_01_up/" in text and "内层目录 sub" in text and "内层目录 ." not in text


def test_listing_digest_groups_odd_names_by_shape(tmp_path):
    """30 .png.xml sidecars must not push the one re-shoot out of the window (ds_G):
    odd names are grouped by shape with counts, and small groups list every name."""
    d = tmp_path / "ds"
    d.mkdir()
    names = [f"s04_bwd/{k:04d}.png" for k in range(1, 31) if k != 17] + ["s04_bwd/0017.5.png"]
    names += [f"s04_bwd/{k:04d}.png.xml" for k in range(1, 31) if k != 17]     # 29 sidecars
    _zip(d / "s04_bwd.zip", names)
    text = RS.build_listing(str(d))
    assert "主流式样 ####.png x29（如 0001.png .. 0030.png）" in text
    assert "####.png.xml x29（如 0001.png.xml .. 0030.png.xml）" in text
    assert "####.#.png x1: 0017.5.png" in text
    assert "不合群的文件名（按式样分组）" in text
    # one sidecar per tile (30 vs 30): the image shape still wins the "main" slot
    _zip(d / "s05_fwd.zip", [f"s05_fwd/{k:04d}.png" for k in range(1, 31)]
         + [f"s05_fwd/{k:04d}.png.xml" for k in range(1, 31)])
    text = RS.build_listing(str(d))
    assert "s05_fwd.zip  60 个文件  主流式样 ####.png x30" in text


def test_listing_collapses_identical_containers_and_lists_differing_ones(tmp_path):
    """45 zips (ds_E): the 42 regular columns collapse into one line, the two columns
    with re-shoots (beyond the old 40-entry window) are always listed on their own."""
    d = tmp_path / "ds"
    d.mkdir()
    for i in range(1, 46):
        direction = "down" if i % 2 == 1 else "up"
        cname = f"L{i:02d}_{direction}"
        keys = [f"{k:04d}" for k in range(1, 31)]
        if i == 43:
            keys = [k for k in keys if k != "0017"] + ["0017.5", "0017.6"]
        if i == 44:
            keys = keys + ["0009.5"]
        _zip(d / f"{cname}.zip", [f"{cname}/{k}.png" for k in sorted(keys)])
    (d / "README.txt").write_text("notes")
    text = RS.build_listing(str(d))
    lines = text.splitlines()
    assert len(lines) == 5, text                       # run + L43 + L44 + L45 + README
    assert lines[0].startswith("L01_down.zip .. L42_up.zip  （42 个压缩包，摘要相同）")
    assert "如 0001.png .. 0030.png" in lines[0]
    assert "内层目录式样 L##_down x21, L##_up x21" in lines[0]
    assert lines[1].startswith("L43_down.zip") and "0017.5.png, 0017.6.png" in lines[1]
    assert lines[2].startswith("L44_up.zip") and "0009.5.png" in lines[2]
    assert lines[3].startswith("L45_down.zip") and "内层目录 L45_down" in lines[3]
    assert lines[4] == "README.txt"
    assert "只列了前 40" not in text


def test_listing_fig3_miniature_shows_every_trap(fig3_dir):
    text = RS.build_listing(str(fig3_dir))
    assert "0003.5.png, 0003.6.png" in text              # the re-shoots are visible
    assert "内层目录 a_L21_up" in text                    # the inner folder named like the zip
    assert "Thumbs.db" in text and "notes.txt" in text


# ------------------------------------------------------------------ repair loop (experiment D, change 1)
STRICT = {**NEW_LAYOUT, "tile_regex": r"^img_(?P<idx>\d{4})\.png$"}      # misses img_0009.5.png
BAD_GROUP = {**NEW_LAYOUT, "group_regex": r"^col_(?P<index>\d+)_(?P<direction>up)$"}


def _unmatched_reshoot(scheme):
    return ("col_05_up.zip/col_05_up/img_0009.5.png" if scheme == "zip"
            else "col_05_up/img_0009.5.png")


def test_repair_prompt_carries_layout_report_and_names(new_dir, scheme):
    report = T.validate_layout(str(new_dir), STRICT)
    assert not report["ok"]
    base = RS.plan_prompt(str(new_dir), RS.build_listing(str(new_dir)))
    p = RS.repair_prompt(base, STRICT, report)
    assert p.startswith(base)
    # the previous layout is echoed as JSON (regex backslashes escaped the JSON way)
    assert json.dumps(STRICT["tile_regex"]) in p and "布局校验：不通过" in p
    assert _unmatched_reshoot(scheme) in p and "请修正" in p
    assert "请给出修正后的完整布局 JSON" in p
    # an unparsable previous answer gets the JSON-only reminder instead
    p2 = RS.repair_prompt(base, None, None)
    assert p2.startswith(base) and "请只输出 JSON" in p2


def test_repair_second_attempt_rescues_hidden_reshoot(new_dir, scheme, harness):
    """attempt 1 = strict regex (fails on img_0009.5.png); attempt 2 gets the validator's
    report fed back and answers with the tolerant regex -> validated, saved, ledger kept."""
    harness.wire(available=True, answers=[STRICT, NEW_LAYOUT])
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == ["grid_planner", "grid_planner"]
    # the second prompt is the repair prompt: listing + previous layout + report + names
    p1, p2 = FakePool.prompts
    assert p1 == RS.plan_prompt(str(new_dir), RS.build_listing(str(new_dir)))
    assert p2.startswith(p1)
    assert json.dumps(STRICT["tile_regex"]) in p2 and "布局校验：不通过" in p2
    assert _unmatched_reshoot(scheme) in p2 and "请修正" in p2
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["layout"] == NEW_LAYOUT and rec["validation"]["ok"]
    assert rec["source"] == "kimi+feedback" and rec["stopped_because"] == "ok"
    assert rec["attempts_used"] == 2 and rec["max_attempts"] == 2
    a1, a2 = rec["attempts"]
    assert a1["n"] == 1 and a1["ok"] is False and a1["layout"] == STRICT and a1["feedback"] is False
    assert a1["unmatched_images"] == [_unmatched_reshoot(scheme)] and a1["errors"]
    assert a2["n"] == 2 and a2["ok"] is True and a2["layout"] == NEW_LAYOUT and a2["feedback"] is True
    for a in (a1, a2):
        assert {"seconds", "prompt_tokens", "completion_tokens", "duplicate_idx"} <= set(a)
    assert len(harness.captured["tiles"]) == 15


def test_repair_stops_on_repeated_regex_not_on_max_attempts(new_dir, harness):
    """The ds_F signature: the model hands back the same (group_regex, tile_regex) pair,
    so the loop stops after attempt 2 even with --plan-attempts 3 (a third call would
    be wasted) and the ledger says why."""
    harness.wire(available=True, answers=[BAD_GROUP, dict(BAD_GROUP), NEW_LAYOUT])
    with pytest.raises(SystemExit) as ei:
        harness.run("--data", str(new_dir), "--work", str(harness.work), "--plan-attempts", "3")
    assert ei.value.code != 0
    assert FakePool.calls == ["grid_planner", "grid_planner"]      # never the third
    assert not (harness.work / "layout.json").exists()
    rej = json.loads((harness.work / "layout_rejected.json").read_text(encoding="utf-8"))
    assert rej["stopped_because"] == "repeated_regex" and rej["attempts_used"] == 2
    assert rej["source"] == "kimi+feedback" and rej["max_attempts"] == 3
    assert rej["validation"]["ok"] is False and rej["layout"] == BAD_GROUP
    assert [a["ok"] for a in rej["attempts"]] == [False, False]


def test_repair_max_attempts_and_cli_flag(new_dir, harness):
    # default cap 2: two different failing layouts -> max_attempts, second is a feedback call
    other_bad = {**NEW_LAYOUT, "tile_regex": r"^img_(?P<idx>\d{3})\.png$"}
    harness.wire(available=True, answers=[STRICT, other_bad, NEW_LAYOUT])
    with pytest.raises(SystemExit):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == ["grid_planner", "grid_planner"]
    rej = json.loads((harness.work / "layout_rejected.json").read_text(encoding="utf-8"))
    assert rej["stopped_because"] == "max_attempts" and rej["attempts_used"] == 2
    assert [a["layout"]["tile_regex"] for a in rej["attempts"]] == \
        [STRICT["tile_regex"], other_bad["tile_regex"]]
    # --plan-attempts 1: the old one-shot behaviour, no feedback call at all
    work2 = harness.work.parent / "work_one"
    FakePool.calls, FakePool.prompts = [], []
    harness.wire(available=True, answers=[STRICT, NEW_LAYOUT])
    with pytest.raises(SystemExit):
        harness.run("--data", str(new_dir), "--work", str(work2), "--plan-attempts", "1")
    assert FakePool.calls == ["grid_planner"]
    rej = json.loads((work2 / "layout_rejected.json").read_text(encoding="utf-8"))
    assert rej["stopped_because"] == "max_attempts" and rej["attempts_used"] == 1
    assert rej["source"] == "kimi"


def test_repair_unparsable_first_answer_then_ok(new_dir, harness):
    harness.wire(available=True, answers=[{"_unparsed": "sorry, no json"}, NEW_LAYOUT])
    with pytest.raises(_StopAfterScan):
        harness.run("--data", str(new_dir), "--work", str(harness.work))
    assert FakePool.calls == ["grid_planner", "grid_planner"]
    assert "请只输出 JSON" in FakePool.prompts[1]
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["attempts"][0]["ok"] is False and rec["attempts"][0]["layout"] is None
    assert rec["attempts"][1]["ok"] is True and rec["source"] == "kimi+feedback"


def test_plan_flag_uses_repair_loop_and_records_ledger(new_dir, harness):
    harness.wire(available=True, answers=[STRICT, NEW_LAYOUT])
    harness.run("--data", str(new_dir), "--work", str(harness.work), "--plan")
    assert FakePool.calls == ["grid_planner", "grid_planner"]
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["validation"]["ok"] and rec["attempts_used"] == 2 and rec["stopped_because"] == "ok"
    # one-shot success keeps source "kimi" and a one-entry ledger
    work2 = harness.work.parent / "work_plan_one"
    FakePool.calls, FakePool.prompts = [], []
    harness.wire(available=True, answers=[NEW_LAYOUT])
    harness.run("--data", str(new_dir), "--work", str(work2), "--plan")
    assert FakePool.calls == ["grid_planner"]
    rec = json.loads((work2 / "layout.json").read_text(encoding="utf-8"))
    assert rec["source"] == "kimi" and rec["attempts_used"] == 1 and len(rec["attempts"]) == 1


def test_plan_flag_no_key_prints_listing_and_saves_nothing(new_dir, harness, capsys):
    harness.wire(available=False)
    harness.run("--data", str(new_dir), "--work", str(harness.work), "--plan", "--no-ai")
    out = capsys.readouterr().out
    assert "col_03_up" in out and "没有可校验的布局" in out
    assert FakePool.calls == [] and not (harness.work / "layout.json").exists()


# ------------------------------------------------------------------ first container reverse (experiment D, change 3)
def test_first_container_reverse_warning_lands_in_layout_json(new_dir, harness):
    """new_dir starts with col_03_up = the reverse token: filenames cannot decide whether
    that is right, so the validator warns (never errors) and the note reaches layout.json."""
    harness.wire(available=True, answer=NEW_LAYOUT)
    harness.run("--data", str(new_dir), "--work", str(harness.work), "--plan")
    rec = json.loads((harness.work / "layout.json").read_text(encoding="utf-8"))
    assert rec["validation"]["ok"]
    ws = rec["validation"]["warnings"]
    assert any("第一个容器（index 最小）被判为反向" in w and "reverse_direction_token 可能取反" in w
               for w in ws), ws


def test_first_container_forward_gives_no_reverse_warning(fig3_dir):
    rep = T.validate_layout(str(fig3_dir), FIG3_LAYOUT)
    assert rep["ok"] and not any("被判为反向" in w for w in rep["warnings"])
    # parity fallback always starts forward, so it never triggers the note either
    rep = T.validate_layout(str(fig3_dir), {**FIG3_LAYOUT, "group_regex": r"^a_L(?P<index>\d+)_"})
    assert rep["ok"] and not any("被判为反向" in w for w in rep["warnings"])
