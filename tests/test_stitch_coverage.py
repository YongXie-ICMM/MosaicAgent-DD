"""Synthetic coverage-policy regressions; no raw acquisition or model calls."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stitch as RS  # noqa: E402
import tiles as T  # noqa: E402


def tile(row, col):
    return T.Tile(col=col, col_idx=col, direction="down", order=row, key=float(row),
                  name=f"mosaic_r{row}_c{col}.png", zip_path="", inner="",
                  nominal_row=row)


@pytest.fixture(autouse=True)
def synthetic_geometry(monkeypatch):
    monkeypatch.setattr(T, "TILE_W", 320)
    monkeypatch.setattr(T, "TILE_H", 180)


def coverage_case(name):
    if name == "unique-centre":
        centre = tile(1, 1)
        tiles = [centre, tile(1, 0), tile(1, 2), tile(0, 1), tile(2, 1)]
        positions = np.array([(0, 0), (-256, 0), (256, 0), (0, -144), (0, 144)], float)
        # Exact geometry: 64% covered, but the central 36% has no other source.
        return tiles, positions, {centre.tid}
    tiles = [tile(row, col) for col in range(2) for row in range(4)]
    positions = np.array([(col * 270, row * 140) for col in range(2) for row in range(4)], float)
    drop = {tiles[5].tid, tiles[6].tid}
    if name == "shifted-overlap":
        positions[5] = positions[4]
        positions[6] = positions[7]
    return tiles, positions, drop


@pytest.mark.parametrize("case,legacy_rescues", [
    ("adjacent-drops", True),
    ("unique-centre", False),
    ("shifted-overlap", False),
])
@pytest.mark.parametrize("policy", ["legacy", "preserve-grid"])
def test_grid_policy_preserves_selected_tiles_without_changing_legacy(case, legacy_rescues,
                                                                    policy, tmp_path, capsys):
    tiles, positions, drop = coverage_case(case)
    state = RS.State(tmp_path / "state.json")
    rescued = RS.step_rescue(tiles, drop, positions, state, policy=policy)
    expected = drop if policy == "preserve-grid" or legacy_rescues else set()
    assert rescued == expected
    persisted = json.loads(state.path.read_text())
    assert persisted["coverage_policy"] == policy
    assert persisted["rescue"] == sorted(expected)
    assert "rescue" in persisted["_done"]
    if policy == "preserve-grid":
        log = capsys.readouterr().out
        assert "coverage-policy rescue (preserve-grid)" in log
        for tid in drop:
            assert tid in log


def test_default_policy_keeps_legacy_coverage_threshold(tmp_path):
    tiles, positions, drop = coverage_case("unique-centre")
    state = RS.State(tmp_path / "state.json")
    assert RS.step_rescue(tiles, drop, positions, state) == set()
    assert state.d["coverage_policy"] == "legacy"


def test_preserve_grid_rescues_only_selected_tiles_and_keeps_qc_reasons(tmp_path):
    tiles, positions, selected_drop = coverage_case("shifted-overlap")
    omitted = "not-selected.png"
    drop = selected_drop | {omitted}
    qc = {"drop": sorted(drop), "closed_form": sorted(selected_drop),
          "verdicts": {tid: {"category": "defocus", "keep": False,
                             "reason": "Synthetic low focus score", "_ai": False}
                       for tid in drop}}
    state = RS.State(tmp_path / "state.json")
    state.d["qc"] = copy.deepcopy(qc)
    # The policy depends on selected grid identities, not finite or distinct coordinates.
    positions[:] = np.nan
    assert RS.step_rescue(tiles, drop, positions, state, policy="preserve-grid") == selected_drop
    assert state.d["qc"] == qc
    assert json.loads(state.path.read_text())["qc"] == qc


@pytest.mark.parametrize("policy", ["legacy", "preserve-grid"])
def test_no_drop_clears_previous_rescues_and_records_policy(policy, tmp_path):
    state = RS.State(tmp_path / "state.json")
    state.d["rescue"] = ["old-rescue.png"]
    assert RS.step_rescue([], set(), np.empty((0, 2)), state, policy=policy) == set()
    persisted = json.loads(state.path.read_text())
    assert persisted["rescue"] == []
    assert persisted["coverage_policy"] == policy
    assert "rescue" in persisted["_done"]


def test_unknown_coverage_policy_is_rejected_before_state_changes(tmp_path):
    state = RS.State(tmp_path / "state.json")
    with pytest.raises(ValueError, match="Unknown coverage policy"):
        RS.step_rescue([], set(), np.empty((0, 2)), state, policy="unknown")
    assert state.d == {}
    assert not state.path.exists()
