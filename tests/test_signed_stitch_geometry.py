"""Signed image geometry, preserving scanner IDs; synthetic/offline fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

import stitch_profile as SP
from tools.stitch_setup import service


ROOT = Path(__file__).resolve().parents[1]


def profile(vectors=(0, 140, -250, -4)):
    return {"schema_version": 1, "name": "synthetic-leftward",
            "image_size": [320, 180], "nominal_vectors": list(vectors),
            "geometry_source": "measured: synthetic overlap geometry"}


def test_leftward_profile_keeps_signed_vectors_and_historical_profile_unchanged():
    assert SP.validate_profile(profile())["nominal_vectors"] == [0.0, 140.0, -250.0, -4.0]
    assert SP.validate_profile(SP.historical_profile()) == SP.historical_profile()
    # A diagonal/sheared but valid leftward scan retains both signed components.
    p = profile((6, 140, -250, -4))
    assert SP.validate_profile(p)["nominal_vectors"] == list(map(float, p["nominal_vectors"]))


@pytest.mark.parametrize("vectors", [(0, 140, -320, 0), (0, 140, 0, 0),
                                     (0, -140, -250, 0), (100, 100, -100, -100),
                                     (255, 1, 1, 191)])
def test_signed_profiles_still_reject_no_overlap_or_degenerate_geometry(vectors):
    with pytest.raises(SP.ProfileError):
        SP.validate_profile(profile(vectors))


def test_leftward_profile_runs_through_actual_preparation_cli_and_default_colour(tmp_path, monkeypatch):
    """Twelve real PNG crops, full CLI, no model/AI; IDs are never mirrored."""
    raw, output = tmp_path / "raw", tmp_path / "outputs"
    raw.mkdir()
    rng = np.random.default_rng(101)
    # Correlated texture is shared by the actual neighbouring overlaps.
    grey = cv2.GaussianBlur(rng.normal(0, 26, (620, 850)).astype(np.float32), (0, 0), 0.7)
    canvas = np.array([130, 150, 175], np.float32)[None, None, :] + grey[:, :, None]
    truth = {}
    for col in range(3):
        for row in range(4):
            x, y = (2 - col) * 250 + 10, row * 140 + (2 - col) * 4 + 10
            image = np.clip(np.rint(canvas[y:y + 180, x:x + 320]), 0, 255).astype(np.uint8)
            name = f"mosaic_r{row}_c{col}.png"
            assert cv2.imwrite(str(raw / name), image)
            truth[name] = [x, y]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    monkeypatch.setattr(service, "kimi_status", lambda: {"mode": "no-ai", "configured": False})
    inspection = service.dispatch("inspect", {"data_dir": str(raw)})["inspection"]
    prepared = service.dispatch("prepare", {
        "data_dir": str(raw), "fingerprint": inspection["fingerprint"], "profile_kind": "custom",
        "calibration_note": "Synthetic known image overlaps: increasing columns move leftwards.",
        "nominal_vectors": [0, 140, -250, -4], "scale_div": 1, "out_scale": 1,
        "full": True, "output_parent": str(output), "ai_mode": "off"})
    argv = json.loads(Path(prepared["argv_path"]).read_text())
    assert "--no-ai" in argv and argv[argv.index("--colour-correction") + 1] == "auto"
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=90,
                            env=dict(os.environ, PYTHONUTF8="1"))
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    bundle = Path(prepared["bundle_dir"])
    state = json.loads((bundle / "work/state.json").read_text())
    stats = state["register"]["stats"]
    assert stats["image_column_direction"] == "left"
    assert stats["image_column_direction_source"] == "registered_overlap_pairs"
    assert stats["measured_dx"] < 0 and stats["measured_overlap_x_px"] > 0
    assert state["summary"]["tiles_used"] == 12
    assert state["colour_correction"]["render_verified"] is True
    assert state["colour_correction"]["status"] == "applied_at_render"
    assert set(state["render"]["rendered_tids"]) == set(stats["row_abs"])
    tids = json.loads((bundle / "work/positions_tids.json").read_text())
    pos = np.load(bundle / "work/positions.npy")
    # Tile IDs preserve filename/column identity; solved positions retain the
    # independently known leftward image ordering, without file reindexing.
    names = [tid.split("/")[-1] for tid in tids]
    expected = np.array([truth[name] for name in names])
    delta = pos - expected
    assert np.linalg.norm(delta - np.median(delta, axis=0), axis=1).max() < 1.0
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()} == before
