"""Colour-balance check against a recorded reference substrate colour; synthetic pixels only."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakepipeline import color_diagnostics as cd  # noqa: E402

REFERENCE = {"schema_version": 1, "kind": "reference_substrate_colour",
             "mean_rgb": [219.2, 170.9, 170.0], "std_rgb": [9.5, 8.3, 8.1], "tolerance_fraction": 0.05}
SUBSTRATE = np.array([219, 171, 170], np.float32)
MONOLAYER = np.array([196, 129, 171], np.float32)


def scene(height=240, width=320, gradient=8.0, seed=0):
    """Bright substrate with a vertical illumination gradient and darker crystal patches
    covering about 30 % of the frame; a few bright 'nuclei' pixels that are never flat."""
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), np.float32)
    image[:] = SUBSTRATE
    rows = np.linspace(-gradient, gradient, height, dtype=np.float32)[:, None, None]
    image += rows
    for _ in range(12):
        y, x = rng.integers(0, height - 40), rng.integers(0, width - 40)
        image[y:y + 40, x:x + 40] = MONOLAYER + rows[y:y + 40]
    for _ in range(30):
        y, x = rng.integers(0, height), rng.integers(0, width)
        image[y, x] = (60, 90, 230)
    image += rng.normal(0, 0.8, image.shape).astype(np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def test_substrate_is_the_brightest_flat_plateau_despite_gradient():
    result = cd.background_colour(scene())
    assert result["method"] == "brightest_flat_luminance_plateau"
    assert np.allclose(result["median_rgb"], SUBSTRATE, atol=3)
    assert result["flat_fraction"] > 0.8
    assert result["selected_plateau"]["share_of_flat"] > 0.5
    assert len(result["plateaus"]) == 2          # substrate and monolayer, not a bin per gradient step


def test_matching_colour_balance_is_within_tolerance():
    check = cd.colour_check(scene(), REFERENCE)
    assert check["verdict"] == "within_tolerance"
    assert all(abs(g - 1) < 0.02 for g in check["gain_rgb"])


@pytest.mark.parametrize("gain", [(0.977, 0.892, 1.068), (1.05, 0.94, 1.0)])
def test_shifted_colour_balance_is_reported_with_the_inverse_gain(gain):
    shifted = cd.apply_channel_gain(scene(), gain)
    check = cd.colour_check(shifted, REFERENCE)
    assert check["verdict"] == "colour_balance_differs_from_reference"
    assert np.allclose(check["gain_rgb"], 1 / np.array(gain), atol=0.02)
    shift = np.array(check["channel_shift_rgb"])
    assert np.all(np.sign(shift) == np.sign(np.array(gain) - 1))   # bluer when B gain > 1, etc.


def test_pixel_count_change_alone_does_not_change_the_verdict():
    full = scene(height=480, width=640)
    small = np.asarray(Image.fromarray(full).resize((320, 240), Image.BOX))
    a, b = cd.colour_check(full, REFERENCE), cd.colour_check(small, REFERENCE)
    assert a["verdict"] == b["verdict"] == "within_tolerance"
    assert np.allclose(a["gain_rgb"], b["gain_rgb"], atol=0.01)


def test_noise_without_flat_area_falls_back_to_global_median():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    result = cd.background_colour(noise)
    assert result["method"] == "global_median_insufficient_flat_area"
    assert result["selected_plateau"] is None


def test_uniform_far_colour_is_an_implausible_gain_not_a_correction():
    check = cd.colour_check(np.full((24, 32, 3), (130, 70, 120), np.uint8), REFERENCE)
    assert check["verdict"] == "implausible_gain_check_inputs"
    with pytest.raises(ValueError, match="implausible"):
        cd.apply_channel_gain(np.zeros((4, 4, 3), np.uint8), check["gain_rgb"])


def test_black_background_is_unusable():
    check = cd.colour_check(np.zeros((40, 40, 3), np.uint8), REFERENCE)
    assert check["verdict"] == "unusable_black_background"
    assert check["gain_rgb"] is None


def test_no_reference_is_explicit():
    check = cd.colour_check(scene(), None)
    assert check["verdict"] == check["status"] == "no_reference"
    assert cd.summarize([check])["verdict"] == "no_reference"


def test_apply_gain_clips_and_never_edits_the_input():
    image = scene()
    before = image.copy()
    out = cd.apply_channel_gain(image, (1.2, 1.2, 1.2))
    assert out.dtype == np.uint8 and out.max() == 255
    assert np.array_equal(image, before)


def test_reference_loading_validates_schema(tmp_path):
    path = tmp_path / "ref.json"
    assert cd.load_reference(path) is None
    path.write_text(json.dumps(REFERENCE), encoding="utf-8")
    loaded = cd.load_reference(path)
    assert loaded["tolerance_fraction"] == 0.05
    for broken in [dict(REFERENCE, kind="other"), dict(REFERENCE, mean_rgb=[1, 2]),
                   dict(REFERENCE, mean_rgb=[300, 1, 1]), dict(REFERENCE, tolerance_fraction=1.5),
                   dict(REFERENCE, schema_version=2)]:
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            cd.load_reference(path)


def test_shipped_reference_record_is_valid_and_documented():
    reference = cd.load_reference(ROOT / "configs/reference_substrate_colour.json")
    assert reference["kind"] == "reference_substrate_colour"
    assert len(reference["source_frames"]) == 2
    assert all(len(frame["member_sha256"]) == 64 for frame in reference["source_frames"])
    assert "not an independent reflectance calibration" in reference["description"]
    assert any("diagnostic probe" in limit for limit in reference["limits"])


def test_summary_reports_the_worst_verdict():
    checks = [cd.colour_check(scene(), REFERENCE),
              cd.colour_check(cd.apply_channel_gain(scene(), (0.9, 0.9, 1.1)), REFERENCE)]
    summary = cd.summarize(checks)
    assert summary["verdict"] == "colour_balance_differs_from_reference"
    assert len(summary["gains_rgb"]) == 2
    assert "not a scale problem" in summary["meaning"]
