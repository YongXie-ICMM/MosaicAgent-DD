"""Scale-adapted inference preserves source coordinates; no trained model is used."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

FP = Path(__file__).resolve().parents[1] / "flakepipeline"
sys.path.insert(0, str(FP))
import inference_geometry as geometry
import seg_model
import stages


def plan(native=(1920, 1080), reference=(3840, 2160), confirmed=True):
    return geometry.plan_inference_geometry(
        native, reference, same_physical_fov_confirmed=confirmed,
        provenance={"scope": "synthetic software check; no physical calibration"})


@pytest.fixture
def model(monkeypatch):
    class Recorder:
        def __init__(self):
            self.inputs = []
            self.loads = 0

        def __call__(self, x):
            self.inputs.append(x.detach().clone())
            logits = torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), device=x.device)
            logits[:, 2] = 5
            return {"out": logits}

    instance = Recorder()

    def load(*args, **kwargs):
        instance.loads += 1
        return instance, {"fixture": "constant class 2; not a trained checkpoint"}

    monkeypatch.setattr(seg_model, "load_segmenter", load)
    return instance


def run(tmp_path, image, *, inference_geometry=None, **updates):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "input.png"
    Image.fromarray(image).save(path)
    ctx = {"work": tmp_path, "sample": "test", "weights": "unused.pth",
           "tile": 512, "overlap": 64, "threads": 1, "batch": 3,
           "filler": None, "border_trim_px": 0}
    if inference_geometry is not None:
        ctx["inference_geometry"] = inference_geometry
    ctx.update(updates)
    result = stages.stage_segment(ctx, path)
    return result, np.load(result.data["mask"])


def test_adapted_tiles_reach_model_size_and_return_only_native_pixels(tmp_path, model):
    image = np.full((301, 517, 3), [40, 100, 180], dtype=np.uint8)
    result, mask = run(tmp_path, image, inference_geometry=plan())
    assert model.inputs and all(tuple(x.shape[2:]) == (512, 512) for x in model.inputs)
    assert mask.shape == image.shape[:2] and np.all(mask == 2)
    assert sum(result.data["raw_counts"].values()) == 301 * 517
    assert result.data["raw_counts"]["1L"] == 301 * 517
    assert result.data["native_image_pixels"] == 301 * 517
    assert result.data["uncovered_native_pixels"] == 0
    assert result.data["source_crop_tile"] == 256
    assert result.data["source_crop_overlap"] == 32
    assert result.data["requested_model_tile"] == 512
    assert result.data["requested_model_overlap"] == 64
    assert result.data["resample_mode"]["input"] == "bilinear"
    assert result.data["resample_mode"]["labels"] == "nearest"
    assert result.data["counts_coordinate_system"] == "native_image_pixels"
    assert result.data["inference_geometry"]["physical_calibration_established"] is False


def test_partial_native_tile_is_padded_without_stretching_valid_source(tmp_path, model):
    image = np.full((37, 61, 3), 128, dtype=np.uint8)
    result, mask = run(tmp_path, image, inference_geometry=plan())
    assert mask.shape == (37, 61) and mask.size == 37 * 61
    x = model.inputs[0]
    assert tuple(x.shape) == (1, 3, 512, 512)
    # The actual photo occupies only its scaled extent. Its bottom-right padding
    # remains raw black, rather than stretching a small photograph to the tile.
    expected_photo = torch.from_numpy((np.full(3, 128, np.float32) / 255 - stages.MEAN) / stages.STD)
    expected_black = torch.from_numpy(-stages.MEAN / stages.STD)
    torch.testing.assert_close(x[0, :, 10, 10], expected_photo)
    torch.testing.assert_close(x[0, :, -1, -1], expected_black)
    assert result.data["valid_native_pixels"] == image.shape[0] * image.shape[1]


def test_nearest_label_restore_preserves_class_ids_and_native_boundary(tmp_path, monkeypatch):
    def striped_model(x):
        logits = torch.zeros((x.shape[0], 4, x.shape[2], x.shape[3]))
        logits[:, 1, :, :256] = 8
        logits[:, 3, :, 256:] = 8
        return {"out": logits}
    monkeypatch.setattr(seg_model, "load_segmenter", lambda *args, **kwargs: (striped_model, {}))
    result, mask = run(tmp_path, np.full((256, 256, 3), 120, np.uint8), inference_geometry=plan())
    assert np.all(mask[:, :128] == 1) and np.all(mask[:, 128:] == 3)
    assert set(np.unique(mask)) == {1, 3}
    assert result.data["raw_counts"]["2L"] == 128 * 256
    assert result.data["raw_counts"]["TL"] == 128 * 256


def test_counts_still_exclude_native_filler_support(tmp_path, model):
    image = np.full((269, 283, 3), 120, dtype=np.uint8)
    image[:7] = 255
    result, mask = run(tmp_path, image, inference_geometry=plan(), filler="white")
    assert np.all(mask[:7] == 0) and np.all(mask[7:] == 2)
    assert sum(result.data["raw_counts"].values()) == (269 - 7) * 283
    assert result.data["valid_native_pixels"] == (269 - 7) * 283
    assert result.data["uncovered_native_pixels"] == 0


def test_default_mode_never_interpolates_and_keeps_existing_padding(tmp_path, model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unadapted inference must not interpolate")
    monkeypatch.setattr(torch.nn.functional, "interpolate", forbidden)
    image = np.full((57, 79, 3), [40, 100, 180], dtype=np.uint8)
    result, mask = run(tmp_path, image)
    x = model.inputs[0]
    assert tuple(x.shape) == (1, 3, 64, 96)
    assert torch.count_nonzero(x[:, :, 57:]) == 0
    assert torch.count_nonzero(x[:, :, :, 79:]) == 0
    expected = torch.from_numpy((image[0, 0].astype(np.float32) / 255 - stages.MEAN) / stages.STD)
    torch.testing.assert_close(x[0, :, 0, 0], expected, rtol=0, atol=0)
    assert mask.shape == (57, 79) and np.all(mask == 2)
    assert result.data["inference_geometry"] is None
    assert result.data["resample_mode"]["input"] == "none"


def test_unit_scale_plan_matches_default_without_resizing(tmp_path, model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit scale must retain the existing path")
    monkeypatch.setattr(torch.nn.functional, "interpolate", forbidden)
    image = np.full((541, 577, 3), 127, dtype=np.uint8)
    _, plain = run(tmp_path / "plain", image)
    _, adapted = run(tmp_path / "unit", image,
                     inference_geometry=plan((3840, 2160), (3840, 2160)))
    assert np.array_equal(plain, adapted)


@pytest.mark.parametrize("change", ["unconfirmed", "native_tile", "scale", "status", "permitted", "ctx_tile", "ctx_overlap"])
def test_blocked_or_inconsistent_geometry_never_loads_weights(tmp_path, model, change):
    value = deepcopy(plan())
    updates = {}
    if change == "unconfirmed":
        value = plan(confirmed=False)
    elif change == "native_tile":
        value["native_tile"] = 255
    elif change == "scale":
        value["model_resize_scale"] = 1.0
    elif change == "status":
        value["status"] = "needs_calibration"
    elif change == "permitted":
        value["adapted_mode_allowed"] = False
    elif change == "ctx_tile":
        updates["tile"] = 256
    else:
        updates["overlap"] = 32
    with pytest.raises(ValueError, match="geometry|calibration"):
        run(tmp_path, np.full((40, 50, 3), 120, np.uint8), inference_geometry=value, **updates)
    assert model.loads == 0
    assert not (tmp_path / "02_segment").exists()
