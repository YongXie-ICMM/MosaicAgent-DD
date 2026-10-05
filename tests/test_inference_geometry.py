"""Geometry-contract checks; no model, image resizing or hardware calls."""
import json
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakepipeline.inference_geometry import plan_inference_geometry, geometry_from_layer_contract


def plan(native=(1920, 1080), reference=(3840, 2160), **kwargs):
    return plan_inference_geometry(native, reference, **kwargs)


def test_confirmed_half_resolution_preserves_model_window_and_scales_pixel_thresholds():
    result = plan(same_physical_fov_confirmed=True,
                  provenance={"scope": "provisional_same_fov_hypothesis_for_comparison",
                              "reference_size_source": "previous inference originals"})
    assert result["status"] == "ready" and result["adapted_mode_allowed"] is True
    assert (result["native_tile"], result["native_overlap"]) == (256, 32)
    assert (result["model_tile"], result["model_overlap"]) == (512, 64)
    assert result["model_resize_scale"] == 2.0
    assert result["native_linear_threshold_factor"] == 0.5
    assert result["native_area_threshold_factor"] == 0.25
    assert result["physical_stage_motion"] == "unchanged"
    assert result["network_architecture_changed"] is False
    assert result["physical_calibration_established"] is False
    assert result["segmentation_accuracy_established"] is False
    assert result["reference_kind"] == "previous_inference_capture_mode_not_training_crop"
    assert result["provenance"]["scope"] == "provisional_same_fov_hypothesis_for_comparison"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("native", [(1920, 1080), (3840, 2160)])
def test_even_equal_resolution_does_not_establish_same_physical_fov(native):
    result = plan(native)
    assert result["status"] == "needs_calibration"
    assert result["adapted_mode_allowed"] is False
    assert result["reasons"] == ["same_physical_field_of_view_not_confirmed"]
    for field in ("native_tile", "native_overlap", "model_resize_scale",
                  "native_linear_threshold_factor", "native_area_threshold_factor"):
        assert result[field] is None


def test_confirmed_equal_resolution_uses_identity_mapping_without_accuracy_claim():
    result = plan((3840, 2160), same_physical_fov_confirmed=True)
    assert result["native_tile"] == 512 and result["native_overlap"] == 64
    assert result["model_resize_scale"] == 1.0
    assert result["native_area_threshold_factor"] == 1.0
    assert result["physical_pixel_size_um"] is None


def test_anisotropy_cannot_silently_distort_the_image():
    result = plan((1920, 1200), same_physical_fov_confirmed=True)
    assert result["status"] == "needs_calibration"
    assert result["reasons"] == ["anisotropic_dimension_ratio_requires_calibration"]
    assert result["native_tile"] is None


@pytest.mark.parametrize("native,reference", [((0, 1080), (3840, 2160)), ((1920, -1080), (3840, 2160)),
                                            ((1920, 1080), (0, 2160))])
def test_nonpositive_dimensions_block_planning_without_division(native, reference):
    result = plan(native, reference, same_physical_fov_confirmed=True)
    assert result["status"] == "needs_calibration"
    assert result["reasons"] == ["nonpositive_image_dimensions"]


@pytest.mark.parametrize("native", [(True, 1080), (1920.0, 1080), [1920], "1920x1080", {"width": 1920, "height": 1080}])
def test_dimensions_require_integer_width_height(native):
    with pytest.raises(ValueError):
        plan(native, same_physical_fov_confirmed=True)


@pytest.mark.parametrize("confirmed", [1, "true", None])
def test_fov_confirmation_is_not_coerced_from_untrusted_input(confirmed):
    with pytest.raises(ValueError, match="must be a boolean"):
        plan(same_physical_fov_confirmed=confirmed)


def test_fractional_native_windows_are_not_silently_rounded():
    result = plan((1280, 720), same_physical_fov_confirmed=True)
    assert result["status"] == "needs_calibration"
    assert result["reasons"] == ["nonintegral_native_tile_or_overlap"]
    assert result["native_tile"] is None and result["native_overlap"] is None
    # Tile integral but overlap fractional still must fail.
    result = plan(same_physical_fov_confirmed=True, model_overlap=63)
    assert result["status"] == "needs_calibration"


@pytest.mark.parametrize("kwargs", [{"model_tile": True}, {"model_tile": 512.0}, {"model_tile": 0},
                                   {"model_overlap": False}, {"model_overlap": 64.0},
                                   {"model_overlap": -1}, {"model_overlap": 512}])
def test_model_window_configuration_is_strict(kwargs):
    with pytest.raises(ValueError):
        plan(same_physical_fov_confirmed=True, **kwargs)


def test_provenance_is_preserved_but_cannot_override_computed_fields():
    source = {"note": "Comparison only; physical field of view not measured."}
    result = plan(same_physical_fov_confirmed=True, provenance=source)
    source["note"] = "changed caller object"
    assert result["provenance"]["note"].startswith("Comparison only")
    for bad in ({"status": "ready"}, {"scope": True}, {"note": ""}, "trusted"):
        with pytest.raises(ValueError):
            plan(provenance=bad)


def write_layer_contract(folder, out_scale=1):
    # Use the actual stitch-setup producer, not an independently invented schema.
    from tools.stitch_setup.service import _layer_input_contract
    profile = {"schema_version": 1, "name": "synthetic-1080p",
               "image_size": [1920, 1080], "nominal_vectors": [0, 1000, 1800, 0],
               "geometry_source": "synthetic-test", "scale_div": 1, "out_scale": out_scale}
    inventory = {"image_size": [1920, 1080], "fingerprint": "a" * 64, "tile_count": 4}
    inspection = {"image_size": [1920, 1080], "fingerprint": "b" * 64,
                  "layout_mode": "flat-grid", "acquisition": {"present": False}}
    resolved = {"profile": profile, "scale_div": 1, "out_scale": out_scale, "full": True}
    preflight = {"schema_version": 1, "inspection": inspection,
                 "input_inventory": inventory, "resolved": resolved}
    preflight_text = json.dumps(preflight, indent=2) + "\n"
    profile_text = json.dumps(profile, indent=2) + "\n"
    contract = _layer_input_contract(inspection, inventory, resolved, preflight_text, profile_text)
    (folder / "preflight.json").write_text(preflight_text)
    (folder / "profile.json").write_text(profile_text)
    path = folder / "layer_input_contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n")
    return path


def test_layer_handoff_defaults_to_unconfirmed_fov_and_binds_exact_contract_bytes(tmp_path):
    path = write_layer_contract(tmp_path)
    result = geometry_from_layer_contract(path, (3840, 2160))
    assert result["status"] == "needs_calibration"
    assert result["observed_native_size"] == [1920, 1080]
    assert result["provenance"]["observed_size_source"] == str(path.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() in result["provenance"]["note"]
    assert "not whole-mosaic dimensions" in result["provenance"]["note"]


def test_confirmed_full_resolution_handoff_produces_an_exact_stage_compatible_plan(tmp_path):
    path = write_layer_contract(tmp_path)
    result = geometry_from_layer_contract(path, (3840, 2160), same_physical_fov_confirmed=True)
    assert result["status"] == "ready" and result["native_tile"] == 256
    expected = plan_inference_geometry([1920, 1080], [3840, 2160],
        same_physical_fov_confirmed=True, provenance=result["provenance"])
    assert result == expected
    assert result["physical_calibration_established"] is False


@pytest.mark.parametrize("sibling", ["profile.json", "preflight.json"])
def test_changed_bound_sibling_rejected_before_any_inference(tmp_path, sibling):
    path = write_layer_contract(tmp_path)
    file = tmp_path / sibling
    file.write_bytes(file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        geometry_from_layer_contract(path, (3840, 2160), True)


def test_contract_native_dimensions_must_agree_with_bound_profile(tmp_path):
    path = write_layer_contract(tmp_path)
    value = json.loads(path.read_text())
    value["native_image_size"] = [3840, 2160]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Native image size mismatch"):
        geometry_from_layer_contract(path, (3840, 2160), True)


def test_downsampled_mosaic_needs_explicit_pixel_mapping_even_when_raw_fov_confirmed(tmp_path):
    path = write_layer_contract(tmp_path, out_scale=0.125)
    result = geometry_from_layer_contract(path, (3840, 2160), True)
    assert result["status"] == "needs_calibration"
    assert result["adapted_mode_allowed"] is False
    assert result["native_tile"] is None and result["model_resize_scale"] is None
    assert "mosaic_output_scale_needs_explicit_mapping" in result["reasons"]
    assert result["observed_native_size"] == [1920, 1080]


def test_handoff_cannot_replace_sibling_with_arbitrary_path(tmp_path):
    path = write_layer_contract(tmp_path)
    value = json.loads(path.read_text())
    value["source_binding"]["preflight_file"] = "../preflight.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="sibling"):
        geometry_from_layer_contract(path, (3840, 2160), True)
