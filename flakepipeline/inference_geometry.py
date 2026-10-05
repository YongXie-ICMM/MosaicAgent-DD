"""Pixel-scale planning only; never infer physical field of view from dimensions.

The reference is a previous inference capture mode, not a training crop or a
network architecture requirement. This module does not resize images, labels,
change acquisition motion, or establish segmentation accuracy.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


_PROVENANCE_FIELDS = {
    "scope", "observed_size_source", "reference_size_source",
    "fov_confirmation_source", "note",
}


def _dimensions(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item integer [width, height]")
    if any(type(dimension) is not int for dimension in value):
        raise ValueError(f"{name} dimensions must be integers, not booleans or floats")
    return list(value)


def _provenance(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - _PROVENANCE_FIELDS:
        raise ValueError("Unsupported provenance field; use documented textual source fields")
    result = {}
    for key, text in value.items():
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError(f"provenance.{key} must be nonempty text of at most 4096 characters")
        result[key] = text
    return result


def plan_inference_geometry(
    observed_native_size,
    model_reference_size,
    *,
    same_physical_fov_confirmed=False,
    model_tile=512,
    model_overlap=64,
    provenance=None,
):
    """Return a strict-JSON geometry plan for an explicitly matched physical FOV.

    ``observed_native_size`` and ``model_reference_size`` are integer ``[W, H]``
    dimensions. The latter describes earlier inference photographs, **not** the
    training crop size. ``same_physical_fov_confirmed`` must be a real boolean;
    no pixel-count ratio supplies this confirmation automatically.

    For a confirmed unchanged FOV, isotropic reference/native dimension ratio
    ``s`` gives ``native_tile = model_tile / s`` and
    ``native_overlap = model_overlap / s``. Reference pixel thresholds convert
    by ``1/s`` for lengths and ``1/s**2`` for areas. No threshold rounding is
    applied here. Any fractional native window blocks the adapted mode.

    Invalid input types/configuration raise ValueError. Nonpositive dimensions,
    unknown FOV, anisotropic ratios and fractional windows return an explicit
    ``needs_calibration`` plan with no usable adaptation parameters. A ``ready``
    plan expresses consistency with the caller's stated FOV condition only; it
    does not validate that condition, a physical pixel size, or model accuracy.

    Provenance accepts textual ``scope``, ``observed_size_source``,
    ``reference_size_source``, ``fov_confirmation_source`` and ``note`` fields.
    These are preserved as records and cannot override computed plan fields.
    """
    native = _dimensions(observed_native_size, "observed_native_size")
    reference = _dimensions(model_reference_size, "model_reference_size")
    if type(same_physical_fov_confirmed) is not bool:
        raise ValueError("same_physical_fov_confirmed must be a boolean")
    if type(model_tile) is not int or model_tile <= 0:
        raise ValueError("model_tile must be a positive integer")
    if type(model_overlap) is not int or not 0 <= model_overlap < model_tile:
        raise ValueError("model_overlap must be an integer in [0, model_tile)")
    plan = {
        "schema_version": 1,
        "status": "needs_calibration",
        "adapted_mode_allowed": False,
        "observed_native_size": native,
        "model_reference_size": reference,
        "reference_kind": "previous_inference_capture_mode_not_training_crop",
        "same_physical_fov_confirmed": same_physical_fov_confirmed,
        "model_tile": model_tile,
        "model_overlap": model_overlap,
        "native_tile": None,
        "native_overlap": None,
        "model_resize_scale": None,
        "native_linear_threshold_factor": None,
        "native_area_threshold_factor": None,
        "threshold_conversion": "native threshold = reference pixel threshold * factor; no rounding applied",
        "physical_stage_motion": "unchanged",
        "physical_pixel_size_um": None,
        "physical_calibration_established": False,
        "segmentation_accuracy_established": False,
        "network_architecture_changed": False,
        "provenance": _provenance(provenance),
        "reasons": [],
    }
    reasons = plan["reasons"]
    if any(dimension <= 0 for dimension in native + reference):
        reasons.append("nonpositive_image_dimensions")
        return plan
    if not same_physical_fov_confirmed:
        reasons.append("same_physical_field_of_view_not_confirmed")
        return plan
    scale_x = Fraction(reference[0], native[0])
    scale_y = Fraction(reference[1], native[1])
    if scale_x != scale_y:
        reasons.append("anisotropic_dimension_ratio_requires_calibration")
        return plan
    native_tile = Fraction(model_tile, 1) / scale_x
    native_overlap = Fraction(model_overlap, 1) / scale_x
    if native_tile.denominator != 1 or native_overlap.denominator != 1:
        reasons.append("nonintegral_native_tile_or_overlap")
        return plan
    plan.update(
        status="ready", adapted_mode_allowed=True,
        native_tile=int(native_tile), native_overlap=int(native_overlap),
        model_resize_scale=float(scale_x),
        native_linear_threshold_factor=float(1 / scale_x),
        native_area_threshold_factor=float(1 / (scale_x * scale_x)),
    )
    return plan


def _read_geometry_json(path, expected_sha256=None):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError(f"Geometry source exceeds the bounded read limit: {path.name}")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
                or digest != expected_sha256.lower()):
            raise ValueError(f"Geometry source SHA-256 mismatch: {path.name}")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number in geometry source: {value}")
    data = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError(f"Geometry source must contain a JSON object: {path.name}")
    return data, digest


def geometry_from_layer_contract(
    contract_path,
    reference_size,
    same_physical_fov_confirmed=False,
    model_tile=512,
    model_overlap=64,
):
    """Bind the stitch setup's raw-tile geometry to a later inference plan.

    Reads ``layer_input_contract.json`` and verifies its sibling ``profile.json``
    and ``preflight.json`` hashes. Their recorded native dimensions and render
    settings must agree. Only output scale 1 is currently supported: raw-tile
    size is a capture-mode reference, never the final mosaic's width/height.

    The full contract SHA-256 is retained in ``provenance.note`` so storing this
    plan in the run/cache configuration binds its source. No instrument, model
    or image is opened, and no physical field of view is inferred. This does not
    certify that stitching executed or that a particular mosaic file matches
    this handoff; callers still retain and validate that image's provenance.
    """
    path = Path(contract_path).expanduser().resolve()
    contract, contract_sha = _read_geometry_json(path)
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1 \
            or contract.get("kind") != "layer_input_contract":
        raise ValueError("Unsupported layer input contract")
    source = contract.get("source_binding")
    stitch = contract.get("stitch_profile")
    if not isinstance(source, dict) or not isinstance(stitch, dict):
        raise ValueError("Layer input contract lacks source/profile bindings")
    if source.get("preflight_file") != "preflight.json" or stitch.get("file") != "profile.json":
        raise ValueError("Layer input bindings must name sibling preflight.json and profile.json")
    siblings = {}
    for name, sha in (("profile.json", stitch.get("sha256")),
                      ("preflight.json", source.get("preflight_sha256"))):
        sibling = path.parent / name
        if sibling.resolve().parent != path.parent:
            raise ValueError("Geometry source resolves outside the prepared bundle")
        if sha is None:
            raise ValueError(f"Missing geometry source hash: {name}")
        siblings[name], _ = _read_geometry_json(sibling, sha)
    profile = siblings["profile.json"]
    preflight = siblings["preflight.json"]
    native = _dimensions(contract.get("native_image_size"), "native_image_size")
    if any(value <= 0 for value in native):
        raise ValueError("Contract native_image_size must be positive")
    if contract.get("size_source") != "selected_raw_image_headers":
        raise ValueError("Contract native dimensions must come from selected raw-image headers")
    inspection = preflight.get("inspection", {})
    inventory = preflight.get("input_inventory", {})
    resolved = preflight.get("resolved", {})
    if not all(isinstance(item, dict) for item in (inspection, inventory, resolved)):
        raise ValueError("Invalid preflight geometry records")
    for label, dimensions in (("profile", profile.get("image_size")),
                              ("contract profile", stitch.get("image_size")),
                              ("preflight inspection", inspection.get("image_size")),
                              ("preflight inventory", inventory.get("image_size"))):
        if _dimensions(dimensions, label + " image_size") != native:
            raise ValueError("Native image size mismatch between contract and " + label)
    if resolved.get("profile") != profile:
        raise ValueError("Profile differs from the preflight resolved profile")
    if (source.get("inspection_fingerprint") != inspection.get("fingerprint")
            or source.get("input_inventory_fingerprint") != inventory.get("fingerprint")):
        raise ValueError("Source fingerprints differ from the bound preflight")
    if not source.get("inspection_fingerprint") or not source.get("input_inventory_fingerprint"):
        raise ValueError("Missing source fingerprints in layer input contract")
    scale = stitch.get("out_scale")
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Contract out_scale must be a finite number in (0, 1]")
    if scale != profile.get("out_scale") or scale != resolved.get("out_scale"):
        raise ValueError("Mosaic output scale differs across bound records")
    if (stitch.get("scale_div") != profile.get("scale_div")
            or stitch.get("scale_div") != resolved.get("scale_div")
            or type(stitch.get("full")) is not bool or stitch["full"] != resolved.get("full")):
        raise ValueError("Stitch render settings differ across bound records")
    provenance = {
        "scope": "full_resolution_mosaic_pixel_scale_from_stitch_handoff",
        "observed_size_source": str(path),
        "reference_size_source": "caller-specified previous inference capture mode; not training crop",
        "fov_confirmation_source": "explicit caller confirmation" if same_physical_fov_confirmed is True else "not confirmed",
        "note": (f"layer_input_contract_sha256={contract_sha}; "
                 f"profile_sha256={stitch['sha256']}; preflight_sha256={source['preflight_sha256']}; "
                 "native dimensions refer to raw tiles, not whole-mosaic dimensions; "
                 "the handoff does not establish stitching completion or physical calibration"),
    }
    plan = plan_inference_geometry(native, reference_size,
        same_physical_fov_confirmed=same_physical_fov_confirmed,
        model_tile=model_tile, model_overlap=model_overlap, provenance=provenance)
    if scale != 1:
        plan.update(status="needs_calibration", adapted_mode_allowed=False,
                    native_tile=None, native_overlap=None, model_resize_scale=None,
                    native_linear_threshold_factor=None, native_area_threshold_factor=None)
        plan["reasons"].append("mosaic_output_scale_needs_explicit_mapping")
    return plan
