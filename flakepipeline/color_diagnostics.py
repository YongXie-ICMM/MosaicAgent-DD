"""Acquisition colour-balance check against a recorded reference substrate colour.

Why this module exists (2026-10-05 controlled comparison, see
docs/diagnostics/20261005_colour_balance/README.md): the current 1920 x 1080 images
labelled broad bare-substrate areas as monolayer in both inference modes. A synthetic
1920 x 1080 downsample of a historical 3840 x 2160 frame was labelled correctly by the
same checkpoint and the same 256/32 -> 512 pipeline (substrate recall >= 99.2 %), so the
pixel-count change is handled by the geometry plan. What the network does not tolerate
is the different colour balance of the new capture mode: the new background is bluer
and less green than the reference substrate, and every class is read one level thicker.
A per-channel gain that only moves the background onto the reference colour restored
the expected class structure; the inverse gain applied to a historical frame
reproduced the failure.

This module measures and records that difference. It never edits source pixels,
never changes the default inference, and any gain-corrected prediction is a diagnostic
result (``selected_for_measurement: false``). The reference itself is derived from model
predictions, not from an independent reflectance standard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_TOLERANCE = 0.05
GAIN_LIMITS = (0.5, 2.0)      # a plausible white-balance correction; anything else is reported, never applied


def _rgb_array(image) -> np.ndarray:
    """Accept a PIL image, a path or an HxWx3 array; return uint8 HxWx3 RGB."""
    if isinstance(image, (str, Path)):
        from PIL import Image
        with Image.open(image) as handle:
            image = handle.convert("RGB")
    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return array


def flat_pixel_mask(rgb: np.ndarray, flat_tol: float = 6.0) -> np.ndarray:
    """Pixels whose grey level differs from each 4-neighbour by at most ``flat_tol``.

    Crystal edges, particles and nuclei fail this test; bare substrate, and the interior
    of large uniform crystals, pass. The mask is a *where to measure* aid, not a class.
    """
    gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    flat = np.ones(gray.shape, bool)
    diff = np.abs(np.diff(gray, axis=0)) <= flat_tol
    flat[1:, :] &= diff
    flat[:-1, :] &= diff
    diff = np.abs(np.diff(gray, axis=1)) <= flat_tol
    flat[:, 1:] &= diff
    flat[:, :-1] &= diff
    return flat


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _luminance_plateaus(lum: np.ndarray, *, half_width: int = 12, min_share: float = 0.10) -> list[dict]:
    """Peaks of the flat-pixel luminance histogram whose +-half_width neighbourhood holds
    at least ``min_share`` of the flat pixels, brightest first. A plateau is wide enough
    to absorb an illumination gradient across one uniform area (vignetting), which a
    fixed colour grid would split into several bins."""
    if lum.size == 0:
        return []
    hist = np.bincount(np.clip(np.rint(lum), 0, 255).astype(np.int64), minlength=256).astype(np.float64)
    kernel = np.ones(7) / 7.0
    smooth = np.convolve(hist, kernel, mode="same")
    total = float(hist.sum())
    plateaus = []
    for centre in range(255, -1, -1):
        lo, hi = max(0, centre - 3), min(255, centre + 3)
        if smooth[centre] <= 0 or smooth[centre] < smooth[lo:hi + 1].max():
            continue
        window = hist[max(0, centre - half_width):min(256, centre + half_width + 1)].sum()
        share = window / total
        if share >= min_share:
            if plateaus and abs(plateaus[-1]["centre"] - centre) <= half_width:
                continue      # same plateau, flatter shoulder
            plateaus.append({"centre": int(centre), "share_of_flat": round(float(share), 4)})
    return plateaus


def background_colour(image, *, flat_tol: float = 6.0, min_fraction: float = 0.2,
                      half_width: int = 12, min_share: float = 0.10) -> dict:
    """Estimate the bare-substrate colour from extended flat areas.

    Flat pixels are grouped by luminance plateaus. On SiO2/Si with thin TMD layers the
    bare substrate is normally the brightest extended flat area, so ``median_rgb`` (the
    value compared with the reference) is the median colour of the flat pixels on the
    brightest plateau that covers at least ``min_share`` of them. The record also lists
    the other plateaus, because a frame dominated by crystals or an unusual illumination
    can defeat this assumption; the method is an aid for a reviewer, not a class
    decision. With too little flat area the global median is used and reported.
    """
    rgb = _rgb_array(image)
    flat = flat_pixel_mask(rgb, flat_tol)
    fraction = float(flat.mean()) if flat.size else 0.0
    plateaus, selected = [], None
    if fraction >= min_fraction:
        pixels = rgb[flat]
        lum = _luminance(pixels)
        plateaus = _luminance_plateaus(lum, half_width=half_width, min_share=min_share)
    else:
        pixels = rgb.reshape(-1, 3)
        lum = None
    if plateaus:
        # An illumination gradient can split one uniform area into neighbouring plateaus;
        # chain plateaus closer than 1.5*half_width to the previous one into a single
        # range. A real class step (substrate -> monolayer) is wider than that here.
        group = [plateaus[0]["centre"]]
        for plateau in plateaus[1:]:
            if group[-1] - plateau["centre"] <= 1.5 * half_width:
                group.append(plateau["centre"])
            else:
                break
        low, high = min(group) - half_width, max(group) + half_width
        selected = {"centre": plateaus[0]["centre"], "share_of_flat": plateaus[0]["share_of_flat"],
                    "merged_centres": group, "luminance_range": [int(low), int(high)]}
        member = pixels[(lum >= low) & (lum <= high)]
        median = np.median(member.astype(np.float32), axis=0)
        std = member.astype(np.float32).std(axis=0)
        method, used = "brightest_flat_luminance_plateau", int(len(member))
    else:
        median = np.median(pixels.astype(np.float32), axis=0)
        std = pixels.astype(np.float32).std(axis=0)
        method, used = ("global_median_insufficient_flat_area" if fraction < min_fraction
                        else "global_flat_median_no_plateau"), int(len(pixels))
    return {"median_rgb": [round(float(v), 2) for v in median],
            "std_rgb": [round(float(v), 2) for v in std],
            "flat_fraction": round(fraction, 4), "flat_tol": float(flat_tol),
            "method": method, "pixels_used": used,
            "selected_plateau": selected, "plateaus": plateaus[:6],
            "substrate_assumption": "bare substrate = brightest extended flat area; verify visually",
            "image_size_wh": [int(rgb.shape[1]), int(rgb.shape[0])]}


def load_reference(path) -> dict | None:
    """Validated reference record, or None when the file is absent.

    A present-but-invalid file raises, so a corrupted reference cannot be mistaken for
    'no reference configured'.
    """
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION \
            or data.get("kind") != "reference_substrate_colour":
        raise ValueError("Unsupported reference substrate colour record")
    for key in ("mean_rgb", "std_rgb"):
        value = data.get(key)
        if not isinstance(value, list) or len(value) != 3 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 255 for v in value):
            raise ValueError(f"Reference {key} must be three 8-bit channel values")
    tolerance = data.get("tolerance_fraction", DEFAULT_TOLERANCE)
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or not 0 < tolerance < 1:
        raise ValueError("Reference tolerance_fraction must lie in (0, 1)")
    data["tolerance_fraction"] = float(tolerance)
    return data


def compare_to_reference(observed: dict, reference: dict) -> dict:
    """Per-channel gain that would move the observed background onto the reference.

    ``verdict`` is 'within_tolerance' or 'colour_balance_differs_from_reference'. A gain
    outside GAIN_LIMITS is 'implausible_gain_check_inputs' and must not be applied.
    """
    obs = np.asarray(observed["median_rgb"], np.float64)
    ref = np.asarray(reference["mean_rgb"], np.float64)
    if np.any(obs <= 0):
        return {"status": "unusable_black_background", "gain_rgb": None, "max_abs_deviation": None,
                "verdict": "unusable_black_background", "tolerance_fraction": reference["tolerance_fraction"],
                "reference_mean_rgb": ref.round(2).tolist(), "observed_median_rgb": obs.round(2).tolist()}
    gain = ref / obs
    deviation = float(np.max(np.abs(gain - 1.0)))
    tolerance = reference["tolerance_fraction"]
    if np.any(gain < GAIN_LIMITS[0]) or np.any(gain > GAIN_LIMITS[1]):
        verdict = "implausible_gain_check_inputs"
    elif deviation <= tolerance:
        verdict = "within_tolerance"
    else:
        verdict = "colour_balance_differs_from_reference"
    return {"status": "compared", "gain_rgb": [round(float(g), 4) for g in gain],
            "max_abs_deviation": round(deviation, 4), "tolerance_fraction": tolerance,
            "verdict": verdict, "reference_mean_rgb": ref.round(2).tolist(),
            "observed_median_rgb": obs.round(2).tolist(),
            "channel_shift_rgb": [round(float(v), 2) for v in (obs - ref)]}


def colour_check(image, reference: dict | None, *, flat_tol: float = 6.0) -> dict:
    """One record per image: observed background plus the comparison, or 'no_reference'."""
    observed = background_colour(image, flat_tol=flat_tol)
    record = {"schema_version": SCHEMA_VERSION, "observed": observed}
    if reference is None:
        record.update(status="no_reference", verdict="no_reference", gain_rgb=None)
        return record
    record.update(compare_to_reference(observed, reference))
    record["reference_kind"] = reference.get("kind")
    record["reference_derived_on"] = reference.get("derived_on")
    return record


def apply_channel_gain(image, gain_rgb) -> np.ndarray:
    """Diagnostic-only per-channel multiplication with clipping; returns a new uint8 array.

    The caller must record the gain and mark the result as a probe. Pixels at 255 after
    clipping lose information; the fraction clipped is worth recording by the caller.
    """
    rgb = _rgb_array(image).astype(np.float32)
    gain = np.asarray(gain_rgb, np.float32).reshape(1, 1, 3)
    if gain.size != 3 or np.any(gain < GAIN_LIMITS[0]) or np.any(gain > GAIN_LIMITS[1]):
        raise ValueError("Refusing to apply an implausible channel gain")
    return np.clip(np.rint(rgb * gain), 0, 255).astype(np.uint8)


def summarize(checks: list[dict]) -> dict:
    """Run-level summary for a manifest: worst verdict and the per-image gains."""
    verdicts = [c.get("verdict") for c in checks]
    if not verdicts:
        worst = "no_images"
    elif any(v == "implausible_gain_check_inputs" for v in verdicts):
        worst = "implausible_gain_check_inputs"
    elif any(v == "colour_balance_differs_from_reference" for v in verdicts):
        worst = "colour_balance_differs_from_reference"
    elif all(v == "no_reference" for v in verdicts):
        worst = "no_reference"
    elif any(v == "unusable_black_background" for v in verdicts):
        worst = "unusable_black_background"
    else:
        worst = "within_tolerance"
    return {"schema_version": SCHEMA_VERSION, "verdict": worst,
            "gains_rgb": [c.get("gain_rgb") for c in checks],
            "meaning": ("Observed background colour compared with the recorded reference substrate colour. "
                        "A difference points at acquisition colour balance (white balance, illumination, camera mode), "
                        "which the trained network does not tolerate; it is not a scale problem and not an accuracy result."),
            "action_on_difference": ("Re-acquire with the camera colour balance set so that bare substrate matches the reference, "
                                     "or run `python tools/student_demo.py colour-probe --probe-inference` as a diagnostic; "
                                     "do not use the uncorrected or the probe fractions as measurements.")}
