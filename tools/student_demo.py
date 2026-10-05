#!/usr/bin/env python3
"""Portable, local layer-number demonstration using the supplied trusted assets.

Only segmentation and pixel statistics run. No orchestrator, remote model,
hardware, exclusion policy or repair step is invoked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile


REPO = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_ASSET_BYTES = 1024 * 1024 * 1024
CONFIG = {"device": "cpu", "tile": 512, "overlap": 64, "batch": 1,
          "threads": min(4, os.cpu_count() or 1), "filler": "none",
          "border_trim_px": 0, "map_preview_div": 1, "no_ai": True}


class DemoError(ValueError):
    """An actionable error, before loading a checkpoint when assets are invalid."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def relative_name(value, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise DemoError(f"Invalid {label} path / 路径无效: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise DemoError(f"Unsafe {label} path / 不安全的路径: {value!r}")
    return value


def local_asset(root: Path, value, label: str) -> Path:
    value = relative_name(value, label)
    path = root / value
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DemoError(f"{label} points outside this package / 文件指向本包之外") from exc
    if not path.is_file():
        raise DemoError(f"Missing {label}: {value}. Use the complete student package containing data and weights.\n"
                        f"缺少文件：{value}。请使用含示例数据和权重的完整学生包，不要仅下载源代码。")
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise DemoError(f"{label} exceeds the supported package size / 文件过大")
    return path


def expected_hash(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise DemoError(f"Missing full SHA-256 for {label} / 缺少完整文件校验值")
    return value.lower()


def checked_file(root, record, label):
    if not isinstance(record, dict):
        raise DemoError(f"Missing {label} asset record / 资源清单不完整")
    path = local_asset(root, record.get("path"), label)
    sha = expected_hash(record.get("sha256"), label)
    if digest(path) != sha:
        raise DemoError(f"{label} SHA-256 mismatch. Restore the supplied original; no model was loaded.\n"
                        f"{label} 校验不一致，请重新解压完整原包；尚未加载模型。")
    return path, sha


def validate_assets(root: Path) -> dict:
    """Check all inputs before any segmentation/torch import or output creation."""
    root = root.resolve()
    manifest_path = local_asset(root, "data/demo/assets.json", "asset manifest")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise DemoError("Asset manifest is too large / 资源清单过大")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise DemoError("Cannot read data/demo/assets.json / 无法读取资源清单") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DemoError("Unsupported asset manifest / 不支持的资源清单版本")
    archive, archive_sha = checked_file(root, manifest.get("archive"), "source ZIP")
    weights, weights_sha = checked_file(root, manifest.get("weights"), "checkpoint")
    weight_relative = PurePosixPath(manifest["weights"]["path"])
    if manifest["archive"]["path"] != "data/demo/source_images.zip" or \
            weight_relative.parent != PurePosixPath("weights") or weight_relative.suffix != ".pth":
        raise DemoError("Use the supplied source ZIP and manifest-listed weights/*.pth checkpoint / 请使用本包提供的原图ZIP和清单内指定权重")
    if type(manifest.get("review_required", False)) is not bool:
        raise DemoError("review_required must be true or false / 人工审核标记必须是布尔值")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 20:
        raise DemoError("The demo needs 1–20 manifest-listed images / 示例清单需包含1至20张图")
    identifiers, members = set(), set()
    try:
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            names = [entry.filename for entry in infos]
            if len(set(names)) != len(names):
                raise DemoError("Duplicate ZIP entries / ZIP内有重复文件名")
            for entry in infos:
                relative_name(entry.filename.rstrip("/"), "ZIP entry")
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                    raise DemoError("Symbolic links/encrypted ZIP entries are unsupported / 不支持符号链接或加密ZIP")
            for sample in samples:
                if not isinstance(sample, dict):
                    raise DemoError("Invalid sample record / 样品记录无效")
                sample_id = sample.get("sample_id")
                if not isinstance(sample_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", sample_id) is None or sample_id in identifiers:
                    raise DemoError("Invalid/duplicate sample_id / 样品编号无效或重复")
                member = relative_name(sample.get("member"), "image member")
                if member in members:
                    raise DemoError("An image is listed twice / 同一原图重复列入清单")
                identifiers.add(sample_id); members.add(member)
                entry = source.getinfo(member)
                if entry.is_dir() or not 0 < entry.file_size <= MAX_IMAGE_BYTES:
                    raise DemoError("Invalid image size in ZIP / ZIP内图片大小无效")
                expected = expected_hash(sample.get("sha256"), sample_id)
                if hashlib.sha256(source.read(entry)).hexdigest() != expected:
                    raise DemoError(f"Image checksum mismatch: {member} / 原图校验不一致")
                for axis in ("width", "height"):
                    if type(sample.get(axis)) is not int or not 1 <= sample[axis] <= 32768:
                        raise DemoError(f"Invalid image {axis} / 图片尺寸清单无效")
            if {entry.filename for entry in infos if not entry.is_dir()} != members:
                raise DemoError("ZIP files and asset manifest differ / ZIP文件与资源清单不完全对应")
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DemoError("Cannot verify source ZIP / 无法核验原图ZIP: " + str(exc)) from exc
    return {"manifest": manifest, "manifest_sha256": digest(manifest_path), "archive": archive,
            "archive_sha256": archive_sha, "weights": weights, "weights_sha256": weights_sha,
            "samples": samples}


def load_stages():
    # Deliberately lazy: hashes must be checked before this imports torch via a stage.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from flakepipeline.stages import stage_segment, stage_stats
    return stage_segment, stage_stats


REFERENCE_COLOUR = "configs/reference_substrate_colour.json"


def colour_checks(root: Path, assets: dict) -> tuple[list, dict | None]:
    """Compare each supplied image's bare-substrate colour with the recorded reference.

    Runs before any model import (numpy/PIL only). A missing reference file yields
    'no_reference' records; a present but invalid one raises, so it cannot pass silently.
    The check records acquisition colour balance; it never edits pixels or labels.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from flakepipeline import color_diagnostics as cd
    from PIL import Image
    try:
        reference = cd.load_reference(root / REFERENCE_COLOUR)
    except (OSError, ValueError) as exc:
        raise DemoError("Invalid reference substrate colour record / 参考衬底颜色记录无效: " + str(exc)) from exc
    checks = []
    with zipfile.ZipFile(assets["archive"]) as archive:
        for sample in assets["samples"]:
            with archive.open(sample["member"]) as stream, Image.open(stream) as image:
                record = cd.colour_check(image.convert("RGB"), reference)
            record["sample_id"] = sample["sample_id"]
            checks.append(record)
    return checks, reference


def colour_warning(summary: dict) -> str | None:
    verdict = summary.get("verdict")
    if verdict in ("within_tolerance", "no_reference", "no_images"):
        return None
    gains = ", ".join(str(g) for g in summary.get("gains_rgb", []) if g)
    return ("Colour balance differs from the reference substrate colour (per-channel gain to reference: "
            f"{gains or 'n/a'}). The trained network does not tolerate this; predictions need review and the "
            "acquisition white balance should be corrected before new scans.\n"
            f"颜色/白平衡与参考衬底颜色不一致（校正到参考所需的每通道增益：{gains or '无'}）。"
            "现有网络不能容忍这种偏差；本次预测需人工复核，新扫描前先在采集端校正白平衡。")


def result_record(result):
    return asdict(result)


def run_demo(root: Path = REPO, *, inference_geometry=None, comparison_only=False,
             use_recorded_geometry=True, colour_probe=None) -> Path:
    """Run the layer demo. ``colour_probe`` (developer diagnostic only) is a mapping
    sample_id -> per-channel gain; the gain is applied to a temporary copy of each
    image before inference, the run is marked ``diagnostic_colour_probe`` with
    ``selected_for_measurement: false`` and ``latest.json`` is left untouched."""
    root = root.resolve()
    assets = validate_assets(root)
    from PIL import Image
    # Image decoding is also checked before a model is loaded.
    with zipfile.ZipFile(assets["archive"]) as archive:
        for sample in assets["samples"]:
            with archive.open(sample["member"]) as stream, Image.open(stream) as image:
                if image.size != (sample["width"], sample["height"]):
                    raise DemoError("Image dimensions differ from the manifest / 原图尺寸与清单不一致")
                image.verify()
    checks, colour_reference = colour_checks(root, assets)
    if colour_probe is not None:
        if not isinstance(colour_probe, dict) or set(colour_probe) != {s["sample_id"] for s in assets["samples"]}:
            raise DemoError("A colour probe needs one gain per listed sample / 颜色探针需为每张图提供增益")
        from flakepipeline import color_diagnostics as cd
        for gain in colour_probe.values():
            if not isinstance(gain, (list, tuple)) or len(gain) != 3 or not all(
                    isinstance(g, (int, float)) and not isinstance(g, bool) and cd.GAIN_LIMITS[0] <= g <= cd.GAIN_LIMITS[1] for g in gain):
                raise DemoError("Colour probe gain must be three plausible channel factors / 颜色探针增益无效")
    config = dict(CONFIG)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if use_recorded_geometry and inference_geometry is None and "capture_geometry" in assets["manifest"]:
        from flakepipeline.inference_geometry import plan_inference_geometry
        capture = assets["manifest"]["capture_geometry"]
        if not isinstance(capture, dict) or capture.get("same_physical_fov_confirmed") is not True \
                or not isinstance(capture.get("confirmation_source"), str) or not capture["confirmation_source"].strip():
            raise DemoError("Confirm the recorded field of view before scale adaptation / 请先核对并记录实际视野")
        sizes = {(row["width"], row["height"]) for row in assets["samples"]}
        if len(sizes) != 1:
            raise DemoError("Mixed resolutions require separate geometry plans / 不同分辨率需分开配置")
        inference_geometry = plan_inference_geometry(list(next(iter(sizes))),
            assets["manifest"].get("inference_reference_capture_size"),
            same_physical_fov_confirmed=True, model_tile=CONFIG["tile"], model_overlap=CONFIG["overlap"],
            provenance={"scope": "operator_confirmed_capture_mode_scale_mapping",
                        "observed_size_source": "Decoded and hash-verified supplied camera images",
                        "reference_size_source": "Asset manifest previous inference capture mode, not training crop",
                        "fov_confirmation_source": capture["confirmation_source"]})
    if inference_geometry is not None:
        from flakepipeline.inference_geometry import plan_inference_geometry
        required = {"observed_native_size", "model_reference_size", "same_physical_fov_confirmed"}
        if not isinstance(inference_geometry, dict) or not required.issubset(inference_geometry):
            raise DemoError("Incomplete geometry plan / 识别尺度配置不完整")
        expected = plan_inference_geometry(
            inference_geometry["observed_native_size"], inference_geometry["model_reference_size"],
            same_physical_fov_confirmed=inference_geometry["same_physical_fov_confirmed"],
            model_tile=CONFIG["tile"], model_overlap=CONFIG["overlap"],
            provenance=inference_geometry.get("provenance"))
        if expected != inference_geometry or not expected["adapted_mode_allowed"]:
            raise DemoError("Geometry is not ready / 识别尺度尚未核对")
        if any([row["width"], row["height"]] != expected["observed_native_size"] for row in assets["samples"]):
            raise DemoError("Actual images do not match the geometry / 实际照片尺寸与识别配置不一致")
        config["inference_geometry"] = expected
    segment, stats = load_stages()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run = root / "outputs" / "demo" / stamp
    run.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(run / ".cache" / "matplotlib"))
    review_required = assets["manifest"].get("review_required", False)
    manifest = {"schema_version": 1, "kind": "local_layer_number_demo", "status": "running",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
                "assets_manifest_sha256": assets["manifest_sha256"],
                "archive_sha256": assets["archive_sha256"],
                "weights": {"path": assets["manifest"]["weights"]["path"], "sha256": assets["weights_sha256"]},
                "review_required": review_required,
                "review_status": assets["manifest"].get("review_status", "not_assessed"),
                "review_note": assets["manifest"].get("review_note", ""),
                "remote_model_calls": 0, "orchestrator_used": False, "region_exclusion_applied": False,
                "repairs_applied": False, "physical_calibration": "not_applied",
                "interpretation": "Predicted layer labels; not physical ground truth or instance boundaries.",
                "samples": []}
    manifest_path = run / "results" / "run_manifest.json"
    manifest["substrate"] = assets["manifest"].get("substrate", {})
    manifest["inference_geometry"] = inference_geometry
    manifest["capture_geometry_confirmation"] = assets["manifest"].get("capture_geometry")
    if comparison_only or (inference_geometry and inference_geometry.get("provenance", {}).get("scope") == "provisional_same_fov_hypothesis_for_comparison"):
        manifest["comparison_only"] = True
        manifest["review_required"] = True
        review_required = True
        manifest["review_note"] = "Scale comparison only. Independent physical pixel calibration and layer accuracy are not established."
    from flakepipeline import color_diagnostics as cd
    colour_summary = cd.summarize(checks)
    colour_summary["reference_file"] = REFERENCE_COLOUR if colour_reference is not None else None
    colour_summary["reference_mean_rgb"] = colour_reference["mean_rgb"] if colour_reference is not None else None
    manifest["colour_check"] = colour_summary
    colour_flag = colour_warning(colour_summary) is not None
    if colour_flag:
        manifest["review_required"] = True
        review_required = True
    if colour_probe is not None:
        manifest["kind"] = "diagnostic_colour_probe"
        manifest["review_required"] = True
        review_required = True
        manifest["selected_for_measurement"] = False
        manifest["colour_probe"] = {
            "gain_rgb_by_sample": {k: [float(g) for g in v] for k, v in colour_probe.items()},
            "applied_to": "temporary copy of each decoded image before inference; source bytes unchanged",
            "gain_source": "estimated from the image's own brightest flat area against the recorded reference substrate colour",
            "meaning": ("Diagnostic probe of the colour-balance hypothesis with the unchanged checkpoint and geometry. "
                        "Plausible-looking output is not an accuracy result; independent reference regions are still required."),
            "latest_json_updated": False}
        manifest["review_note"] = ("Colour probe only: per-channel gains were applied to temporary copies. "
                                   "Not a measurement; compare against independent reference regions.")
    atomic_json(manifest_path, manifest)
    try:
        for index, sample in enumerate(assets["samples"], 1):
            print(f"[{index}/{len(assets['samples'])}] Layer prediction / 层数识别: {sample['sample_id']}", flush=True)
            # Recheck immediately before each checkpoint load, including a long run.
            if digest(assets["weights"]) != assets["weights_sha256"]:
                raise DemoError("Checkpoint changed after preflight / 预检后权重发生变化")
            output = run / "results" / sample["sample_id"]
            output.mkdir(parents=True)
            with tempfile.TemporaryDirectory(prefix="mosaic_layer_demo_") as temp, zipfile.ZipFile(assets["archive"]) as archive:
                raw = archive.read(sample["member"])
                if hashlib.sha256(raw).hexdigest() != sample["sha256"].lower():
                    raise DemoError("Original image changed after preflight / 预检后原图发生变化")
                image_path = Path(temp) / (sample["sample_id"] + PurePosixPath(sample["member"]).suffix)
                probe_record = None
                if colour_probe is None:
                    image_path.write_bytes(raw)
                else:
                    # Diagnostic only: the gain is applied to a temporary decoded copy; the
                    # archive bytes and the recorded colour check refer to the original.
                    import io
                    import numpy as np
                    with Image.open(io.BytesIO(raw)) as original:
                        corrected = cd.apply_channel_gain(original.convert("RGB"), colour_probe[sample["sample_id"]])
                    clipped = float(np.mean(corrected == 255))
                    Image.fromarray(corrected).save(image_path, format="PNG")
                    probe_record = {"gain_rgb": [float(g) for g in colour_probe[sample["sample_id"]]],
                                    "clipped_channel_fraction": round(clipped, 6),
                                    "corrected_input_sha256": digest(image_path)}
                context = dict(config, work=output, sample=sample["sample_id"], weights=str(assets["weights"]))
                segmented = segment(context, image_path)
                counted = stats(context, segmented.data["mask"], segmented.data["valid_mask"], exclusion_path=None)
            preview = counted.data.get("layer_map_preview")
            if not preview or not Path(preview).is_file():
                raise DemoError("Statistics did not produce a layer map / 未生成层数预测图")
            mask_color = output / "mask_color.png"
            shutil.copyfile(preview, mask_color)
            check = next((c for c in checks if c["sample_id"] == sample["sample_id"]), None)
            row = {"sample_id": sample["sample_id"], "archive": "data/demo/source_images.zip",
                   "member": sample["member"], "member_sha256": sample["sha256"].lower(),
                   "source_image_size": [sample["width"], sample["height"]],
                   "segmentation": result_record(segmented), "statistics": result_record(counted),
                   "requires_human_review": bool(review_required or segmented.escalate or counted.escalate or colour_flag),
                   "asset_review_status": manifest["review_status"],
                   "asset_review_note": manifest["review_note"],
                   "colour_check": check,
                   "colour_probe": probe_record,
                   "outputs": {"mask_color.png": {"sha256": digest(mask_color), "bytes": mask_color.stat().st_size}}}
            manifest["samples"].append(row)
            atomic_json(manifest_path, manifest)
        manifest["status"] = "completed_with_review_flags" if any(row["requires_human_review"] for row in manifest["samples"]) else "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        if colour_probe is None:
            atomic_json(root / "outputs/demo/latest.json", {"run": run.relative_to(root).as_posix()})
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = str(exc)
        atomic_json(manifest_path, manifest)
        raise
    print("Results / 结果目录: " + str(run), flush=True)
    warning = colour_warning(colour_summary)
    if warning and colour_probe is None:          # the probe command has already printed it
        print("Colour check / 颜色检查: " + warning, flush=True)
    if review_required:
        print("Software completed; model applicability not validated; do not use fractions scientifically.\n"
              "软件运行完成；模型对当前图像的适用性尚未验证，不能将这些比例用于科学结论。", flush=True)
        if manifest["review_note"]:
            print("Review note / 审核说明: " + str(manifest["review_note"]), flush=True)
    if colour_probe is not None:
        print("Diagnostic colour probe; latest.json unchanged; not a measurement. / 颜色诊断探针结果，不更新最新结果指针，不是测量。", flush=True)
    else:
        print("Open 03_open_workbench to compare originals and layer maps. / 双击03_open_workbench查看原图和层数预测。", flush=True)
    return run


def colour_probe(root: Path = REPO, *, probe_inference=False) -> Path:
    """Measure the acquisition colour balance of the supplied images against the reference.

    Writes outputs/demo/colour_probe.json. With ``probe_inference`` it additionally runs
    the unchanged checkpoint on gain-corrected temporary copies as a diagnostic run that
    never becomes the latest result. Nothing here validates accuracy.
    """
    root = root.resolve()
    assets = validate_assets(root)
    checks, reference = colour_checks(root, assets)
    if reference is None:
        raise DemoError(f"No reference substrate colour record ({REFERENCE_COLOUR}) / 缺少参考衬底颜色记录")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from flakepipeline import color_diagnostics as cd
    summary = cd.summarize(checks)
    record = {"status": "diagnostic_only_requires_review", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "reference_file": REFERENCE_COLOUR, "reference_mean_rgb": reference["mean_rgb"],
              "reference_limits": reference.get("limits", []),
              "summary": summary, "checks": checks, "probe_run": None, "selected_for_measurement": False,
              "interpretation": ("Per-image bare-substrate colour versus the recorded reference. A gain near 1 in all "
                                 "channels means the capture colour balance matches the mode on which the checkpoint "
                                 "was previously reviewed; a larger gain means acquisition colour balance differs. "
                                 "Correct it at the camera; a software gain is a diagnostic probe, not calibration.")}
    for check in checks:
        print(f"{check['sample_id']}: {check['verdict']}; background RGB {check['observed']['median_rgb']}; "
              f"gain to reference {check.get('gain_rgb')}", flush=True)
    warning = colour_warning(summary)
    if warning:
        print("Colour check / 颜色检查: " + warning, flush=True)
    else:
        print("Colour balance within tolerance of the reference. / 颜色与参考衬底一致（在容差内）。", flush=True)
    if probe_inference:
        usable = {c["sample_id"]: c.get("gain_rgb") for c in checks}
        if any(g is None for g in usable.values()) or any(c["verdict"] == "implausible_gain_check_inputs" for c in checks):
            raise DemoError("Probe inference needs a plausible gain for every image / 每张图都需要可信的增益才能运行探针")
        run = run_demo(root, colour_probe=usable)
        record["probe_run"] = run.relative_to(root).as_posix()
    path = root / "outputs/demo/colour_probe.json"
    atomic_json(path, record)
    print("Colour probe record / 颜色诊断记录: " + str(path), flush=True)
    return path


def compare_scale(root: Path = REPO) -> Path:
    """Compare a stated same-FOV hypothesis; do not accept it as calibration."""
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from flakepipeline.inference_geometry import plan_inference_geometry
    assets = validate_assets(root)
    sizes = {(row["width"], row["height"]) for row in assets["samples"]}
    if len(sizes) != 1:
        raise DemoError("Mixed native resolutions need separate comparisons / 不同分辨率需要分开比较")
    reference = assets["manifest"].get("inference_reference_capture_size")
    capture = assets["manifest"].get("capture_geometry", {})
    confirmed = isinstance(capture, dict) and capture.get("same_physical_fov_confirmed") is True \
        and isinstance(capture.get("confirmation_source"), str) and bool(capture["confirmation_source"].strip())
    geometry = plan_inference_geometry(list(next(iter(sizes))), reference,
        same_physical_fov_confirmed=True, model_tile=CONFIG["tile"], model_overlap=CONFIG["overlap"],
        provenance={"scope": "confirmed_fov_scale_comparison" if confirmed else "provisional_same_fov_hypothesis_for_comparison",
                    "observed_size_source": "Decoded and hash-verified supplied camera images",
                    "reference_size_source": "Recorded previous 0409 inference capture mode, not training crop size",
                    "fov_confirmation_source": capture["confirmation_source"] if confirmed else "ASSUMED ONLY for this comparison; not instrument-confirmed"})
    if not geometry["adapted_mode_allowed"]:
        raise DemoError("Cannot form this scale comparison / 无法生成尺度对照: " + ", ".join(geometry["reasons"]))
    print("Comparison only; field-of-view record is retained; accuracy is not established. / 仅作对照，保留实际视野确认记录；尚未验证准确率。", flush=True)
    native = run_demo(root, comparison_only=True, use_recorded_geometry=False)
    adapted = run_demo(root, inference_geometry=geometry, comparison_only=True)
    record = {"status": "comparison_only_requires_review", "geometry": geometry,
              "direct_run": native.relative_to(root).as_posix(),
              "adapted_run": adapted.relative_to(root).as_posix(),
              "selected_for_measurement": False, "physical_stage_motion": "unchanged"}
    path = root / "outputs/demo/scale_comparison.json"
    atomic_json(path, record)
    print("Scale comparison / 尺度对照: " + str(path), flush=True)
    return path


def prepare_workbench_config(root: Path = REPO) -> Path:
    root = root.resolve()
    latest = root / "outputs/demo/latest.json"
    inference = ""
    if latest.is_file():
        record = json.loads(latest.read_text(encoding="utf-8"))
        relative = relative_name(record.get("run"), "latest run")
        if not relative.startswith("outputs/demo/"):
            raise DemoError("Invalid latest demo path / 最近结果路径无效")
        run = root / relative
        try:
            run.resolve().relative_to((root / "outputs/demo").resolve())
        except ValueError as exc:
            raise DemoError("Demo results point outside this package / 结果目录指向本包之外") from exc
        if (run / "results/run_manifest.json").is_file():
            inference = str(run.resolve())
    path = root / "outputs/workbench/workbench.local.json"
    atomic_json(path, {"project_root": str(root), "scan_repo": str(root / "acquisition"),
                       "inference_results": inference})
    return path


def available_port(start=8792):
    for port in range(start, start + 20):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise DemoError("No available local workbench port / 工作台本地端口均被占用")


def open_workbench(root: Path = REPO, no_browser=False) -> int:
    root = root.resolve()
    config = prepare_workbench_config(root)
    command = [sys.executable, "-B", str(root / "tools/analysis_workbench/server.py"),
               "--config", str(config), "--port", str(available_port())]
    if no_browser:
        command.append("--no-browser")
    return subprocess.call(command, cwd=root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "workbench", "compare-scale", "colour-probe", "color-probe"),
                        nargs="?", default="run")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--probe-inference", action="store_true",
                        help="colour-probe only: also run the unchanged checkpoint on gain-corrected temporary copies (diagnostic)")
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            run_demo()
        elif args.action == "compare-scale":
            compare_scale()
        elif args.action in ("colour-probe", "color-probe"):
            colour_probe(probe_inference=args.probe_inference)
        elif args.action == "check":
            assets = validate_assets(REPO)
            print(f"Verified {len(assets['samples'])} images and supplied checkpoint; no model loaded. / 原图与权重校验通过，未加载模型。")
        else:
            return open_workbench(no_browser=args.no_browser)
    except (DemoError, OSError, ValueError, ImportError) as exc:
        print("Not completed / 尚未完成: " + str(exc), file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Run 01_install first. / 请先运行01_install安装依赖。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
