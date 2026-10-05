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


def result_record(result):
    return asdict(result)


def run_demo(root: Path = REPO, *, inference_geometry=None, comparison_only=False,
             use_recorded_geometry=True) -> Path:
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
                image_path.write_bytes(raw)
                context = dict(config, work=output, sample=sample["sample_id"], weights=str(assets["weights"]))
                segmented = segment(context, image_path)
                counted = stats(context, segmented.data["mask"], segmented.data["valid_mask"], exclusion_path=None)
            preview = counted.data.get("layer_map_preview")
            if not preview or not Path(preview).is_file():
                raise DemoError("Statistics did not produce a layer map / 未生成层数预测图")
            mask_color = output / "mask_color.png"
            shutil.copyfile(preview, mask_color)
            row = {"sample_id": sample["sample_id"], "archive": "data/demo/source_images.zip",
                   "member": sample["member"], "member_sha256": sample["sha256"].lower(),
                   "source_image_size": [sample["width"], sample["height"]],
                   "segmentation": result_record(segmented), "statistics": result_record(counted),
                   "requires_human_review": bool(review_required or segmented.escalate or counted.escalate),
                   "asset_review_status": manifest["review_status"],
                   "asset_review_note": manifest["review_note"],
                   "outputs": {"mask_color.png": {"sha256": digest(mask_color), "bytes": mask_color.stat().st_size}}}
            manifest["samples"].append(row)
            atomic_json(manifest_path, manifest)
        manifest["status"] = "completed_with_review_flags" if any(row["requires_human_review"] for row in manifest["samples"]) else "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        atomic_json(root / "outputs/demo/latest.json", {"run": run.relative_to(root).as_posix()})
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = str(exc)
        atomic_json(manifest_path, manifest)
        raise
    print("Results / 结果目录: " + str(run), flush=True)
    if review_required:
        print("Software completed; model applicability not validated; do not use fractions scientifically.\n"
              "软件运行完成；模型对当前图像的适用性尚未验证，不能将这些比例用于科学结论。", flush=True)
        if manifest["review_note"]:
            print("Review note / 审核说明: " + str(manifest["review_note"]), flush=True)
    print("Open 03_open_workbench to compare originals and layer maps. / 双击03_open_workbench查看原图和层数预测。", flush=True)
    return run


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
    parser.add_argument("action", choices=("run", "check", "workbench", "compare-scale"), nargs="?", default="run")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            run_demo()
        elif args.action == "compare-scale":
            compare_scale()
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
