"""Read-only bridge from scanner records to a verified raw-tile inventory.

Recorded stage steps are provenance, never converted into pixel geometry here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


GRID_NAME = re.compile(r"^mosaic_r(\d+)_c(\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.I)


def _fail(message):
    raise ValueError("Acquisition metadata / 采集记录: " + message)


def _json(data, label):
    def invalid(value):
        _fail(f"Non-finite value in {label}: {value}")
    try:
        result = json.loads(data, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"Cannot read {label}; finish or repair the acquisition journal first / 请先完成或修复采集日志: {exc}")
    if not isinstance(result, dict):
        _fail(f"{label} must contain JSON objects")
    return result


def _read(path, limit):
    if not path.exists():
        return None, None
    if path.is_symlink() or not path.is_file():
        _fail(f"Expected a regular local metadata file: {path.name}")
    before = path.stat()
    if before.st_size > limit:
        _fail(f"{path.name} exceeds the inspection size limit")
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        _fail(f"{path.name} changed while reading; inspect after acquisition pauses / 读取时日志变化，请暂停采集后重试")
    return data, {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _size(value, label):
    if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int or v <= 0 for v in value):
        _fail(f"{label} must be [width, height] in positive integer pixels")
    return value


def inspect_acquisition(data_dir, inventory, layout_mode):
    folder = Path(data_dir)
    session_bytes, session_source = _read(folder / "session.json", 8 * 1024 * 1024)
    event_bytes, event_source = _read(folder / "events.jsonl", 64 * 1024 * 1024)
    sources = [s for s in (session_source, event_source) if s]
    evidence = {"schema_version": 1, "present": bool(sources), "source_files": sources,
                "calibration_status": "unverified_for_current_mode", "geometry_inferred": False}
    warnings = []
    if not sources:
        return evidence, warnings
    if session_bytes is None:
        _fail("events.jsonl exists without session.json; supply the matching session record / 缺少对应 session.json")
    if layout_mode != "flat-grid":
        _fail("Scanner session records require mosaic_r<row>_c<col> raw files / 采集记录须与行列命名的原始图块一起检查")
    session = _json(session_bytes, "session.json")
    if session.get("schema_version") not in (1, 2):
        _fail("Unsupported scanner session schema_version")
    config, camera = session.get("scan_config"), session.get("camera")
    if not isinstance(config, dict) or not isinstance(camera, dict):
        _fail("session.json needs scan_config and camera objects")
    size = inventory["image_size"]
    for key in ("actual_resolution", "requested_resolution"):
        if camera.get(key) is not None and _size(camera[key], f"camera.{key}") != size:
            _fail(f"camera.{key} does not match raw image dimensions / 相机记录与原图尺寸不一致")
    contract = config.get("acquisition_contract")
    expected_size = None
    if contract is not None:
        if not isinstance(contract, dict) or contract.get("schema_version") != 1:
            _fail("Unsupported acquisition_contract schema")
        expected_size = _size(contract.get("expected_image_size"), "acquisition_contract.expected_image_size")
        if expected_size != size:
            _fail("Expected image size does not match raw images / 预期尺寸与原图不一致")
        if contract.get("actual_image_size") is not None and _size(contract["actual_image_size"], "acquisition_contract.actual_image_size") != size:
            _fail("Contract actual image size does not match raw images / 合同实测尺寸与原图不一致")
        if contract.get("verification_status") == "mismatch":
            _fail("Scanner recorded an acquisition size mismatch / 采集程序记录了尺寸不匹配")
        if contract.get("verification_status") != "verified_received_frame":
            warnings.append("Acquisition size contract was not verified on a received frame / 采集尺寸合同尚未通过实帧验证。")
        if contract.get("raw_images_resized") is not False:
            warnings.append("The session does not confirm unresized raw capture / 记录未确认原始采集无缩放。")
    nx, ny = config.get("nx"), config.get("ny")
    if any(type(v) is not int or v <= 0 for v in (nx, ny)):
        _fail("scan_config.nx/ny must be positive integers")
    planned = nx * ny
    if session.get("planned_positions") is not None and session["planned_positions"] != planned:
        _fail("planned_positions conflicts with nx × ny / 计划点数与行列数不符")
    selected = {}
    for tile in inventory["tiles"]:
        name = tile["inner"]
        match = GRID_NAME.fullmatch(name)
        if not match:
            _fail("A selected tile does not have scanner grid coordinates")
        row, col = map(int, match.groups())
        if row >= ny or col >= nx:
            _fail(f"{name} lies outside the recorded scan grid / 图块坐标超出采集网格")
        selected[name] = tile
    steps = {}
    for key in ("dx_steps", "dy_steps"):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            _fail(f"scan_config.{key} must be finite")
        steps[key] = value
    session_id = session.get("session_id")
    captures = {}
    if event_bytes is not None:
        for number, raw in enumerate(event_bytes.splitlines(), 1):
            if not raw.strip():
                continue
            event = _json(raw, f"events.jsonl line {number}")
            if session_id and event.get("session_id") not in (None, session_id):
                _fail("events.jsonl belongs to a different session / 日志属于另一采集批次")
            if event.get("event") != "capture_success":
                continue
            name = event.get("filename") or event.get("image_relative_path")
            if not isinstance(name, str) or not GRID_NAME.fullmatch(name):
                _fail("capture_success has no valid tile filename")
            if name in captures and captures[name] != event:
                warnings.append("Repeated capture_success entries are present; unique filenames were counted / 存在重复保存记录，按不同文件名计数。")
            captures[name] = event
            tile = selected.get(name)
            if tile is None:
                continue
            if event.get("saved_image_resolution") is not None and _size(event["saved_image_resolution"], "saved_image_resolution") != size:
                _fail(f"Recorded dimensions differ for {name} / 保存记录与原图尺寸不同")
            if event.get("image_sha256") is not None and event["image_sha256"] != tile.get("sha256"):
                _fail(f"SHA-256 mismatch for {name}; these are not the recorded original bytes / 文件与采集时原图校验值不一致")
            if event.get("image_bytes") is not None and event["image_bytes"] != tile.get("source", {}).get("size"):
                _fail(f"Recorded byte size differs for {name} / 文件大小与采集记录不符")
            match = GRID_NAME.fullmatch(name)
            if event.get("row") is not None and event["row"] != int(match[1]) or event.get("col") is not None and event["col"] != int(match[2]):
                _fail(f"Capture coordinates conflict with {name}")
        missing = sorted(set(selected) - set(captures))
        if missing:
            _fail(f"Selected raw tiles lack capture_success records: {', '.join(missing[:3])} / 部分原图缺少保存成功记录")
    else:
        warnings.append("events.jsonl is absent; saved-image checksums and capture outcomes cannot be cross-checked / 缺少事件日志，无法逐张核对保存结果与校验值。")
    count = len(selected)
    photos_saved = session.get("photos_saved")
    if photos_saved is not None and (type(photos_saved) is not int or photos_saved < count):
        _fail("photos_saved is smaller than the selected raw grid or is invalid / 保存计数小于当前原图数或无效")
    checksum_count = sum(1 for name in selected if captures.get(name, {}).get("image_sha256") is not None)
    if event_bytes is not None and checksum_count < count:
        warnings.append("Some selected images have no recorded acquisition checksum; names and dimensions alone cannot verify unchanged bytes / 部分图像没有采集校验值，仅文件名和尺寸不能确认原始字节未变。")
    if count < planned:
        scope = "subset" if session.get("status") == "completed" else "partial_or_subset"
        warnings.append(f"Only {count} of {planned} planned tiles are selected; this is not the complete scan / 只选择了计划 {planned} 张中的 {count} 张，不是完整扫描。")
    else:
        scope = "full_recorded_grid"
    if session.get("status") != "completed":
        warnings.append("Scanner session is not marked completed; these images do not establish a completed acquisition / 采集记录未标为完成，不能称为完整扫描。")
    if session.get("stage_simulated"):
        warnings.append("Scanner records simulated stage movement / 该记录使用模拟位移台。")
    warnings.append("Pixel displacement calibration is still required. Recorded stage steps and 1920 × 1080 dimensions do not establish unchanged optics, field of view or compatibility with the historical 4K geometry / 仍须核对像素位移标定；位移台步数与图像尺寸不能证明旧标定适用。")
    evidence.update(session_id=session_id, session_schema_version=session["schema_version"],
                    session_status=session.get("status"), expected_image_size=expected_size,
                    recorded_image_size=camera.get("actual_resolution"), actual_image_size=size,
                    grid={"nx": nx, "ny": ny, "planned_tiles": planned, "selected_tiles": count,
                          "selection_scope": scope}, steps=steps, order=config.get("order"),
                    sample_id=config.get("sample_id"), objective=config.get("objective"),
                    camera={key: camera.get(key) for key in ("backend", "capture_api", "camera_index", "requested_resolution", "resolution_source")},
                    recorded_calibration_status=config.get("calibration_status"),
                    acquisition_contract=contract, unique_capture_records=len(captures) if event_bytes is not None else None,
                    matching_selected_capture_records=count if event_bytes is not None else None,
                    checksum_verified_selected_tiles=checksum_count,
                    physical_coverage_verified=False)
    return evidence, list(dict.fromkeys(warnings))
