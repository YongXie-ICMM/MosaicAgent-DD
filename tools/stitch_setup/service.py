"""Local, preparation-only stitching setup; no subprocesses, hardware or AI calls.

The web server owns origin/CSRF checks. This module inspects local raw images and
writes a new bundle only after binding an explicit profile to their inventory.
Heavy image modules are imported on demand, never when the workbench starts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = {
    "group_regex": r"(?i)^(?:a_)?L(?P<index>\d+)_(?P<direction>up|down)(?:\.zip)?$",
    "tile_regex": r"(?i)^(?P<idx>\d+(?:\.\d+)?)\.(?:png|jpg|jpeg|tif|tiff|bmp)$",
    "serpentine": True,
    "major_axis": "column",
    "reverse_direction_token": "up",
}


def _modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("stitch_profile"), importlib.import_module("tiles")


def _text(body, key, *, required=False, limit=4096, multiline=False):
    value = body.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text / 必须为文本。")
    value = value.strip()
    if len(value) > limit or any(ord(ch) < 32 and not (multiline and ch in "\n\r\t") for ch in value):
        raise ValueError(f"{key} is too long or contains control characters / 文本过长或含控制字符。")
    if required and not value:
        raise ValueError(f"{key} is required / 不能为空。")
    return value


def _finite_number(body, key, default):
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number / 必须为有限数值。")
    return value


def _read_layout(path, tiles_module):
    if not path.is_file() or path.stat().st_size > 131072:
        raise ValueError("Layout must be an existing JSON file under 128 KiB / 布局须为小于 128 KiB 的现有 JSON 文件。")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value in layout: {value}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read layout JSON / 无法读取布局 JSON: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("layout"), dict):
        data = data["layout"]
    if not isinstance(data, dict):
        raise ValueError("Layout must be a JSON object / 布局须为 JSON 对象。")
    if data.get("major_axis", "column") != "column":
        raise ValueError("Row-major layouts are not supported here / 此入口暂不支持按行扫描布局。")
    if "serpentine" in data and type(data["serpentine"]) is not bool:
        raise ValueError("layout.serpentine must be a boolean / 必须为布尔值。")
    return {key: data[key] for key in tiles_module.LAYOUT_FIELDS if key in data}


def _fingerprint(inventory, data_dir, layout, layout_path, acquisition):
    values = {"schema_version": 1, "input_fingerprint": inventory["fingerprint"],
              "data_dir": str(data_dir), "layout": layout, "layout_path": layout_path,
              "acquisition": acquisition}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _flat_companions(data_dir, tiles_module):
    """Keep provisional/rejected frames separate from the scanner's saved grid."""
    excluded = []
    column_name = re.compile(DEFAULT_LAYOUT["group_regex"])
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith((".", "_")):
            continue
        if path.is_file() and path.suffix.lower() == ".zip":
            raise ValueError("Flat-grid images and ZIP containers are mixed; select one acquisition directory / 平铺图像与 ZIP 容器混合，请选择单次采集目录。")
        if not path.is_dir():
            continue
        if path.name.casefold() == "review_candidates":
            excluded.append((path, "review_candidates contains provisional/rejected frames, excluded from raw tiles / 候选或被拒绝帧不计入原始拼接图像"))
            continue
        if column_name.fullmatch(path.name) or any(
                child.is_file() and (child.suffix.lower().lstrip(".") in tiles_module.IMAGE_EXTS
                                    or child.suffix.lower() == ".zip")
                for child in path.rglob("*")):
            raise ValueError(f"Flat-grid images and another image container are mixed / 平铺图像与其他图像容器混合: {path}")
        excluded.append((path, "non-image/log directory excluded from raw tiles / 非图像或日志目录不计入原始拼接图像"))
    return excluded


def _inspect(body):
    data_dir = Path(_text(body, "data_dir", required=True)).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError("Raw tile directory does not exist / 原始图像目录不存在。")
    layout_text = _text(body, "layout_path")
    layout_path = Path(layout_text).expanduser().resolve() if layout_text else None
    core, tiles_module = _modules()
    selected = core.scan_grid_tiles(data_dir)
    warnings = []
    excluded = []
    if selected is not None:
        if layout_path:
            raise ValueError("Flat mosaic_r<row>_c<col> files already define their grid; clear layout_path / 平铺坐标文件无需布局 JSON，请清空该项。")
        excluded = _flat_companions(data_dir, tiles_module)
        warnings.extend(f"{reason}: {path}" for path, reason in excluded)
        layout, mode = None, "flat-grid"
    else:
        loose_images = [p.name for p in data_dir.iterdir()
                        if p.is_file() and not p.name.startswith(".")
                        and p.suffix.lower().lstrip(".") in tiles_module.IMAGE_EXTS]
        if loose_images:
            raise ValueError("Loose images must use mosaic_r<row>_c<col> names; a finished mosaic is not raw tiles / 平铺原图须有行列坐标，已有拼图不能代替原始瓦片。")
        layout = _read_layout(layout_path, tiles_module) if layout_path else dict(DEFAULT_LAYOUT)
        mode = "provided-layout" if layout_path else "named-columns"
        report = tiles_module.validate_layout(str(data_dir), layout, min_match=1.0)
        if not report["ok"]:
            raise ValueError(tiles_module.format_layout_report(report))
        warnings.extend(report.get("warnings", []))
        selected = tiles_module.scan_dataset(str(data_dir), layout=layout)
    counts = Counter(tile.col for tile in selected)
    if len(counts) < 2 or min(counts.values(), default=0) < 2:
        raise ValueError("Stitching setup requires at least two columns with two tiles each / 拼接设置至少需要两列，每列两张原图。")
    inventory = core.inspect_tiles(selected)
    w, h = inventory["image_size"]
    sizes = [int(record.get("uncompressed_bytes", record.get("source", {}).get("size", 0)))
             for record in inventory["tiles"]]
    # Keep the UI practical; all exact divisors remain available to direct API users.
    divisors = [value for value in range(1, min(w, h) // 8 + 1)
                if w % value == 0 and h % value == 0]
    default_divisor = max(value for value in divisors if value <= 8) if divisors else 1
    metadata = importlib.import_module("tools.stitch_setup.acquisition_metadata")
    acquisition, acquisition_warnings = metadata.inspect_acquisition(data_dir, inventory, mode)
    warnings.extend(acquisition_warnings)
    fingerprint = _fingerprint(inventory, data_dir, layout, str(layout_path) if layout_path else "", acquisition)
    summary = {
        "fingerprint": fingerprint, "data_dir": str(data_dir),
        "layout_path": str(layout_path) if layout_path else "", "layout_mode": mode,
        "image_size": [w, h], "tile_count": inventory["tile_count"],
        "column_count": len(counts), "file_mb_min": min(sizes) / 1_000_000,
        "file_mb_max": max(sizes) / 1_000_000,
        "scale_div_choices": divisors, "default_scale_div": default_divisor,
        "default_output_parent": str(data_dir.parent / "_stitch_results"),
        "excluded_paths": [str(path) for path, _ in excluded],
        "acquisition": acquisition,
    }
    return {"inspection": summary, "inventory": inventory, "layout": layout,
            "warnings": warnings}


def _prepare_profile(body, inspection, inventory):
    core, _ = _modules()
    kind = _text(body, "profile_kind", required=True)
    note = _text(body, "calibration_note", limit=4000, multiline=True)
    if kind == "historical":
        profile = core.historical_profile()
    elif kind == "rescaled":
        confirmations = body.get("confirmations", {})
        if not isinstance(confirmations, dict) or any(
                confirmations.get(key) is not True for key in core.CONFIRMATIONS):
            raise ValueError("Confirm all three same-optics/FOV/stage-step facts explicitly / 请逐项确认光学条件、完整视野和位移步数均相同。")
        profile = core.rescaled_historical_profile(inspection["image_size"],
                                                   **{key: True for key in core.CONFIRMATIONS})
    elif kind == "custom":
        if not note:
            raise ValueError("A calibration note is required for measured vectors / 自定义位移必须填写标定来源说明。")
        profile = core.validate_profile({
            "schema_version": 1, "name": "measured-acquisition",
            "image_size": inspection["image_size"],
            "nominal_vectors": body.get("nominal_vectors"),
            "geometry_source": "measured: " + note,
        })
    else:
        raise ValueError("Choose historical, rescaled or custom profile / 请选择历史、等视野缩放或自定义标定。")
    divisor = body.get("scale_div", inspection["default_scale_div"])
    if type(divisor) is not int:
        raise ValueError("scale_div must be an integer / 缓存缩小倍数须为整数。")
    out_scale = _finite_number(body, "out_scale", 1 / 8)
    full = body.get("full", False)
    if type(full) is not bool:
        raise ValueError("full must be a boolean / 读取原图须为布尔值。")
    resolved = core.resolve_profile(profile, inventory, scale_div=divisor,
                                    out_scale=out_scale, full=full)
    resolved["profile"] = core.validate_profile({**resolved["profile"],
                                                 "scale_div": resolved["scale_div"],
                                                 "out_scale": resolved["out_scale"]})
    return resolved, note


def kimi_status(env=None):
    """Whether run_stitch.py will find Kimi credentials, using the same search as
    kimi_agents.load_env (MOSAIC_ENV, the repository .env, environment variables).
    The key itself is never returned or written anywhere. Owner decision 2026-10-05:
    the student workflow uses Kimi for the stitching judgement steps exactly as the
    original pipeline did, so --no-ai is added only when no credentials are present."""
    kimi = None
    if env is None:
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            kimi = importlib.import_module("kimi_agents")
            env = kimi.load_env()
        except Exception as exc:  # pragma: no cover - a broken client module must not block a local stitch
            return {"configured": False, "mode": "no-ai", "detail": f"kimi_agents unavailable: {exc}"}
    configured = bool(env.get("KIMI_API_KEY"))
    status = {"configured": configured, "mode": "kimi" if configured else "no-ai"}
    if configured:
        status["vision_model"] = env.get("KIMI_VISION_MODEL") or getattr(kimi, "DEFAULT_MODEL", None) or "default"
        status["base_url"] = env.get("KIMI_BASE_URL") or getattr(kimi, "DEFAULT_BASE_URL", None) or "default"
        status["steps"] = ["qc_vote_on_borderline_focus", "conflict_choice_between_duplicates", "seam_inspection"]
    else:
        status["detail"] = "no KIMI_API_KEY in MOSAIC_ENV, the repository .env or the environment; statistics-only fallbacks"
    return status


def _command(argv):
    if os.name == "nt":
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv), "PowerShell"
    return shlex.join(argv), "POSIX shell"


def _write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _layer_input_contract(inspection, inventory, resolved, preflight_text, profile_text):
    """Carry measured input geometry into later layer analysis without deciding
    model compatibility or equating a raw tile with the not-yet-rendered mosaic."""
    acquisition = inspection['acquisition']
    contract = acquisition.get('acquisition_contract')
    return {
        'schema_version': 1,
        'kind': 'layer_input_contract',
        'native_image_size': list(inventory['image_size']),
        'size_source': 'selected_raw_image_headers',
        'source_binding': {
            'inspection_fingerprint': inspection['fingerprint'],
            'input_inventory_fingerprint': inventory['fingerprint'],
            'selected_tile_count': inventory['tile_count'],
            'layout_mode': inspection['layout_mode'],
            'acquisition_source_files': acquisition.get('source_files', []),
            'preflight_file': 'preflight.json',
            'preflight_sha256': hashlib.sha256(preflight_text.encode('utf-8')).hexdigest(),
        },
        'stitch_profile': {
            'file': 'profile.json',
            'name': resolved['profile']['name'],
            'sha256': hashlib.sha256(profile_text.encode('utf-8')).hexdigest(),
            'image_size': list(resolved['profile']['image_size']),
            'scale_div': resolved['scale_div'],
            'out_scale': resolved['out_scale'],
            'full': resolved['full'],
        },
        'acquisition_verification': {
            'metadata_present': acquisition['present'],
            'session_id': acquisition.get('session_id'),
            'verification_status': contract.get('verification_status', 'not_recorded') if isinstance(contract, dict) else 'not_recorded',
            'acquisition_contract': contract,
            'calibration_status': acquisition.get('calibration_status', 'unverified_for_current_mode'),
            'physical_coverage_verified': acquisition.get('physical_coverage_verified', False),
        },
        'inference_geometry_status': 'needs_model_and_scale_check',
        'inference_input_image_size': None,
        'model_selected': False,
        'inference_executed': False,
        'stitch_executed': False,
        'resize_applied_by_handoff': False,
        'note': ('Native dimensions describe the selected raw tiles. Verify the reference acquisition mode, '
                 'applicable checkpoint and actual field of view before layer inference. The selected mosaic '
                 'render scale is recorded separately; no final mosaic size or model compatibility is inferred.'),
    }


def _prepare(body):
    fingerprint = _text(body, "fingerprint", required=True, limit=64)
    if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise ValueError("Inspect raw tiles before preparing / 请先检查原始图像。")
    checked = _inspect(body)
    inspection = checked["inspection"]
    if fingerprint != inspection["fingerprint"]:
        raise ValueError("Inputs, acquisition records or layout changed since inspection; inspect again / 检查后图像、采集记录或布局已改变，请重新检查。")
    resolved, note = _prepare_profile(body, inspection, checked["inventory"])
    data_dir = Path(inspection["data_dir"])
    output_text = _text(body, "output_parent") or inspection["default_output_parent"]
    output_parent = Path(output_text).expanduser().resolve()
    if output_parent == data_dir or data_dir in output_parent.parents:
        raise ValueError("Save the bundle outside the raw tile directory / 输出目录须位于原始图像目录之外。")
    if output_parent.exists() and not output_parent.is_dir():
        raise ValueError("Output parent must be a directory / 输出位置须为目录。")
    output_parent.mkdir(parents=True, exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix="stitch_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_",
                                  dir=str(output_parent)))
    profile_path = bundle / "profile.json"
    preflight_path = bundle / "preflight.json"
    argv_path = bundle / "argv.json"
    instructions_path = bundle / "instructions.txt"
    layer_contract_path = bundle / "layer_input_contract.json"
    ai_mode = _text(body, "ai_mode", limit=16) or "auto"
    if ai_mode not in ("auto", "off"):
        raise ValueError("ai_mode must be 'auto' or 'off' / 只能为 auto 或 off。")
    kimi = kimi_status() if ai_mode == "auto" else {"configured": False, "mode": "no-ai", "detail": "disabled by request (ai_mode=off)"}
    argv = [sys.executable, str(ROOT / "run_stitch.py"), "--data", str(data_dir),
            "--work", str(bundle / "work"), "--out", str(bundle / "mosaic.png"),
            "--stitch-profile", str(profile_path), "--scale-div", str(resolved["scale_div"]),
            "--out-scale", str(resolved["out_scale"]), "--input-preflight", str(preflight_path)]
    if kimi["mode"] == "no-ai":
        argv.append("--no-ai")
    if checked["layout"] is not None:
        argv.extend(["--layout", str(bundle / "layout.json")])
    if resolved["full"]:
        argv.append("--full")
    command, shell = _command(argv)
    warnings = list(checked["warnings"]) + list(resolved["warnings"])
    layer_next_step = ("Layer input geometry is recorded in layer_input_contract.json; verify the reference acquisition mode, "
                       "applicable checkpoint and actual field of view before inference / "
                       "层数输入尺寸已保存至 layer_input_contract.json；参考采集模式、适用权重与实际视野需核对。")
    warnings.append(layer_next_step)
    if kimi["mode"] == "kimi":
        warnings.append("Kimi is configured: the stitch votes on borderline-focus tiles, chooses between duplicate "
                        "candidates and spot-checks seams by sending tile thumbnails to the Moonshot service "
                        f"(vision model {kimi['vision_model']}); every call is cached and logged under work/kimi_cache / "
                        f"已配置 Kimi：拼接会把瓦片缩略图发送给 Moonshot 服务做质检投票、同格位二选一和接缝抽查（视觉模型 {kimi['vision_model']}），"
                        "每次调用都缓存并记录在 work/kimi_cache。")
    else:
        warnings.append("Kimi is not configured: borderline tiles are kept for human review, duplicates are resolved by "
                        "focus score and seams are not spot-checked automatically / "
                        "未配置 Kimi：边缘瓦片全部保留待人工复核，同格位候选按对焦分数保留，接缝不自动抽查。")
    preflight = {
        "schema_version": 1, "status": "profile_prepared", "stitch_executed": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inspection": inspection, "input_inventory": checked["inventory"],
        "layout": checked["layout"], "resolved": resolved,
        "calibration_note": note, "warnings": warnings, "argv": argv, "kimi": kimi,
    }
    instructions = (
        "拼接设置已准备；尚未执行拼接 / Stitch profile prepared; stitching has NOT run.\n\n"
        "1. 原始数据保持原位；此目录保存图像清单、明确选择的标定和运行参数。\n"
        "   Raw inputs stay in place. This bundle records their inventory, selected calibration and arguments.\n"
        "2. 确认本机 Python 环境已安装项目依赖，再检查并运行下列命令。\n"
        "   Verify the Python environment and project dependencies, then review and run the command below.\n"
        + ("3. 命令在本地拼接，并调用 Kimi 看图做质检投票、同格位二选一和接缝抽查（使用 .env 里的 KIMI_API_KEY，瓦片缩略图会发送到 Moonshot 服务）；不会启动扫描。\n"
           "   The command stitches locally and calls Kimi for QC votes, duplicate choice and seam inspection (KIMI_API_KEY from .env; tile thumbnails are sent to the Moonshot service); it does not start acquisition.\n"
           if kimi["mode"] == "kimi" else
           "3. 命令仅在本地拼接，包含 --no-ai；不会启动扫描或请求外部模型。\n"
           "   The command stitches locally with --no-ai; it does not start acquisition or call an external model.\n")
        + 
        "4. 成图后仍须检查真实相邻图像、重叠、断开区域和配准报告；产出文件不等于拼接验收。\n"
        "   Verify adjacent raw tiles, overlap, disconnected areas and registration reports before accepting a mosaic.\n\n"
        f"Shell / 参数格式: {shell}\n{command}\n\n"
        "argv.json 是精确的参数数组；没有生成可执行脚本。\n"
        "argv.json is the exact argument array. No executable script was generated.\n"
        "Windows: 请在 PowerShell 中运行，勿粘贴到 CMD / Run the command in PowerShell, not CMD.\n\n"
        f"标定说明 / Calibration note: {note or '(not supplied / 未填写)'}\n"
        + ("\n注意 / Notes:\n" + "\n".join("- " + warning for warning in warnings) + "\n" if warnings else "")
    )
    if inspection["acquisition"]["present"]:
        _write_new(bundle / "acquisition_metadata.json", _json(inspection["acquisition"]))
    profile_text = _json(resolved["profile"])
    preflight_text = _json(preflight)
    layer_contract = _layer_input_contract(inspection, checked["inventory"], resolved, preflight_text, profile_text)
    _write_new(profile_path, profile_text)
    if checked["layout"] is not None:
        _write_new(bundle / "layout.json", _json(checked["layout"]))
    _write_new(argv_path, _json(argv))
    _write_new(instructions_path, instructions)
    _write_new(preflight_path, preflight_text)
    _write_new(layer_contract_path, _json(layer_contract))
    return {"ok": True, "status": "profile_prepared", "stitch_executed": False,
            "bundle_dir": str(bundle), "profile_path": str(profile_path),
            "preflight_path": str(preflight_path), "argv_path": str(argv_path),
            "layer_input_contract_path": str(layer_contract_path), "layer_next_step": layer_next_step,
            "instructions_path": str(instructions_path), "command": command,
            "command_shell": shell, "profile": resolved["profile"],
            "inspection": inspection, "warnings": warnings}


def dispatch(action, body):
    """Return JSON-safe summaries; expected input failures raise ValueError/OSError."""
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object / 请求须为 JSON 对象。")
    if action == "inspect":
        checked = _inspect(body)
        return {"ok": True, "status": "inspected", "inspection": checked["inspection"],
                "warnings": checked["warnings"]}
    if action == "prepare":
        return _prepare(body)
    raise ValueError("Unknown stitching setup action / 未知拼接设置操作。")
