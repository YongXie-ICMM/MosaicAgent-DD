"""Cache dependencies and byte integrity, using local synthetic inputs only.

No segmentation inference, model access, or network calls are required.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import orchestrator as O  # noqa: E402
import review as RV  # noqa: E402
from agents import Result  # noqa: E402
from test_orchestrator_smoke import cfg, wire  # noqa: E402,F401


def _spec(orchestrator, stage):
    return next(spec for spec in orchestrator._specs() if spec["name"] == stage)


def _record(orchestrator, tmp_path, stage, key, result, outputs):
    manifest = {"stages": {}}
    orchestrator._record(manifest, tmp_path / "manifest.json", stage, key, result,
                         outputs, [], O.Decision("accept"))
    return manifest


@pytest.fixture
def windows_default_encoding(monkeypatch):
    """Emulate legacy Windows text defaults without changing UTF-8 explicitly
    requested by the application. This catches omissions on every platform.
    """
    read_text, write_text = Path.read_text, Path.write_text

    def read(path, encoding=None, errors=None, **kwargs):
        return read_text(path, encoding=encoding or "cp1252", errors=errors, **kwargs)

    def write(path, data, encoding=None, errors=None, **kwargs):
        return write_text(path, data, encoding=encoding or "cp1252", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)


def test_chinese_pipeline_and_review_roundtrip_with_cp1252_default(cfg, monkeypatch,
                                                                  windows_default_encoding):
    wire(monkeypatch)
    samples = cfg.pop("samples")
    samples[0]["name"] = "中文样品一"
    cfg["exclusion_rule"] = "盐残留区域须人工复核"
    result = O.Orchestrator(cfg).run(samples)
    manifest_path = Path(result["samples"][0]["manifest"])
    assert "中文样品一".encode("utf-8") in manifest_path.read_bytes()
    assert RV.load_json(manifest_path, {})["sample"] == "中文样品一"
    replay = O.Orchestrator(cfg).run(samples)
    assert all(action == "cache" for action in replay["samples"][0]["actions"].values())

    work = Path(cfg["work_root"])
    RV.push_queue(work, {"sample": "中文样品一", "stage": "audit", "reason": "边界需要复核"})
    assert RV.resolve_entry(work, "中文样品一", "audit", "审核员乙", "已检查原图，仍保留不确定区") == 1
    RV.push_queue(work, {"sample": "中文样品一", "stage": "stats", "reason": "分母需复查"})
    RV.clear_queue(work, "中文样品一", "stats")
    queue = RV.load_json(RV.queue_path(work), [])
    assert len(queue) == 1
    assert queue[0]["resolve_note"] == "已检查原图，仍保留不确定区"
    RV._cmd_decide(work, SimpleNamespace(sample="中文样品一", rank=1, keep=True, exclude=False,
                                        by="审核员甲", note="有清楚晶棱，保留该区域",
                                        category="multilayer_crystals"))
    verdict = RV.load_json(RV.verdicts_path(work), [])[0]
    assert verdict["decided_by"] == "审核员甲"
    assert verdict["note"] == "有清楚晶棱，保留该区域"
    assert "审核员甲".encode("utf-8") in RV.verdicts_path(work).read_bytes()


@pytest.mark.parametrize("changed_input", ["mosaic", "valid_mask"])
def test_audit_does_not_reuse_changed_image_or_valid_support_with_same_mask(tmp_path, changed_input):
    ctx = {"work": tmp_path, "sample": "S", "no_ai": True}
    state = {key: str(tmp_path / f"{key}.dat") for key in ("mosaic", "mask", "valid_mask")}
    for key, path in state.items():
        Path(path).write_bytes(f"synthetic {key} v1".encode())
    output = tmp_path / "S_audit.json"
    output.write_text('{"synthetic":true}', encoding="utf-8")
    orchestrator = O.Orchestrator({"work_root": str(tmp_path)})
    spec = _spec(orchestrator, "audit")
    key = spec["key"](ctx, state)
    manifest = _record(orchestrator, tmp_path, "audit", key, Result(ok=True), [output])
    assert orchestrator._cached(manifest, "audit", spec["key"](ctx, state)) is not None

    mask_before = Path(state["mask"]).read_bytes()
    Path(state[changed_input]).write_bytes(f"synthetic {changed_input} v2".encode())
    assert Path(state["mask"]).read_bytes() == mask_before
    assert orchestrator._cached(manifest, "audit", spec["key"](ctx, state)) is None


def test_stats_valid_support_change_invalidates_cache_and_changes_real_denominator(tmp_path, monkeypatch,
                                                                                  windows_default_encoding):
    # Exercise the actual numeric statistics stage. Disable only its optional
    # matplotlib rendering to keep this test independent of plotting packages.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    ctx = {"work": tmp_path, "sample": "统计样品", "map_preview_div": 1}
    state = {"mask": str(tmp_path / "classes.npy"), "valid_mask": str(tmp_path / "valid.npy")}
    labels = np.array([[1, 2], [3, 2]], dtype=np.uint8)
    np.save(state["mask"], labels)
    np.save(state["valid_mask"], np.ones((2, 2), dtype=bool))
    orchestrator = O.Orchestrator({"work_root": str(tmp_path)})
    spec = _spec(orchestrator, "stats")
    result = spec["run"](ctx, state)
    assert result.data["counted_px"] == 4
    assert result.data["ratios_pct"]["2L"] == 25
    manifest = _record(orchestrator, tmp_path, "stats", spec["key"](ctx, state),
                       result, spec["outputs"](ctx, result.data))
    assert orchestrator._cached(manifest, "stats", spec["key"](ctx, state)) is not None

    # Semantic labels and exclusion mask are unchanged; only scanned support
    # changes. Reusing the previous result would incorrectly retain N=4.
    np.save(state["valid_mask"], np.array([[True, False], [True, False]]))
    assert np.array_equal(np.load(state["mask"]), labels)
    assert orchestrator._cached(manifest, "stats", spec["key"](ctx, state)) is None
    updated = spec["run"](ctx, state)
    assert updated.data["counted_px"] == 2
    assert updated.data["ratios_pct"]["2L"] == 50


@pytest.mark.parametrize("stage", O.Orchestrator.STAGES)
def test_same_size_output_damage_is_not_reused_even_with_original_mtime(cfg, monkeypatch, stage):
    wire(monkeypatch)
    samples = cfg.pop("samples")
    first = O.Orchestrator(cfg).run(samples)
    path = Path(first["samples"][0]["manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    record = manifest["stages"][stage]
    output = Path(record["outputs"][0])
    original = output.read_bytes()
    stat = output.stat()
    assert record["output_sha256"][str(output)] == hashlib.sha256(original).hexdigest()
    # Existence, size, and modification time stay unchanged; hash must catch it.
    damaged = bytes([original[0] ^ 1]) + original[1:]
    output.write_bytes(damaged)
    os.utime(output, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    next_run = O.Orchestrator(cfg).run(samples)
    assert next_run["samples"][0]["actions"][stage] == "run"
    assert output.read_bytes() == original
    second_manifest = json.loads(path.read_text(encoding="utf-8"))
    assert second_manifest["stages"][stage]["output_sha256"][str(output)] == hashlib.sha256(original).hexdigest()


def test_legacy_manifest_without_output_hash_is_recomputed_once(cfg, monkeypatch):
    wire(monkeypatch)
    samples = cfg.pop("samples")
    first = O.Orchestrator(cfg).run(samples)
    path = Path(first["samples"][0]["manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["stages"]["audit"]["output_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    second = O.Orchestrator(cfg).run(samples)
    assert second["samples"][0]["actions"]["audit"] == "run"
    assert all(action == "cache" for stage, action in second["samples"][0]["actions"].items() if stage != "audit")
    third = O.Orchestrator(cfg).run(samples)
    assert all(action == "cache" for action in third["samples"][0]["actions"].values())


@pytest.mark.parametrize("case", ["missing_output", "directory", "missing_hash", "null_hash", "empty_outputs"])
def test_incomplete_or_nonfile_output_never_produces_cache_hit(tmp_path, case):
    orchestrator = O.Orchestrator({"work_root": str(tmp_path)})
    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    key = {"synthetic_input": "unchanged"}
    manifest = _record(orchestrator, tmp_path, "audit", key, Result(ok=True), [output])
    record = manifest["stages"]["audit"]
    if case == "missing_output":
        output.unlink()
    elif case == "directory":
        output.unlink()
        output.mkdir()
    elif case == "missing_hash":
        record["output_sha256"] = {}
    elif case == "null_hash":
        record["output_sha256"][str(output)] = None
    elif case == "empty_outputs":
        record["outputs"] = []
        record["output_sha256"] = {}
    assert orchestrator._cached(manifest, "audit", key) is None


def test_unreadable_output_keeps_failed_record_without_enabling_cache(tmp_path, monkeypatch):
    orchestrator = O.Orchestrator({"work_root": str(tmp_path)})
    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    key = {"synthetic_input": "unchanged"}

    def unreadable(*args, **kwargs):
        raise PermissionError("synthetic unreadable output")

    monkeypatch.setattr(O, "_hash_file", unreadable)
    manifest = _record(orchestrator, tmp_path, "audit", key,
                       Result(ok=False, escalate=True), [output])
    assert manifest["stages"]["audit"]["output_sha256"][str(output)] is None
    assert orchestrator._cached(manifest, "audit", key) is None
