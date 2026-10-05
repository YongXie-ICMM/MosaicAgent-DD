"""Portable student workflow tests with synthetic pixels and a fake checkpoint."""
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import student_demo as demo


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "Student package with spaces"
    (root / "data/demo").mkdir(parents=True)
    (root / "weights").mkdir()
    pixels = io.BytesIO()
    Image.new("RGB", (32, 24), (130, 70, 120)).save(pixels, format="PNG")
    raw = pixels.getvalue()
    archive = root / "data/demo/source_images.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("images/camera_01.png", raw)
    weights = root / "weights/model_0409_all.pth"
    weights.write_bytes(b"synthetic checkpoint; never deserialize")
    manifest = {"schema_version": 1,
                "archive": {"path": "data/demo/source_images.zip", "sha256": demo.digest(archive)},
                "weights": {"path": "weights/model_0409_all.pth", "sha256": demo.digest(weights)},
                "samples": [{"sample_id": "camera_01", "member": "images/camera_01.png",
                             "sha256": hashlib.sha256(raw).hexdigest(), "width": 32, "height": 24}]}
    (root / "data/demo/assets.json").write_text(json.dumps(manifest))
    return root


def update_manifest(root, function):
    path = root / "data/demo/assets.json"
    data = json.loads(path.read_text())
    function(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("relative", ["weights/model_0409_all.pth", "data/demo/source_images.zip"])
def test_asset_hash_failure_precedes_model_import_and_output_creation(bundle, monkeypatch, relative):
    (bundle / relative).write_bytes(b"changed")
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    with pytest.raises(demo.DemoError, match="SHA-256 mismatch"):
        demo.run_demo(bundle)
    assert not (bundle / "outputs").exists()


def test_missing_assets_gives_student_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not load model"))
    with pytest.raises(demo.DemoError, match="complete student package"):
        demo.run_demo(tmp_path)


@pytest.mark.parametrize("member", ["../image.png", "/image.png", "C:/image.png", "nested\\image.png"])
def test_unsafe_zip_members_are_rejected_before_model(bundle, monkeypatch, member):
    path = bundle / "data/demo/source_images.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"pixels")
    update_manifest(bundle, lambda data: data["archive"].update(sha256=demo.digest(path)))
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not load model"))
    with pytest.raises(demo.DemoError, match="path"):
        demo.run_demo(bundle)
    assert not (bundle.parent / "image.png").exists()


def test_zip_and_manifest_must_match_exactly(bundle):
    path = bundle / "data/demo/source_images.zip"
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unlisted.png", b"extra")
    update_manifest(bundle, lambda data: data["archive"].update(sha256=demo.digest(path)))
    with pytest.raises(demo.DemoError, match="manifest differ"):
        demo.validate_assets(bundle)


def test_per_image_hash_and_dimensions_checked_before_model(bundle, monkeypatch):
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not load model"))
    update_manifest(bundle, lambda data: data["samples"][0].update(sha256="0" * 64))
    with pytest.raises(demo.DemoError, match="Image checksum"):
        demo.run_demo(bundle)
    with zipfile.ZipFile(bundle / "data/demo/source_images.zip") as archive:
        sha = hashlib.sha256(archive.read("images/camera_01.png")).hexdigest()
    update_manifest(bundle, lambda data: data["samples"][0].update(sha256=sha, width=31))
    with pytest.raises(demo.DemoError, match="dimensions differ"):
        demo.run_demo(bundle)


@dataclass
class FakeResult:
    ok: bool = True
    data: dict = field(default_factory=dict)
    confidence: float = 1.0
    evidence: str = "synthetic test result"
    escalate: bool = False
    escalate_reason: str = ""


def test_demo_reuses_stages_retains_raw_and_writes_workbench_pairing(bundle, monkeypatch):
    inputs, contexts = [], []
    before = {name: demo.digest(bundle / name) for name in
              ("data/demo/source_images.zip", "weights/model_0409_all.pth")}

    def segment(context, image):
        inputs.append(image)
        assert Image.open(image).size == (32, 24)
        contexts.append(context)
        return FakeResult(data={"mask": str(context["work"] / "mask.npy"),
                                "valid_mask": str(context["work"] / "valid.npy")},
                          escalate=True, escalate_reason="Review synthetic example")

    def stats(context, mask, valid, exclusion_path=None):
        assert exclusion_path is None
        assert mask.endswith("mask.npy") and valid.endswith("valid.npy")
        out = context["work"] / "04_stats/map.png"
        out.parent.mkdir()
        Image.new("RGB", (32, 24), (0, 128, 0)).save(out)
        return FakeResult(data={"layer_map_preview": str(out), "ratios_pct": {"1L": 100.0}})

    monkeypatch.setattr(demo, "load_stages", lambda: (segment, stats))
    first = demo.run_demo(bundle)
    second = demo.run_demo(bundle)
    assert first != second
    assert all(not path.exists() for path in inputs)
    assert not (bundle / "data/demo/images").exists()
    assert before == {name: demo.digest(bundle / name) for name in before}
    for context in contexts:
        assert (context["device"], context["tile"], context["overlap"]) == ("cpu", 512, 64)
        assert context["filler"] == "none" and context["border_trim_px"] == 0
        assert "inference_geometry" not in context
    result = json.loads((second / "results/run_manifest.json").read_text())
    assert result["status"] == "completed_with_review_flags"
    assert result["remote_model_calls"] == 0
    assert result["region_exclusion_applied"] is False
    row = result["samples"][0]
    assert row["archive"] == "data/demo/source_images.zip"
    assert row["member"] == "images/camera_01.png"
    assert row["outputs"]["mask_color.png"]["sha256"] == demo.digest(second / "results/camera_01/mask_color.png")
    assert json.loads((bundle / "outputs/demo/latest.json").read_text())["run"] == second.relative_to(bundle).as_posix()

    # Moving the whole package regenerates absolute configuration without edits.
    moved = bundle.parent / "Moved package"
    shutil.copytree(bundle, moved)
    config = demo.prepare_workbench_config(moved)
    data = json.loads(config.read_text())
    assert data["project_root"] == str(moved.resolve())
    assert data["scan_repo"] == str(moved / "acquisition")
    assert data["inference_results"] == str(moved / second.relative_to(bundle))
    from tools.analysis_workbench.server import Workbench
    monkeypatch.chdir(moved)
    workbench = Workbench(config)
    workbench.collect()
    layer_artifacts = [record["public"] for record in workbench.artifacts.values()
                       if record["public"]["tool"] == "layers"]
    assert {record["role"] for record in layer_artifacts} >= {"original", "prediction"}


def test_open_workbench_generates_portable_config_and_no_hardware_arguments(bundle, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "available_port", lambda: 8899)
    monkeypatch.setattr(demo.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    assert demo.open_workbench(bundle, no_browser=True) == 0
    command, kwargs = calls[0]
    assert kwargs["cwd"] == bundle.resolve()
    assert command[-1] == "--no-browser"
    assert command[command.index("--port") + 1] == "8899"
    config = Path(command[command.index("--config") + 1])
    assert json.loads(config.read_text())["project_root"] == str(bundle.resolve())
    assert "--sim" not in command


def test_student_launchers_are_relative_and_use_local_venv():
    for prefix, action in (("02_run_layer_demo", "run"), ("03_open_workbench", "workbench")):
        bat = (ROOT / (prefix + ".bat")).read_text()
        shell = (ROOT / (prefix + ".command")).read_text()
        assert 'cd /d "%~dp0"' in bat and '.venv\\Scripts\\python.exe' in bat
        assert '.venv/bin/python' in shell
        assert "student_demo.py " + action in bat.replace("\\", "/")
        assert "student_demo.py " + action in shell
        subprocess.run(["sh", "-n", str(ROOT / (prefix + ".command"))], check=True)
    assert "requirements-analysis.txt" in (ROOT / "01_install.bat").read_text()
    subprocess.run(["sh", "-n", str(ROOT / "01_install.command")], check=True)


def passing_fake_stages():
    def segment(context, image):
        return FakeResult(data={"mask": "mask.npy", "valid_mask": "valid.npy"})
    def stats(context, mask, valid, exclusion_path=None):
        path = context["work"] / "preview.png"
        Image.new("RGB", (32, 24), (0, 128, 0)).save(path)
        return FakeResult(data={"layer_map_preview": str(path)})
    return segment, stats


def test_asset_review_flag_survives_passing_stage_checks(bundle, monkeypatch, capsys):
    update_manifest(bundle, lambda data: data.update(
        review_required=True, review_status="model_applicability_unverified",
        review_note="Apparent substrate was predicted as monolayer; inspect this example."))
    monkeypatch.setattr(demo, "load_stages", passing_fake_stages)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    run = demo.run_demo(bundle)
    result = json.loads((run / "results/run_manifest.json").read_text())
    assert result["review_required"] is True
    assert result["status"] == "completed_with_review_flags"
    assert result["review_status"] == "model_applicability_unverified"
    row = result["samples"][0]
    assert row["segmentation"]["ok"] is True and row["statistics"]["ok"] is True
    assert row["requires_human_review"] is True
    assert row["asset_review_note"] == result["review_note"]
    assert "do not use fractions scientifically" in capsys.readouterr().out
    assert demo.os.environ["MPLCONFIGDIR"] == str(run / ".cache/matplotlib")


def test_manifest_selects_and_records_another_hash_verified_checkpoint(bundle, monkeypatch):
    original = bundle / "weights/model_0409_all.pth"
    replacement = bundle / "weights/model_0815_trial.pth"
    original.rename(replacement)
    update_manifest(bundle, lambda data: data["weights"].update(path="weights/model_0815_trial.pth"))
    segment, stats = passing_fake_stages()
    seen = []
    def record_weights(context, image):
        seen.append(context["weights"])
        return segment(context, image)
    monkeypatch.setattr(demo, "load_stages", lambda: (record_weights, stats))
    run = demo.run_demo(bundle)
    result = json.loads((run / "results/run_manifest.json").read_text())
    assert seen == [str(replacement)]
    assert result["weights"] == {"path": "weights/model_0815_trial.pth", "sha256": demo.digest(replacement)}
    assert not original.exists()


def comparison_geometry(native=(32, 24), reference=(64, 48)):
    from flakepipeline.inference_geometry import plan_inference_geometry
    return plan_inference_geometry(
        native, reference, same_physical_fov_confirmed=True,
        model_tile=demo.CONFIG["tile"], model_overlap=demo.CONFIG["overlap"],
        provenance={"scope": "provisional_same_fov_hypothesis_for_comparison",
                    "fov_confirmation_source": "Assumed only for a synthetic software check"})


def test_geometry_reaches_stage_and_forces_review_even_when_every_stage_passes(bundle, monkeypatch):
    geometry = comparison_geometry()
    contexts = []
    segment, stats = passing_fake_stages()

    def record(context, image):
        contexts.append(context)
        return segment(context, image)

    monkeypatch.setattr(demo, "load_stages", lambda: (record, stats))
    path = demo.run_demo(bundle, inference_geometry=geometry)
    manifest = json.loads((path / "results/run_manifest.json").read_text())
    assert contexts[0]["inference_geometry"] == geometry
    assert (contexts[0]["tile"], contexts[0]["overlap"]) == (512, 64)
    assert manifest["configuration"]["inference_geometry"] == geometry
    assert manifest["inference_geometry"] == geometry
    assert manifest["comparison_only"] is True
    assert manifest["review_required"] is True
    assert manifest["status"] == "completed_with_review_flags"
    assert manifest["physical_calibration"] == "not_applied"
    row = manifest["samples"][0]
    assert row["segmentation"]["ok"] and row["statistics"]["ok"]
    assert row["requires_human_review"] is True
    assert row["source_image_size"] == [32, 24]
    assert "not established" in row["asset_review_note"]
    assert manifest["remote_model_calls"] == 0 and manifest["repairs_applied"] is False


def test_ready_plan_with_wrong_native_image_dimensions_is_rejected_before_model(bundle, monkeypatch):
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    geometry = comparison_geometry((64, 48), (128, 96))
    assert geometry["status"] == "ready"
    with pytest.raises(demo.DemoError, match="Actual images do not match"):
        demo.run_demo(bundle, inference_geometry=geometry)
    assert not (bundle / "outputs").exists()


def test_modified_derived_geometry_is_rejected_before_model(bundle, monkeypatch):
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    geometry = comparison_geometry()
    geometry["native_tile"] = 128
    with pytest.raises(demo.DemoError, match="Geometry is not ready"):
        demo.run_demo(bundle, inference_geometry=geometry)
    assert not (bundle / "outputs").exists()


def test_scale_comparison_preserves_two_runs_and_never_selects_measurement_result(bundle, monkeypatch):
    update_manifest(bundle, lambda data: data.update(inference_reference_capture_size=[64, 48]))
    contexts = []
    segment, stats = passing_fake_stages()

    def record(context, image):
        contexts.append(context)
        return segment(context, image)

    monkeypatch.setattr(demo, "load_stages", lambda: (record, stats))
    source_before = demo.digest(bundle / "data/demo/source_images.zip")
    weights_before = demo.digest(bundle / "weights/model_0409_all.pth")
    record_path = demo.compare_scale(bundle)
    record = json.loads(record_path.read_text())
    assert record["status"] == "comparison_only_requires_review"
    assert record["selected_for_measurement"] is False
    assert record["physical_stage_motion"] == "unchanged"
    assert record["direct_run"] != record["adapted_run"]
    assert "inference_geometry" not in contexts[0]
    assert contexts[1]["inference_geometry"]["native_tile"] == 256
    assert contexts[1]["inference_geometry"]["provenance"]["scope"] == "provisional_same_fov_hypothesis_for_comparison"
    manifests = [json.loads((bundle / record[key] / "results/run_manifest.json").read_text())
                 for key in ("direct_run", "adapted_run")]
    assert all(m["comparison_only"] is True and m["review_required"] is True for m in manifests)
    assert manifests[0]["weights"] == manifests[1]["weights"]
    assert demo.digest(bundle / "data/demo/source_images.zip") == source_before
    assert demo.digest(bundle / "weights/model_0409_all.pth") == weights_before
    assert json.loads((bundle / "outputs/demo/latest.json").read_text())["run"] == record["adapted_run"]


def test_compare_scale_cli_routes_to_comparison_without_hardware(bundle, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "compare_scale", lambda: calls.append("comparison"))
    monkeypatch.setattr(demo, "run_demo", lambda: pytest.fail("CLI must enter the comparison function"))
    monkeypatch.setattr(demo, "open_workbench", lambda *args, **kwargs: pytest.fail("must not open hardware or UI"))
    assert demo.main(["compare-scale"]) == 0
    assert calls == ["comparison"]


@pytest.mark.parametrize("geometry", [{}, [], {"observed_native_size": [32, 24]}])
def test_incomplete_geometry_gives_actionable_error_before_model(bundle, monkeypatch, geometry):
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    with pytest.raises(demo.DemoError, match="Incomplete geometry"):
        demo.run_demo(bundle, inference_geometry=geometry)
    assert not (bundle / "outputs").exists()


def record_confirmed_capture(bundle, **changes):
    capture = {"same_physical_fov_confirmed": True,
               "confirmation_source": "Operator confirms that only resolution changed; actual field of view is unchanged."}
    capture.update(changes)
    update_manifest(bundle, lambda data: data.update(
        capture_geometry=capture, inference_reference_capture_size=[64, 48]))
    return capture


def test_recorded_confirmation_automatically_selects_native_256_32_and_retains_source(bundle, monkeypatch):
    capture = record_confirmed_capture(bundle)
    contexts = []
    segment, stats = passing_fake_stages()

    def record(context, image):
        contexts.append(context)
        return segment(context, image)

    monkeypatch.setattr(demo, "load_stages", lambda: (record, stats))
    path = demo.run_demo(bundle)
    result = json.loads((path / "results/run_manifest.json").read_text())
    geometry = contexts[0]["inference_geometry"]
    assert (geometry["native_tile"], geometry["native_overlap"]) == (256, 32)
    assert (geometry["model_tile"], geometry["model_overlap"]) == (512, 64)
    assert geometry["model_resize_scale"] == 2.0
    assert geometry["provenance"]["scope"] == "operator_confirmed_capture_mode_scale_mapping"
    assert geometry["provenance"]["fov_confirmation_source"] == capture["confirmation_source"]
    assert result["capture_geometry_confirmation"] == capture
    assert result["inference_geometry"] == geometry
    assert geometry["physical_calibration_established"] is False
    assert geometry["segmentation_accuracy_established"] is False
    assert not result.get("comparison_only", False)


@pytest.mark.parametrize("capture", [None, {}, {"same_physical_fov_confirmed": False, "confirmation_source": "Unknown"},
    {"same_physical_fov_confirmed": 1, "confirmation_source": "Not a boolean"},
    {"same_physical_fov_confirmed": True, "confirmation_source": " "}])
def test_present_but_unconfirmed_capture_blocks_before_model(bundle, monkeypatch, capture):
    update_manifest(bundle, lambda data: data.update(
        capture_geometry=capture, inference_reference_capture_size=[64, 48]))
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    with pytest.raises(demo.DemoError, match="Confirm the recorded field of view"):
        demo.run_demo(bundle)
    assert not (bundle / "outputs").exists()


def test_confirmed_capture_cannot_override_inconsistent_reference_dimensions(bundle, monkeypatch):
    record_confirmed_capture(bundle)
    update_manifest(bundle, lambda data: data.update(inference_reference_capture_size=[64, 49]))
    monkeypatch.setattr(demo, "load_stages", lambda: pytest.fail("must not import model code"))
    with pytest.raises(demo.DemoError, match="Geometry is not ready"):
        demo.run_demo(bundle)
    assert not (bundle / "outputs").exists()


def test_confirmed_capture_comparison_keeps_native_baseline_and_flags_both_runs(bundle, monkeypatch):
    capture = record_confirmed_capture(bundle)
    contexts = []
    segment, stats = passing_fake_stages()

    def record(context, image):
        contexts.append(context)
        return segment(context, image)

    monkeypatch.setattr(demo, "load_stages", lambda: (record, stats))
    comparison = json.loads(demo.compare_scale(bundle).read_text())
    assert "inference_geometry" not in contexts[0]
    assert contexts[1]["inference_geometry"]["native_tile"] == 256
    assert comparison["geometry"]["provenance"]["scope"] == "confirmed_fov_scale_comparison"
    assert comparison["geometry"]["provenance"]["fov_confirmation_source"] == capture["confirmation_source"]
    assert comparison["selected_for_measurement"] is False
    for key in ("direct_run", "adapted_run"):
        result = json.loads((bundle / comparison[key] / "results/run_manifest.json").read_text())
        assert result["capture_geometry_confirmation"] == capture
        assert result["comparison_only"] is True and result["review_required"] is True
