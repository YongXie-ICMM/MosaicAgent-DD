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
