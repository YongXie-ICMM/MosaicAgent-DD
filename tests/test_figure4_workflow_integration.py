"""Synthetic native-mask integration through the existing CLI and workbench plan.

No checkpoint, inference, instrument, external model or experimental data used.
"""
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from tools.analysis_workbench.workflow import DDWorkflow
from tools.analysis_workbench.workflow_analysis import prepare_spatial, validate_plan_result
from tools.analysis_workbench.spatial_review import prepare_review, confirm_review
from flakepipeline.spatial_maps.stats import sha

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    native = np.zeros((400, 500), dtype=np.uint8)
    native[140:260, 180:320] = 3  # Synthetic central aggregate candidate.
    native[160:170, 200:210] = 2  # Exclusions must remove every class inside.
    native[190:200, 240:250] = 1
    native[20:70, 30:80] = 2
    native[300:320, 400:430] = 1
    native[300:315, 100:120] = 3  # Real-looking off-centre TL remains counted.
    valid = np.ones_like(native, dtype=bool); valid[:4] = False
    base = np.zeros_like(valid); base[20:22, 30:32] = True
    paths = {}
    for key, array in (("mask", native), ("valid", valid), ("base", base)):
        paths[key] = tmp_path / (key + ".npy"); np.save(paths[key], array)
    paths["mosaic"] = tmp_path / "matched photo.png"
    Image.new("RGB", (500, 400), (230, 187, 189)).save(paths["mosaic"])
    return paths


def execute(plan):
    run = subprocess.run(plan["argv"], cwd=REPO, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    result = validate_plan_result(plan)
    manifest = json.loads(Path(result["output_path"]).read_text())
    assert manifest["sources"] == plan["source_binding"]["sources"]
    assert {layer: manifest["global"]["counts_px"][layer] for layer in ("1L", "2L", "TL")} == {
        "1L": 2496, "2L": 600, "TL": 300}
    assert manifest["global"]["crystal_px"] == 3396
    assert manifest["global"]["non_scan_px"] == 2000
    assert result["review_required"] is True
    for suffix in ("_Figure4.png", "_Figure4.svg"):
        assert any(Path(path).name.endswith(suffix) for path in result["artifacts"])
    return result, manifest


def test_real_cli_auto_draft_plan_sources_artifacts_and_failure_on_tamper(inputs, tmp_path):
    before = {key: sha(path) for key, path in inputs.items()}
    plan = prepare_spatial(inputs["mask"], inputs["valid"], inputs["base"], "synthetic",
                           tmp_path / "new auto draft", REPO, mosaic_path=inputs["mosaic"],
                           auto_central_exclusion=True)
    # The existing fixed-command scheduler must accept both new CLI flags.
    ticket = DDWorkflow(tmp_path / "Memory", repo=REPO).register_plan(plan)
    assert ticket["kind"] == "spatial"
    assert "--mosaic" in plan["argv"] and "--auto-central-exclusion" in plan["argv"]
    result, manifest = execute(plan)
    assert manifest["automatic_central_exclusion"]["status"] == "selected"
    assert manifest["human_exclusion_confirmation_required"] is True
    assert manifest["exclusion_review_status"] == "automatic_proposal_unconfirmed"
    effective = Path(result["output_path"]).parent / manifest["automatic_central_exclusion"]["effective_exclusion_mask"]
    assert str(effective) in result["artifacts"]
    assert np.load(effective)[20:22, 30:32].all()
    assert {key: sha(path) for key, path in inputs.items()} == before
    figure = next(Path(path) for path in result["artifacts"] if path.endswith("_Figure4.svg"))
    figure.write_text(figure.read_text() + "\n<!-- synthetic tamper -->\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_plan_result(plan)


def test_human_boxes_flow_through_confirmation_cli_and_validate_without_reproposal(inputs, tmp_path):
    review = prepare_review(inputs["mask"], inputs["valid"], inputs["base"], inputs["mosaic"],
                            tmp_path / "Memory" / "reviews", automatic=True)
    original_hashes = {key: sha(path) for key, path in inputs.items()}
    proposal_hash = sha(review["proposal_path"])
    edited = [{"x1": 175, "y1": 135, "x2": 325, "y2": 265}]
    assert edited != review["rectangles"]
    confirmed = confirm_review(review["review_dir"], edited, tmp_path / "confirmation")
    plan = prepare_spatial(inputs["mask"], inputs["valid"], confirmed["exclusion_path"], "synthetic",
                           tmp_path / "new confirmed output", REPO, mosaic_path=inputs["mosaic"],
                           auto_central_exclusion=False)
    assert "--auto-central-exclusion" not in plan["argv"]
    assert "base_exclusions" in plan["source_binding"]["sources"]
    _, manifest = execute(plan)
    assert manifest["rectangles"] == edited
    assert manifest["automatic_central_exclusion"] is None
    assert manifest["human_exclusion_review"]["human_confirmed"] is True
    assert manifest["human_exclusion_confirmation_required"] is False
    assert manifest["exclusion_review_status"] == "human_confirmed"
    assert sha(review["proposal_path"]) == proposal_hash
    assert {key: sha(path) for key, path in inputs.items()} == original_hashes
