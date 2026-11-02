"""Offline advisory tests: synthetic ROI pixels and mocked Kimi, never network."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.analysis_workbench import spatial_review as R
from tools.analysis_workbench import spatial_review_kimi as K


BOX = {"x1": 40, "y1": 30, "x2": 60, "y2": 50}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network is forbidden in these synthetic tests")
    monkeypatch.setattr("urllib.request.urlopen", blocked)


@pytest.fixture
def review(tmp_path):
    native = np.zeros((80, 100), dtype=np.uint8)
    native[30:50, 40:60] = 3
    native[10:20, 10:20] = 2
    mask = tmp_path / "mask.npy"; np.save(mask, native)
    valid = tmp_path / "valid.npy"; np.save(valid, np.ones(native.shape, dtype=bool))
    mosaic = tmp_path / "mosaic.png"
    Image.new("RGB", (100, 80), (230, 187, 189)).save(mosaic)
    return R.prepare_review(mask, valid, None, mosaic, tmp_path / "Memory" / "reviews", automatic=False)


def response(count=1):
    return {"rectangles": [{"index": index, "inclusion": "uncertain", "box_coverage": "too_tight",
                            "normal_flakes_concern": "possible", "confidence": 0.6,
                            "reason": "Inspect the target boundary and nearby faceted flakes."}
                           for index in range(1, count + 1)],
            "summary": "A human should adjust these boxes before confirming."}


class FakeClient:
    model = "kimi-vision-offline-fixture"

    def __init__(self, *, available=True, reply=None, during=None, failure=False):
        self.available = available
        self.reply = response() if reply is None else reply
        self.during = during
        self.failure = failure
        self.requests = []
        self.usage = SimpleNamespace(calls=0, cached=0, failed=0, truncated=0, prompt_tokens=0, completion_tokens=0)

    def chat(self, text, **kwargs):
        self.requests.append((text, kwargs))
        self.usage.calls += 1
        self.usage.prompt_tokens += 50
        self.usage.completion_tokens += 20
        if self.during:
            self.during()
        if self.failure:
            print("PROVIDER_SECRET_MUST_NOT_REACH_LOGS")
            raise RuntimeError("PROVIDER_SECRET_MUST_NOT_REACH_DISK")
        return self.reply


def install(monkeypatch, client):
    monkeypatch.setattr(K, "_make_client", lambda: client)


def test_advice_sends_only_paired_roi_preserves_proposal_and_never_applies(review, monkeypatch):
    directory = Path(review["review_dir"])
    before = {path: K._sha(path) for path in directory.iterdir()}
    sources = {row["path"]: K._sha(row["path"]) for row in review["proposal"]["sources"].values()}
    client = FakeClient(); install(monkeypatch, client)
    result = K.advise_review(directory, [BOX])
    assert result["status"] == "advisory_ready"
    assert result["external_call_attempted"] is True
    assert result["source_hashes_verified_before_request"] is True
    assert result["source_hashes_verified_after_request"] is True
    assert result["boxes_applied"] is result["statistical_results_changed"] is False
    assert result["human_approval_required"] is result["advisory_only"] is True
    assert result["independent_accuracy_established"] is False
    assert len(client.requests) == 1
    prompt, payload = client.requests[0]
    assert str(directory) not in prompt and "mosaic.png" not in prompt
    assert "do not approve" in payload["system"].lower()
    assert payload["want_json"] is True and len(payload["images"]) == 1
    image = payload["images"][0]
    assert image.dtype == np.uint8 and image.shape == (46, 48, 3)
    assert tuple(image[23, 1]) == (189, 187, 230)  # Optical RGB correctly adapted to existing BGR client.
    assert result["request_scope"]["full_mosaic_requested"] is False
    assert result["request_scope"]["outbound_image_count"] == 1
    assert result["previews"][0]["context_crop_rectangle"] == {"x1": 38, "y1": 28, "x2": 62, "y2": 52}
    assert result["usage"]["calls"] == 1
    assert json.loads(Path(result["advisory_path"]).read_text())["advice"] == response()
    assert all(K._sha(path) == digest for path, digest in before.items())
    assert all(K._sha(path) == digest for path, digest in sources.items())
    assert R.load_review(directory)["status"] == "needs_review"
    assert not list(directory.rglob("*effective_exclusion*"))


def test_no_key_is_actionable_unavailable_without_any_upload(review, monkeypatch):
    client = FakeClient(available=False); install(monkeypatch, client)
    result = K.advise_review(review["review_dir"], [BOX])
    assert result["status"] == "unavailable" and result["advice"] is None
    assert "configure KIMI_API_KEY" in result["reason"]
    assert result["external_call_attempted"] is False
    assert result["request_scope"]["outbound_image_count"] == 0
    assert result["previews"] == [] and not client.requests
    assert Path(result["request_path"]).is_file() and Path(result["advisory_path"]).is_file()


@pytest.mark.parametrize("boxes", [[], [BOX] * 9, [BOX, BOX],
    [{"x1": True, "y1": 30, "x2": 60, "y2": 50}],
    [{"x1": -1, "y1": 30, "x2": 60, "y2": 50}],
    [{"x1": 0, "y1": 0, "x2": 100, "y2": 80}],
    [{"x1": 1, "y1": 1, "x2": 99, "y2": 79}],
])
def test_invalid_or_whole_canvas_boxes_never_construct_client(review, monkeypatch, boxes):
    def forbidden():
        raise AssertionError("Client must not be constructed")
    monkeypatch.setattr(K, "_make_client", forbidden)
    with pytest.raises(ValueError):
        K.advise_review(review["review_dir"], boxes)
    assert not list(Path(review["review_dir"]).glob("kimi_advisory_*"))


def test_stale_source_fails_before_client_or_artifact_creation(review, monkeypatch):
    mosaic = review["proposal"]["sources"]["mosaic"]["path"]
    Image.new("RGB", (100, 80), (1, 2, 3)).save(mosaic)
    monkeypatch.setattr(K, "_make_client", lambda: pytest.fail("Client must not be constructed"))
    with pytest.raises(ValueError, match="Source changed"):
        K.advise_review(review["review_dir"], [BOX])
    assert not list(Path(review["review_dir"]).glob("kimi_advisory_*"))


@pytest.mark.parametrize("damage", ["source", "proposal_with_new_sidecar"])
def test_sources_and_exact_proposal_reverified_after_mock_request(review, monkeypatch, damage):
    def mutate():
        if damage == "source":
            Image.new("RGB", (100, 80), (5, 6, 7)).save(review["proposal"]["sources"]["mosaic"]["path"])
        else:
            path = Path(review["proposal_path"])
            record = json.loads(path.read_text()); record["rectangles"] = [BOX]
            path.write_text(json.dumps(record))
            (path.parent / "proposal.sha256").write_text(K._sha(path))
    client = FakeClient(during=mutate); install(monkeypatch, client)
    result = K.advise_review(review["review_dir"], [BOX])
    assert result["status"] == "unavailable" and result["advice"] is None
    assert result["external_call_attempted"] is True
    assert result["source_hashes_verified_after_request"] is False
    assert result["boxes_applied"] is False


@pytest.mark.parametrize("damage", ["extra", "missing", "index", "bool_confidence", "nan", "enum", "empty_reason"])
def test_strict_schema_rejects_malformed_advice_without_applying(review, monkeypatch, damage):
    reply = response()
    if damage == "extra": reply["apply"] = True
    elif damage == "missing": reply["rectangles"] = []
    elif damage == "index": reply["rectangles"][0]["index"] = 2
    elif damage == "bool_confidence": reply["rectangles"][0]["confidence"] = True
    elif damage == "nan": reply["rectangles"][0]["confidence"] = float("nan")
    elif damage == "enum": reply["rectangles"][0]["inclusion"] = ["exclude_candidate"]
    elif damage == "empty_reason": reply["rectangles"][0]["reason"] = " "
    client = FakeClient(reply=reply); install(monkeypatch, client)
    result = K.advise_review(review["review_dir"], [BOX])
    assert result["status"] == "unavailable" and result["advice"] is None
    assert result["boxes_applied"] is False
    assert R.load_review(review["review_dir"])["status"] == "needs_review"


def test_provider_error_bodies_are_not_logged_or_persisted(review, monkeypatch, capsys):
    client = FakeClient(failure=True); install(monkeypatch, client)
    result = K.advise_review(review["review_dir"], [BOX])
    assert result["status"] == "unavailable"
    assert "PROVIDER_SECRET" not in capsys.readouterr().out
    for path in Path(result["advisory_path"]).parent.glob("*.json"):
        assert "PROVIDER_SECRET" not in path.read_text()


def test_each_advisory_has_new_directory_and_preserves_earlier_result(review, monkeypatch):
    client = FakeClient(); install(monkeypatch, client)
    first = K.advise_review(review["review_dir"], [BOX])
    digest = K._sha(first["advisory_path"])
    second = K.advise_review(review["review_dir"], [BOX])
    assert first["advisory_path"] != second["advisory_path"]
    assert K._sha(first["advisory_path"]) == digest
