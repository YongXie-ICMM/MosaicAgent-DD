"""Publication-package configuration and headless focus checks; no external calls."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kimi_agents as KA


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    for name in ("MOSAIC_ENV", "KB_ROOT", "KIMI_API_KEY", "KIMI_BASE_URL",
                 "KIMI_MODEL", "KIMI_VISION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(KA, "__file__", str(repo / "kimi_agents.py"))
    return repo


def test_credentials_ignore_unrelated_working_and_home_locations(isolated_config, tmp_path, monkeypatch):
    unrelated = tmp_path / "working"
    unrelated.mkdir()
    (unrelated / ".env").write_text("KIMI_API_KEY=unrelated-cwd-value\n")
    monkeypatch.chdir(unrelated)
    kb = tmp_path / "kb"
    (kb / "5_AI_SYSTEM").mkdir(parents=True)
    (kb / "5_AI_SYSTEM/.env").write_text("KIMI_API_KEY=unrelated-kb-value\n")
    monkeypatch.setenv("KB_ROOT", str(kb))

    def forbidden_home():
        raise AssertionError("Implicit home/Desktop/Downloads discovery is forbidden")

    monkeypatch.setattr(Path, "home", staticmethod(forbidden_home))
    assert KA._env_candidates() == [isolated_config / ".env"]
    assert KA.load_env() == {}


def test_explicit_file_precedes_repo_and_environment_overrides_both(isolated_config, tmp_path, monkeypatch):
    (isolated_config / ".env").write_text(
        "KIMI_API_KEY=repository-placeholder\nKIMI_MODEL=repo-model\n"
        "KIMI_VISION_MODEL=repo-vision\nKIMI_BASE_URL=https://example.invalid/v1\n")
    explicit = tmp_path / "explicit.env"
    explicit.write_text("KIMI_API_KEY=explicit-placeholder\nKIMI_VISION_MODEL=explicit-vision\n")
    monkeypatch.setenv("MOSAIC_ENV", str(explicit))
    assert KA.load_env()["KIMI_API_KEY"] == "explicit-placeholder"
    assert KA.load_env()["KIMI_VISION_MODEL"] == "explicit-vision"
    monkeypatch.setenv("KIMI_API_KEY", "environment-placeholder")
    monkeypatch.setenv("KIMI_VISION_MODEL", "environment-vision")
    config = KA.load_env()
    assert config["KIMI_API_KEY"] == "environment-placeholder"
    assert config["KIMI_VISION_MODEL"] == "environment-vision"
    assert config["KIMI_MODEL"] == "repo-model"


def test_repository_env_is_supported_without_other_files(isolated_config):
    (isolated_config / ".env").write_text("KIMI_API_KEY=repository-placeholder\n")
    assert KA.load_env()["KIMI_API_KEY"] == "repository-placeholder"


def test_focus_backend_does_not_import_gui_or_stage_driver():
    script = '''
import builtins, json, sys
import numpy as np
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "tkinter" or name.startswith("tkinter.") or "ximc" in name.lower():
        raise AssertionError("A GUI or stage driver was imported: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import tiles
from flakepipeline import autoscan_agent as compatibility
from flakepipeline import focus_metrics as focus
backend = tiles._focus_backend()
assert backend is not False
assert not hasattr(compatibility, "MosaicNavigatorGUI")
assert not hasattr(compatibility, "XIMCStageFullStep")
frame = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
assert compatibility.focus_score(frame) == focus.focus_score(frame)
print(json.dumps({"focus": backend[0](frame)["score"]}))
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["focus"] > 0


@pytest.mark.parametrize("module", ["flakepipeline.focus_metrics", "flakepipeline.autoscan_agent"])
def test_no_arguments_do_not_launch_a_gui(module):
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "--check-focus" in result.stderr
