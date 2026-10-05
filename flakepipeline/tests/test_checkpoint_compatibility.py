"""Exercise old/new PyTorch load signatures without loading torch or checkpoints."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def segmenter(monkeypatch):
    torch = ModuleType("torch")
    nn = ModuleType("torch.nn")
    functional = ModuleType("torch.nn.functional")
    nn.Module = type("Module", (), {})
    nn.Sequential = type("Sequential", (), {})
    nn.functional = functional
    torch.nn = nn
    for name, module in (("torch", torch), ("torch.nn", nn),
                         ("torch.nn.functional", functional)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "checkpoint_compat_segmenter", Path(__file__).resolve().parents[1] / "seg_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_old_loader_does_not_receive_weights_only(segmenter):
    calls = []
    checkpoint = object()

    def legacy_load(path, map_location=None, **pickle_load_args):
        calls.append((path, map_location, pickle_load_args))
        return checkpoint

    segmenter.torch.load = legacy_load
    assert segmenter._load_checkpoint("legacy.pth") is checkpoint
    assert calls == [("legacy.pth", "cpu", {})]


def test_modern_loader_preserves_explicit_false(segmenter):
    calls = []
    checkpoint = object()

    def modern_load(path, map_location=None, *, weights_only=True):
        calls.append((path, map_location, weights_only))
        return checkpoint

    segmenter.torch.load = modern_load
    assert segmenter._load_checkpoint("modern.pth") is checkpoint
    assert calls == [("modern.pth", "cpu", False)]


def test_checkpoint_error_propagates_without_retry(segmenter):
    calls = []

    def failing_load(path, map_location=None, *, weights_only=True):
        calls.append((path, map_location, weights_only))
        raise TypeError("invalid checkpoint content")

    segmenter.torch.load = failing_load
    with pytest.raises(TypeError, match="invalid checkpoint content"):
        segmenter._load_checkpoint("invalid.pth")
    assert calls == [("invalid.pth", "cpu", False)]
