"""Recorded physical row identities survive sparse and ambiguous synthetic image content."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import register as R
import tiles as T

W, H, DX, DY = 320, 180, 250, 150


def _grid(tmp_path, monkeypatch, *, missing=(), blank=(), reverse=False, negative_h=False,
          altered=None, nrows=20):
    """Known physical grid; altered content simulates a deceptive correlation peak."""
    rng = np.random.default_rng(74)
    canvas = cv2.GaussianBlur(rng.integers(20, 220, size=(nrows * DY + H, 3 * DX + W, 3),
                                         dtype=np.uint8), (0, 0), 1.0)
    h = -DX if negative_h else DX
    monkeypatch.setattr(R, "TILE_W", W)
    monkeypatch.setattr(R, "TILE_H", H)
    monkeypatch.setattr(R, "NOMINAL_FULL", (0, DY, h, 0))
    cache = tmp_path / "cache"
    ts, truth = [], {}
    for c in range(3):
        rows = range(nrows - 1, -1, -1) if reverse else range(nrows)
        for order, r in enumerate(rows):
            if (r, c) in missing:
                continue
            x = (2 - c) * DX if negative_h else c * DX
            t = T.Tile(col=c, col_idx=c, direction="up" if reverse else "down", order=order,
                       key=float(order), name=f"mosaic_r{r}_c{c}.png", zip_path="", inner="",
                       nominal_row=r)
            content_row = (altered or {}).get((r, c), r)
            image = canvas[content_row * DY:content_row * DY + H, x:x + W].copy()
            if (r, c) in blank:
                image[:] = 128
            path = Path(T.cache_path(str(cache), t))
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), image)
            ts.append(t)
            truth[t.tid] = (x, r * DY)
    return ts, str(cache), truth, (0, DY, h, 0)


def _run(ts, cache, prior, *, estimate_prior=False):
    edges, stats = R.build_edges(ts, cache, scale_div=1, workers=2, verbose=False,
                                nominal=prior, prior=None if estimate_prior else prior,
                                row_policy="recorded-grid")
    pos, diag = R.solve_positions(len(ts), edges, anchors=R.nominal_positions(ts, stats))
    assert stats["row_policy"] == "recorded-grid"
    assert stats["row_abs"] == {t.tid: t.nominal_row for t in ts}
    assert all(s["how"] == "recorded-grid" and s["offset"] == 0 for s in stats["segments"])
    assert len(pos) == len(ts) and np.isfinite(pos).all()
    assert diag["n_components"] == 1
    return edges, stats, pos


def _position_error(ts, truth, pos):
    expected = np.asarray([truth[t.tid] for t in ts], float)
    delta = pos - expected
    return np.linalg.norm(delta - np.median(delta, axis=0), axis=1)


def test_missing_two_rows_and_blank_tail_keep_recorded_spacing(tmp_path, monkeypatch):
    ts, cache, truth, prior = _grid(tmp_path, monkeypatch, missing=((15, 1), (16, 1)),
                                   blank=((17, 1), (18, 1), (19, 1)))
    edges, stats, pos = _run(ts, cache, prior, estimate_prior=True)
    assert len(ts) == 58
    a = next(i for i, t in enumerate(ts) if t.name == "mosaic_r14_c1.png")
    b = next(i for i, t in enumerate(ts) if t.name == "mosaic_r17_c1.png")
    gap = next(e for e in edges if e.i == a and e.j == b)
    assert gap.response == 1e-3 and gap.dy == pytest.approx(3 * stats["prior_full"]["dy_v"])
    assert _position_error(ts, truth, pos).max() < 1.0
    assert abs(stats["prior_full"]["dy_v"] - DY) < 1.0


@pytest.mark.parametrize("content_row", [7, 6], ids=["false-zero-step", "false-reverse-step"])
def test_deceptive_pixel_pair_cannot_change_row_identity(tmp_path, monkeypatch, content_row):
    # r8 has the same appearance as r7, or appears to precede r7, despite its recorded row.
    ts, cache, truth, prior = _grid(tmp_path, monkeypatch, altered={(8, 1): content_row})
    edges, stats, pos = _run(ts, cache, prior, estimate_prior=True)
    a = next(i for i, t in enumerate(ts) if t.name == "mosaic_r7_c1.png")
    b = next(i for i, t in enumerate(ts) if t.name == "mosaic_r8_c1.png")
    edge = next(e for e in edges if e.i == a and e.j == b)
    assert edge.response == 1e-3 and abs(edge.dy - DY) < 1.0
    assert _position_error(ts, truth, pos).max() < 1.0


def test_reverse_capture_order_uses_signed_recorded_delta(tmp_path, monkeypatch):
    ts, cache, truth, prior = _grid(tmp_path, monkeypatch, reverse=True)
    edges, _, pos = _run(ts, cache, prior, estimate_prior=True)
    vertical = [e for e in edges if e.kind == "v"]
    assert vertical and all(e.dy < 0 for e in vertical)
    assert _position_error(ts, truth, pos).max() < 1.0


def test_negative_horizontal_pitch_keeps_orientation_and_positive_overlap(tmp_path, monkeypatch):
    ts, cache, truth, prior = _grid(tmp_path, monkeypatch, negative_h=True)
    _, stats, pos = _run(ts, cache, prior, estimate_prior=True)
    assert stats["prior_full"]["dx_h"] < 0 and stats["measured_dx"] < 0
    assert abs(stats["measured_overlap_x_px"] - (W - DX)) < 1.0
    assert stats["nominal_overlap_x_px"] == W - DX
    assert _position_error(ts, truth, pos).max() < 1.0


def test_recorded_nonzero_row_origin_is_not_normalized_per_column(tmp_path, monkeypatch):
    ts, cache, truth, prior = _grid(tmp_path, monkeypatch, nrows=5)
    for t in ts:
        t.nominal_row += 40
    _, stats, pos = _run(ts, cache, prior)
    assert min(stats["row_abs"].values()) == 40
    assert _position_error(ts, truth, pos).max() < 1.0


@pytest.mark.parametrize("row", [True, 1.0, 1.25, float("nan")])
def test_recorded_grid_rejects_noninteger_rows_before_loading_images(tmp_path, row):
    t = T.Tile(0, 0, "down", 0, 0.0, "test.png", "", "", row)
    with pytest.raises(ValueError, match="integer nominal_row"):
        R.build_edges([t], str(tmp_path), row_policy="recorded-grid")


def test_recorded_grid_rejects_duplicate_rows_before_loading_images(tmp_path):
    ts = [T.Tile(0, 0, "down", i, float(i), f"{i}.png", "", "", 4) for i in range(2)]
    with pytest.raises(ValueError, match="Duplicate recorded-grid row"):
        R.build_edges(ts, str(tmp_path), row_policy="recorded-grid")


def test_unknown_policy_is_rejected_before_loading_images(tmp_path):
    with pytest.raises(ValueError, match="Unknown registration row_policy"):
        R.build_edges([], str(tmp_path), row_policy="automatic")


def test_default_inference_preserves_historical_serpentine_result(tmp_path):
    ts, truth = R._make_synthetic(str(tmp_path / "cache"))
    edges, stats = R.build_edges(ts, str(tmp_path / "cache"), workers=2, verbose=False)
    explicit_edges, explicit_stats = R.build_edges(ts, str(tmp_path / "cache"), workers=2,
                                                 verbose=False, row_policy="infer")
    assert stats["row_policy"] == "infer"
    assert edges == explicit_edges
    assert stats["row_abs"] == explicit_stats["row_abs"]
    assert stats["seg_offsets"] == explicit_stats["seg_offsets"]
    assert len({stats["row_abs"][t.tid] - truth[t.tid][2] for t in ts}) == 1
