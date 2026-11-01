"""Ordinary stitching colour correction: synthetic pixels only, no network/hardware."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest

import blend as B
import stitch_colour as SC
import tiles as T


W, H = 320, 180
DX, DY = 270, 140


def quiet(*_):
    pass


def fixture(tmp_path, *, zipped=False, blank=None, ncols=5, nrows=6):
    """One sensor-colour state changes halfway down a column, then restores."""
    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(44)
    w, h = (ncols - 1) * DX + W, (nrows - 1) * DY + H
    texture = cv2.GaussianBlur(rng.normal(0, 22, (h, w)).astype(np.float32), (0, 0), 0.7)
    rgb = np.array([174.0, 142.0, 128.0]) + texture[:, :, None]
    rgb += rng.normal(0, 1.2, rgb.shape)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field = np.stack([0.92 + 0.16 * xx / W, 0.95 + 0.10 * yy / H,
                      0.93 + 0.14 * xx / W], axis=2)
    field /= field.mean(axis=(0, 1))
    tiles, positions, exposure = [], [], {}
    changed = np.array([1.08, 1.03, 0.95], np.float32)
    archive = zipfile.ZipFile(tmp_path / "source.zip", "w") if zipped else None
    try:
        for c in range(ncols):
            for r in range(nrows):
                # Increasing acquisition columns lie leftward, as on the scanner.
                x, y = (ncols - 1 - c) * DX, r * DY
                factor = changed if c == 2 and 2 <= r <= 4 else np.ones(3)
                exposure[(r, c)] = factor.copy()
                image = np.clip(np.rint(rgb[y:y + H, x:x + W] * field * factor), 0, 255).astype(np.uint8)
                if blank == (r, c) or blank == "all":
                    image[:] = 0
                name = f"mosaic_r{r}_c{c}.png"
                memory = io.BytesIO()
                Image.fromarray(image).save(memory, format="PNG")
                if archive:
                    archive.writestr("scan/" + name, memory.getvalue())
                else:
                    (tmp_path / name).write_bytes(memory.getvalue())
                tile = T.Tile(col=c, col_idx=c, direction="down", order=r, key=float(r),
                              name=name, zip_path=str(tmp_path / "source.zip" if archive else tmp_path),
                              inner="scan/" + name if archive else name, nominal_row=r)
                tiles.append(tile)
                positions.append([x, y])
    finally:
        if archive:
            archive.close()
    return tiles, np.array(positions, np.float64), field[:, :, ::-1].copy(), exposure


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def test_mid_column_colour_state_is_measured_without_touching_sources(tmp_path, monkeypatch):
    selected, positions, ff, exposure = fixture(tmp_path)
    before = hashes(tmp_path)
    before_positions = positions.copy()
    def forbidden(*_):
        raise AssertionError("Colour matching must not read registration thumbnails")
    monkeypatch.setattr(T, "load_small", forbidden)
    gains, report = SC.solve_tile_gains(selected, positions, ff, binding={"input_manifest": "synthetic"}, log=quiet)
    assert report["status"] == "matched"
    assert report["graph"]["connected"]
    assert report["direction"]["next_column_side"] == "left"
    assert report["source_binding"] == {"input_manifest": "synthetic"}
    assert report["edges_used"] >= 45
    assert report["solver"]["edge_rms_after"] < report["solver"]["edge_rms_before"]
    assert report["measured_overlap_log_rms_after"] < report["measured_overlap_log_rms_before"]
    corrected = np.array([np.log(gains[t.tid]) + np.log(exposure[(t.nominal_row, t.col)]) for t in selected])
    assert np.max(corrected.std(axis=0)) < 0.012
    # The state changes within c2, so one constant column gain would not suffice.
    affected = next(t for t in selected if (t.nominal_row, t.col) == (3, 2))
    normal = next(t for t in selected if (t.nominal_row, t.col) == (0, 2))
    relative = np.array(gains[affected.tid]) / np.array(gains[normal.tid])
    assert relative[0] < 0.96 and relative[2] > 1.01
    assert hashes(tmp_path) == before
    np.testing.assert_array_equal(positions, before_positions)
    assert set(gains) == {t.tid for t in selected}
    json.dumps(report, allow_nan=False)


def test_zip_and_loose_sources_have_identical_rgb_gains(tmp_path):
    loose, pos, ff, _ = fixture(tmp_path / "loose")
    zipped, pos_zip, ff_zip, _ = fixture(tmp_path / "zipped", zipped=True)
    gain_loose, report_loose = SC.solve_tile_gains(loose, pos, ff, log=quiet)
    gain_zip, report_zip = SC.solve_tile_gains(zipped, pos_zip, ff_zip, log=quiet)
    assert gain_loose == gain_zip
    assert report_loose["edges"] == report_zip["edges"]


def test_isolated_tile_refuses_unconstrained_gain_and_preserves_rejections(tmp_path):
    selected, pos, ff, _ = fixture(tmp_path, blank=(0, 0))
    with pytest.raises(SC.StitchColourError) as caught:
        SC.solve_tile_gains(selected, pos, ff, log=quiet)
    report = caught.value.report
    assert report["failure_reason"] == "disconnected_colour_graph"
    assert not report["graph"]["connected"]
    assert report["edges_rejected"] >= 2
    assert [selected[0].tid] in report["graph"]["components"]
    assert "gains_rgb_by_tid" not in report


def test_all_rejected_edges_are_retained_in_failure_evidence(tmp_path):
    selected, pos, ff, _ = fixture(tmp_path, blank="all", ncols=2, nrows=2)
    with pytest.raises(SC.StitchColourError) as caught:
        SC.solve_tile_gains(selected, pos, ff, log=quiet)
    assert caught.value.report["edges_rejected"] == 4
    assert len(caught.value.report["edges"]) == 4


@pytest.mark.parametrize("kind", ["missing_grid", "duplicate", "no_position", "bad_field", "upward_rows"])
def test_invalid_geometry_or_field_fails_explicitly(tmp_path, kind):
    selected, pos, ff, _ = fixture(tmp_path, ncols=3, nrows=3)
    if kind == "missing_grid":
        selected.pop(4)
        pos = np.delete(pos, 4, axis=0)
    elif kind == "duplicate":
        selected[1] = selected[0]
    elif kind == "no_position":
        pos[0] = np.nan
    elif kind == "bad_field":
        ff[0, 0, 0] = np.nan
    elif kind == "upward_rows":
        pos[:, 1] *= -1
    with pytest.raises(SC.StitchColourError) as caught:
        SC.solve_tile_gains(selected, pos, ff, log=quiet)
    assert caught.value.report["status"] == "failed"
    json.dumps(caught.value.report, allow_nan=False)


def test_single_tile_has_explicit_identity_gains(tmp_path):
    selected, pos, ff, _ = fixture(tmp_path, ncols=1, nrows=1)
    gains, report = SC.solve_tile_gains(selected, pos, ff, log=quiet)
    assert gains == {selected[0].tid: [1.0, 1.0, 1.0]}
    assert report["status"] == "identity_single_tile"


@pytest.mark.parametrize("ncols,nrows", [(1, 3), (3, 1)])
def test_one_dimensional_grids_need_only_their_actual_overlap_edges(tmp_path, ncols, nrows):
    selected, pos, ff, _ = fixture(tmp_path, ncols=ncols, nrows=nrows)
    gains, report = SC.solve_tile_gains(selected, pos, ff, log=quiet)
    assert report["graph"]["connected"]
    assert report["edges_used"] == 2
    assert set(gains) == {t.tid for t in selected}


def _render_tile(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "TILE_W", 40)
    monkeypatch.setattr(B, "TILE_H", 30)
    image = np.empty((30, 40, 3), np.uint8)
    image[:] = (160, 120, 80)  # BGR, distinctive channel order and pre-gain saturation.
    source = tmp_path / "tile.png"
    cv2.imwrite(str(source), image)
    tile = T.Tile(col=0, col_idx=0, direction="down", order=0, key=0, name=source.name,
                  zip_path=str(tmp_path), inner=source.name, nominal_row=0)
    return tile, image


def test_render_combines_rgb_gain_and_flatfield_once_across_bands(tmp_path, monkeypatch):
    tile, source = _render_tile(tmp_path, monkeypatch)
    ff = np.full((5, 6, 3), 0.5, np.float32)
    gains = {tile.tid: [0.6, 0.75, 0.5]}
    paths = [tmp_path / name for name in ("single.png", "banded.png", "uncached.png")]
    first = B.render([tile], [[0, 0]], str(paths[0]), out_scale=1, use_full=True,
                     flatfield=ff, tile_gains_rgb=gains, band_px=None, verbose=False)
    banded = B.render([tile], [[0, 0]], str(paths[1]), out_scale=1, use_full=True,
                      flatfield=ff, tile_gains_rgb=gains, band_px=7, verbose=False)
    uncached = B.render([tile], [[0, 0]], str(paths[2]), out_scale=1, use_full=True,
                        flatfield=ff, tile_gains_rgb=gains, band_px=7, carry_bytes=0, verbose=False)
    expected = np.clip(source.astype(np.float32) / 0.5 * np.array([0.5, 0.75, 0.6], np.float32), 0, 255).astype(np.uint8)
    for path in paths:
        np.testing.assert_array_equal(cv2.imread(str(path)), expected)
    assert first["n_decodes"] == banded["n_decodes"] == 1
    assert uncached["n_decodes"] == 5
    # Premature clipping after /flatfield would give B=127, rather than B=160.
    assert cv2.imread(str(paths[0]))[0, 0].tolist() == [160, 180, 96]
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "tile.png")), source)


def test_cached_thumbnail_render_also_applies_gain_once(tmp_path, monkeypatch):
    tile, _ = _render_tile(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    cached = Path(T.cache_path(str(cache), tile))
    cached.parent.mkdir(parents=True)
    thumbnail = np.full((15, 20, 3), [160, 120, 80], np.uint8)
    cv2.imwrite(str(cached), thumbnail)
    out = tmp_path / "cached.png"
    info = B.render([tile], [[0, 0]], str(out), out_scale=0.5, cache_dir=str(cache),
                    flatfield=np.full((5, 6, 3), 0.5, np.float32),
                    tile_gains_rgb={tile.tid: [0.6, 0.75, 0.5]}, band_px=4, verbose=False)
    expected = np.full((15, 20, 3), [160, 180, 96], np.uint8)
    np.testing.assert_array_equal(cv2.imread(str(out)), expected)
    assert info["n_decodes"] == 1


@pytest.mark.parametrize("gain", [[float("nan"), 1, 1], [1, 1], [0.49, 1, 1], [1, 2.01, 1], "bad"])
def test_render_refuses_invalid_gains_before_output(tmp_path, monkeypatch, gain):
    tile, _ = _render_tile(tmp_path, monkeypatch)
    out = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="gain|RGB"):
        B.render([tile], [[0, 0]], str(out), out_scale=1, use_full=True,
                 tile_gains_rgb={tile.tid: gain}, verbose=False)
    assert not out.exists()


def test_render_refuses_missing_tile_gain(tmp_path, monkeypatch):
    tile, _ = _render_tile(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="IDs"):
        B.render([tile], [[0, 0]], str(tmp_path / "bad.png"), out_scale=1, use_full=True,
                 tile_gains_rgb={}, verbose=False)


def test_none_gain_path_is_byte_identical_to_prior_render(tmp_path, monkeypatch):
    """Baseline decoded BGR pixels generated by 986d7c4:blend.py before edits.

    A decoded control hash needs neither historical Git objects in a source ZIP
    nor matching PNG/zlib implementations. Omitted and explicit None arguments
    must additionally produce byte-identical PNGs in the current environment.
    """
    tile, _ = _render_tile(tmp_path, monkeypatch)
    ff = np.linspace(0.8, 1.2, 5 * 6 * 3, dtype=np.float32).reshape(5, 6, 3)
    omitted, current = tmp_path / "omitted.png", tmp_path / "current.png"
    kwargs = dict(out_scale=1, use_full=True, flatfield=ff, band_px=7, verbose=False)
    B.render([tile], [[0, 0]], str(omitted), **kwargs)
    B.render([tile], [[0, 0]], str(current), tile_gains_rgb=None, **kwargs)
    assert current.read_bytes() == omitted.read_bytes()
    pixels = cv2.imread(str(current))
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == "837f5de2efda9cb51059963c9bfc45db3bd449fc299d2c4113cbcd96c58fc504"
