"""Synthetic renders bind output contributions and omissions to source tile IDs."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import blend as B  # noqa: E402
from tiles import Tile, cache_path  # noqa: E402


@pytest.fixture(autouse=True)
def small_render_geometry(monkeypatch):
    monkeypatch.setattr(B, "TILE_W", 8)
    monkeypatch.setattr(B, "TILE_H", 6)


def source_tiles(tmp_path, records):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    tiles = []
    for i, (name, colour) in enumerate(records):
        tile = Tile(col=3, col_idx=0, direction="down", order=i, key=float(i),
                    name=name, zip_path=str(source), inner=name, nominal_row=i)
        raw = source / name
        cached = Path(cache_path(str(cache), tile))
        cached.parent.mkdir(parents=True, exist_ok=True)
        if colour is None:
            raw.write_bytes(b"synthetic undecodable image")
            cached.write_bytes(raw.read_bytes())
        else:
            image = np.full((6, 8, 3), colour, np.uint8)
            assert cv2.imwrite(str(raw), image)
            assert cv2.imwrite(str(cached), image)
        tiles.append(tile)
    return tiles, cache


@pytest.mark.parametrize("carry_bytes", [0, 1 << 20])
def test_render_reports_contributed_ids_in_source_order(tmp_path, carry_bytes):
    tiles, cache = source_tiles(tmp_path, [
        ("third.png", (30, 60, 90)),
        ("first.png", (50, 80, 110)),
        ("second.png", (70, 100, 130)),
    ])
    target = tmp_path / "normal.png"
    result = B.render(tiles, np.array([[-8, -6], [0, -6], [-8, 0]]), str(target),
                      out_scale=1, cache_dir=str(cache), feather=False, band_px=2,
                      carry_bytes=carry_bytes, verbose=False)
    image = cv2.imread(str(target))
    assert image.shape == (12, 16, 3)
    assert image[2, 2].tolist() == [30, 60, 90]
    assert image[2, 10].tolist() == [50, 80, 110]
    assert image[8, 2].tolist() == [70, 100, 130]
    assert result["rendered_tids"] == [t.tid for t in tiles]
    assert result["n_placed"] == len(result["rendered_tids"]) == 3
    assert result["failed_load_tids"] == [] and result["n_failed_loads"] == 0
    assert result["missing_position_tids"] == []


def test_render_reports_absent_and_nonfinite_positions(tmp_path):
    tiles, cache = source_tiles(tmp_path, [
        ("present.png", (30, 60, 90)),
        ("absent.png", (50, 80, 110)),
        ("nan.png", (70, 100, 130)),
        ("inf.png", (90, 120, 150)),
        ("last.png", (110, 140, 170)),
    ])
    positions = {tiles[0].tid: (0, 0), tiles[2].tid: (np.nan, 0),
                 tiles[3].tid: (0, np.inf), tiles[4].tid: (8, 6)}
    target = tmp_path / "missing-positions.png"
    result = B.render(tiles, positions, str(target), out_scale=1,
                      cache_dir=str(cache), feather=False, band_px=2, verbose=False)
    image = cv2.imread(str(target))
    assert image[2, 2].tolist() == [30, 60, 90]
    assert image[8, 10].tolist() == [110, 140, 170]
    assert result["rendered_tids"] == [tiles[0].tid, tiles[4].tid]
    assert result["missing_position_tids"] == [t.tid for t in tiles[1:4]]
    assert result["n_placed"] == len(result["rendered_tids"]) == 2
    assert result["failed_load_tids"] == [] and result["n_failed_loads"] == 0


@pytest.mark.parametrize("use_full", [False, True])
def test_render_separates_failed_loads_from_missing_positions(tmp_path, use_full):
    tiles, cache = source_tiles(tmp_path, [
        ("bad-z.png", None),
        ("good-z.png", (30, 60, 90)),
        ("missing.png", (50, 80, 110)),
        ("bad-a.png", None),
        ("good-a.png", (70, 100, 130)),
    ])
    positions = {tiles[0].tid: (0, 0), tiles[1].tid: (8, 0),
                 tiles[3].tid: (16, 0), tiles[4].tid: (24, 0)}
    target = tmp_path / "failed-loads.png"
    result = B.render(tiles, positions, str(target), out_scale=1,
                      cache_dir=str(cache), use_full=use_full, feather=False,
                      band_px=2, carry_bytes=0, verbose=False)
    image = cv2.imread(str(target))
    assert image.shape == (6, 32, 3)
    assert image[2, 2].tolist() == [255, 255, 255]
    assert image[2, 10].tolist() == [30, 60, 90]
    assert image[2, 18].tolist() == [255, 255, 255]
    assert image[2, 26].tolist() == [70, 100, 130]
    assert result["rendered_tids"] == [tiles[1].tid, tiles[4].tid]
    assert result["failed_load_tids"] == [tiles[0].tid, tiles[3].tid]
    assert result["missing_position_tids"] == [tiles[2].tid]
    assert result["n_placed"] == len(result["rendered_tids"]) == 2
    assert result["n_failed_loads"] == len(result["failed_load_tids"]) == 2
    assert result["n_decodes"] == 6  # Each good tile spans three uncached bands.


def test_render_reports_tile_that_fails_after_contributing_one_band(tmp_path, monkeypatch):
    tiles, cache = source_tiles(tmp_path, [("partial.png", (30, 60, 90))])
    tile = tiles[0]
    original_load_full = B.load_full
    loads = []

    def corrupt_after_first_decode(source):
        image = original_load_full(source)
        loads.append(source.tid)
        if len(loads) == 1:
            # The next uncached band really attempts to decode invalid bytes.
            (Path(source.zip_path) / source.inner).write_bytes(b"synthetic damaged image")
        return image

    monkeypatch.setattr(B, "load_full", corrupt_after_first_decode)
    target = tmp_path / "partial-load-failure.png"
    result = B.render(tiles, {tile.tid: (0, 0)}, str(target), out_scale=1,
                      cache_dir=str(cache), use_full=True, feather=False,
                      band_px=2, carry_bytes=0, verbose=False)
    image = cv2.imread(str(target))
    assert image.shape == (6, 8, 3)
    assert np.all(image[:2] == np.array([30, 60, 90], np.uint8))
    assert np.all(image[2:] == 255)
    assert loads == [tile.tid, tile.tid]
    assert result["rendered_tids"] == result["failed_load_tids"] == [tile.tid]
    assert result["missing_position_tids"] == []
    assert result["n_placed"] == result["n_failed_loads"] == result["n_decodes"] == 1
