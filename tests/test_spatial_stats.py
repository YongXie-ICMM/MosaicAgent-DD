"""Synthetic regression tests for layer-map pixel accounting (not accuracy)."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from flakepipeline.spatial_maps.stats import LAYERS, RGB, analyze, load_labels, sha


def rect(x1, y1, x2, y2):
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


def test_palette_indices_are_not_class_ids(tmp_path):
    indices = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    image = Image.fromarray(indices, mode="P")
    # Native Mosaic palette indices 1/2 map to 2L/1L, unlike this core's IDs.
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:4] = RGB[[0, 2, 1, 3]]
    image.putpalette(palette.reshape(-1).tolist())
    path = tmp_path / "permuted.png"
    image.save(path)
    np.testing.assert_array_equal(load_labels(path), [[0, 2, 1, 3], [3, 1, 2, 0]])


def test_rgb_and_opaque_rgba_masks(tmp_path):
    expected = np.array([[3, 0], [1, 2]], dtype=np.uint8)
    for mode in ("RGB", "RGBA"):
        path = tmp_path / f"{mode}.png"
        image = Image.fromarray(RGB[expected], mode="RGB").convert(mode)
        image.save(path)
        np.testing.assert_array_equal(load_labels(path), expected)


@pytest.mark.parametrize("mode", ["RGB", "P"])
def test_unknown_colour_is_rejected(tmp_path, mode):
    path = tmp_path / "unknown.png"
    image = Image.new("RGB", (2, 2), (1, 128, 0))
    if mode == "P":
        image = image.convert("P")
    image.save(path)
    with pytest.raises(ValueError, match="Unknown"):
        load_labels(path)


@pytest.mark.parametrize("mode", ["RGBA", "P"])
def test_transparent_pixels_are_rejected(tmp_path, mode):
    path = tmp_path / "transparent.png"
    if mode == "RGBA":
        image = Image.new("RGBA", (2, 2), (0, 128, 0, 254))
        image.save(path)
    else:
        image = Image.new("P", (2, 2), 0)
        image.putpalette([0, 128, 0] + [0] * 765)
        image.save(path, transparency=0)
    with pytest.raises(ValueError, match="Transparent"):
        load_labels(path)


def test_uneven_grid_retains_remainder_and_orientation():
    labels = (np.arange(35).reshape(5, 7) % 4).astype(np.uint8)
    table, xs, ys, whole = analyze(labels, [], 3, 2)
    np.testing.assert_array_equal(xs, [0, 2, 4, 7])
    np.testing.assert_array_equal(ys, [0, 2, 5])
    assert sum(t["tile_canvas_px"] for t in table) == 35
    for i, cell in enumerate(table):
        assert (cell["row_top_1based"], cell["column_left_1based"]) == (i // 3 + 1, i % 3 + 1)
        crop = labels[cell["y0_px"]:cell["y1_px"], cell["x0_px"]:cell["x1_px"]]
        assert [cell[f"{k}_px"] for k in LAYERS] == [int((crop == k).sum()) for k in (1, 2, 3)]
    assert whole["counts_px"] == dict(background=9, **{"1L": 9, "2L": 9, "TL": 8})


def test_valid_support_and_union_exclusions_are_disjoint_and_do_not_mutate():
    labels = (np.arange(35).reshape(5, 7) % 4).astype(np.uint8)
    valid = np.ones((5, 7), dtype=bool)
    valid[0, :] = False
    valid[3, 2] = False
    arbitrary = np.zeros((5, 7), dtype=bool)
    arbitrary[0, 0] = True  # Outside support: must not count as user-excluded.
    arbitrary[4, 6] = True
    arbitrary[2, 2] = True  # Inside a rectangle: count only once.
    original = [a.copy() for a in (labels, valid, arbitrary)]
    rectangles = [rect(1, 1, 4, 3), rect(2, 2, 5, 4)]
    table, _, _, whole = analyze(labels, rectangles, 3, 2, valid, arbitrary)
    # Independent explicit coordinate set, including rectangle intersection.
    region = {(x, y) for y in range(1, 3) for x in range(1, 4)}
    region |= {(x, y) for y in range(2, 4) for x in range(2, 5)}
    region |= {(0, 0), (6, 4), (2, 2)}
    excluded = {(x, y) for x, y in region if valid[y, x]}
    kept = [(x, y) for y in range(5) for x in range(7) if valid[y, x] and (x, y) not in region]
    expected = [sum(labels[y, x] == k for x, y in kept) for k in range(4)]
    assert whole["non_scan_px"] == 8
    assert whole["excluded_px"] == len(excluded) == 10
    assert whole["retained_canvas_px"] == len(kept) == 17
    assert list(whole["counts_px"].values()) == expected
    for cell in table:
        assert cell["retained_canvas_px"] + cell["excluded_px"] + cell["non_scan_px"] == cell["tile_canvas_px"]
    for before, after in zip(original, (labels, valid, arbitrary)):
        np.testing.assert_array_equal(before, after)


def test_global_composition_is_weighted_by_crystal_pixels():
    labels = np.array([[1, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint8)
    table, _, _, whole = analyze(labels, [], 2, 1)
    assert [t["1L_of_crystal_pct"] for t in table] == [100, 0]
    assert whole["of_crystal_pct"]["1L"] == 20
    assert whole["of_crystal_pct"]["2L"] == 80
    assert whole["of_retained_canvas_pct"]["1L"] == 12.5
    assert whole["crystal_coverage_pct"] == 62.5


@pytest.mark.parametrize("kind,status,excluded,non_scan", [
    ("no_crystal", "no_crystal", 0, 0),
    ("excluded", "excluded", 6, 0),
    ("no_scan", "no_scan", 0, 6),
])
def test_empty_categories_have_explicit_undefined_fractions(kind, status, excluded, non_scan):
    labels = np.zeros((2, 3), dtype=np.uint8)
    valid = np.full_like(labels, kind != "no_scan", dtype=bool)
    excluded_mask = np.full_like(labels, kind in ("excluded", "no_scan"), dtype=bool)
    table, _, _, whole = analyze(labels, [], 1, 1, valid, excluded_mask)
    assert table[0]["status"] == status
    assert whole["excluded_px"] == excluded
    assert whole["non_scan_px"] == non_scan
    assert all(v is None for v in whole["of_crystal_pct"].values())
    assert table[0]["1L_of_crystal_pct"] is None
    assert whole["crystal_coverage_pct"] == (0 if kind == "no_crystal" else None)
    json.dumps({"table": table, "global": whole}, allow_nan=False)


def test_partial_support_and_full_exclusion_status():
    labels = np.ones((2, 4), dtype=np.uint8)
    valid = np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.uint8)
    table, _, _, whole = analyze(labels, [rect(0, 0, 2, 2)], 2, 1, valid)
    assert [cell["status"] for cell in table] == ["excluded", "valid"]
    assert whole["non_scan_px"] == 4
    assert whole["excluded_px"] == 2
    assert whole["of_crystal_pct"] == {"1L": 100, "2L": 0, "TL": 0}


@pytest.mark.parametrize("labels", [
    np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=bool),
    np.zeros((2, 2, 1), dtype=int), np.array([[0, 4]]), np.array([[-1, 1]]),
])
def test_invalid_label_arrays_are_rejected(labels):
    with pytest.raises(ValueError, match="Labels"):
        analyze(labels, [], 1, 1)


@pytest.mark.parametrize("columns,rows", [(0, 1), (4, 1), (1, 3), (True, 1), (1, 1.5)])
def test_invalid_grid_is_rejected(columns, rows):
    with pytest.raises(ValueError, match="Grid"):
        analyze(np.zeros((2, 3), dtype=np.uint8), [], columns, rows)


@pytest.mark.parametrize("region", [rect(-1, 0, 1, 1), rect(0, 0, 4, 1), rect(0, 0, 0, 1),
                                    rect(0, 0, 1.0, 1), rect(False, 0, 1, 1), {"x1": 0}])
def test_invalid_exclusion_rectangles_are_rejected(region):
    with pytest.raises(ValueError, match="Exclusion"):
        analyze(np.zeros((2, 3), dtype=np.uint8), [region], 1, 1)


@pytest.mark.parametrize("name", ["valid_mask", "exclusion_mask"])
@pytest.mark.parametrize("mask", [np.zeros((3, 2)), np.full((2, 3), 2), np.full((2, 3), np.nan),
                                  np.full((2, 3), 0.2), np.full((2, 3), "1")])
def test_invalid_binary_masks_are_rejected(name, mask):
    with pytest.raises(ValueError, match=name):
        analyze(np.zeros((2, 3), dtype=np.uint8), [], 1, 1, **{name: mask})


def test_sha_matches_standard_digest_for_multiblock_file(tmp_path):
    content = b"synthetic spatial fixture" * 100000
    path = tmp_path / "large.bin"
    path.write_bytes(content)
    assert sha(path) == hashlib.sha256(content).hexdigest()
