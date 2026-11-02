"""Offline synthetic tests of opt-in central exclusion and three-panel export."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from flakepipeline.spatial_maps.figure4 import (
    CentralExclusionNeedsReview, automatic_central_exclusion, draw_figure4,
)
from flakepipeline.spatial_maps.stats import analyze, sha


def central_fixture():
    labels = np.zeros((400, 500), np.uint8)
    labels[140:260, 180:320] = 3
    # Canonical single/bilayer labels do not enter TL proposals.
    labels[20:70, 30:80] = 1
    labels[320:360, 410:450] = 2
    return labels, np.ones(labels.shape, bool)


def test_unique_central_bbox_order_and_ceil_padding_preserve_inputs():
    labels, valid = central_fixture()
    before_labels, before_valid = labels.copy(), valid.copy()
    result = automatic_central_exclusion(labels, valid)
    assert result['status'] == 'selected'
    assert result['selected']['bbox_xyxy'] == [180, 140, 320, 260]
    assert result['selected']['legacy_bbox_x0_x1_y0_y1'] == [180, 320, 140, 260]
    assert result['rectangles'] == [{'x1': 168, 'y1': 130, 'x2': 332, 'y2': 270}]
    assert result['padding_px'] == {'x_per_side': 12, 'y_per_side': 10}
    assert result['policy']['canonical_ids'] == {'background': 0, '1L': 1, '2L': 2, 'TL': 3}
    np.testing.assert_array_equal(labels, before_labels)
    np.testing.assert_array_equal(valid, before_valid)


@pytest.mark.parametrize('case', ['no_tl', 'off_centre', 'unsupported', 'small', 'oversized', 'edge_connected'])
def test_absent_or_ineligible_proposals_stop_for_review(case):
    labels, valid = central_fixture()
    if case == 'no_tl':
        labels[labels == 3] = 1
    elif case == 'off_centre':
        labels[labels == 3] = 0
        labels[60:160, 60:160] = 3
    elif case == 'unsupported':
        valid[140:260, 180:320] = False
    elif case == 'small':
        labels[labels == 3] = 0
        labels[192:208, 244:260] = 3
    elif case == 'oversized':
        labels[40:360, 40:460] = 3
    elif case == 'edge_connected':
        # Closing retracts this stripe from the border; original connectivity
        # still identifies it as an edge object rather than a central island.
        labels[labels == 3] = 0
        labels[164:236, :] = 3
    before = labels.copy()
    with pytest.raises(CentralExclusionNeedsReview, match='needs_review') as error:
        automatic_central_exclusion(labels, valid)
    assert error.value.audit['status'] == 'needs_review'
    assert error.value.audit['rectangles'] == []
    if case == 'edge_connected':
        reasons = error.value.audit['candidates'][0]['rejection_reasons']
        assert 'connected_to_canvas_edge_in_original_sampled_tl' in reasons
    np.testing.assert_array_equal(labels, before)


def test_two_disconnected_centre_containing_boxes_are_ambiguous():
    labels = np.zeros((400, 400), np.uint8)
    labels[100:300, 100:300] = 3
    labels[120:280, 120:280] = 0
    labels[180:220, 180:220] = 3
    with pytest.raises(CentralExclusionNeedsReview, match='Multiple') as error:
        automatic_central_exclusion(labels, np.ones(labels.shape, bool))
    assert error.value.audit['eligible_candidates'] == 2
    assert sum(c['eligible'] for c in error.value.audit['candidates']) == 2


def test_full_resolution_counts_are_not_candidate_area_estimates():
    labels, valid = central_fixture()
    # A native one-pixel hole is invisible to stride-four measurement.
    labels[141, 181] = 1
    valid[142, 182] = False
    result = automatic_central_exclusion(labels, valid)
    _, _, _, before = analyze(labels, [], 5, 4, valid)
    _, _, _, after = analyze(labels, result['rectangles'], 5, 4, valid)
    assert result['selected']['estimated_tl_area_px'] == 140 * 120
    assert before['counts_px']['TL'] == 140 * 120 - 2
    assert after['counts_px']['TL'] == 0
    assert after['counts_px']['1L'] == 2500
    assert after['counts_px']['2L'] == 1600


@pytest.mark.parametrize('kwargs', [{'downsample': 0}, {'closing_size': 2}, {'padding_fraction': -1},
                                    {'min_component_canvas_fraction': .5, 'max_bbox_canvas_fraction': .4}])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        automatic_central_exclusion(*central_fixture(), **kwargs)


def test_draw_preserves_sources_and_exports_named_editable_panels(tmp_path):
    labels, valid = central_fixture()
    policy = automatic_central_exclusion(labels, valid)
    omitted = np.zeros(labels.shape, bool)
    # Existing arbitrary exclusion support must be reflected in the same table.
    omitted[25:35, 40:50] = True
    table, xs, ys, whole = analyze(labels, policy['rectangles'], 5, 4, valid, omitted)
    data = dict(panel='synthetic', width=500, height=400, rectangles=policy['rectangles'],
                table=table, xedges=xs, yedges=ys, **{'global': whole})
    photo = tmp_path/'colour.png'
    Image.fromarray(np.full((400, 500, 3), [225, 185, 180], np.uint8)).save(photo)
    source_hash = sha(photo)
    before_labels, before_valid, before_omitted = labels.copy(), valid.copy(), omitted.copy()
    outputs = draw_figure4(data, labels, valid, omitted, photo, tmp_path/'out')
    assert Path(outputs['png']).name == 'synthetic_Figure4.png'
    root = ET.parse(outputs['svg']).getroot()
    assert root.attrib['width'] == '510.23622pt'
    assert root.attrib['height'] == '382.677165pt'
    text = ''.join(root.itertext())
    for caption in ('Colour-matched optical', 'Layer segmentation', 'Local composition', 'Whole-map composition'):
        assert caption in text
    assert 'Composition by row' not in text
    assert 'saved pixels' in text
    assert len(root.findall('.//{http://www.w3.org/2000/svg}image')) == 2
    assert len(root.findall('.//{http://www.w3.org/2000/svg}text')) > 10
    with Image.open(outputs['png']) as result:
        assert result.size == (3543, 2657)
    assert 'retained classified MoS₂ pixels' in Path(outputs['caption']).read_text()
    assert sha(photo) == source_hash
    for actual, expected in ((labels, before_labels), (valid, before_valid), (omitted, before_omitted)):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='already exists'):
        draw_figure4(data, labels, valid, omitted, photo, tmp_path/'out')


def test_draw_rejects_unbound_photo_dimensions_before_writing(tmp_path):
    labels, valid = central_fixture()
    table, xs, ys, whole = analyze(labels, [], 5, 4, valid)
    data = dict(panel='synthetic', width=500, height=400, rectangles=[], table=table,
                xedges=xs, yedges=ys, **{'global': whole})
    photo = tmp_path/'wrong.png'; Image.new('RGB', (20, 20)).save(photo)
    with pytest.raises(ValueError, match='dimensions'):
        draw_figure4(data, labels, valid, np.zeros_like(valid), photo, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_draw_rejects_table_that_omits_existing_bitmap_exclusion(tmp_path):
    labels, valid = central_fixture()
    table, xs, ys, whole = analyze(labels, [], 5, 4, valid)
    data = dict(panel='synthetic', width=500, height=400, rectangles=[], table=table,
                xedges=xs, yedges=ys, **{'global': whole})
    omitted = np.zeros_like(valid); omitted[20:40, 30:50] = True
    with pytest.raises(ValueError, match='exclusion union'):
        draw_figure4(data, labels, valid, omitted, tmp_path/'unused.png', tmp_path/'out')
    assert not (tmp_path/'out').exists()
