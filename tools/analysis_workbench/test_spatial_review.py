"""Synthetic, offline tests of human-confirmed, source-bound ROI review."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from flakepipeline.spatial_maps import _load_sources
from flakepipeline.spatial_maps.stats import analyze, sha

SPEC = importlib.util.spec_from_file_location('spatial_review_under_test', Path(__file__).with_name('spatial_review.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


@pytest.fixture
def inputs(tmp_path):
    native = np.zeros((400, 500), np.uint8)
    native[140:260, 180:320] = 3
    native[20:70, 30:80] = 2  # native 2 is canonical 1L
    valid = np.ones(native.shape, bool); valid[0:4, :] = False
    base = np.zeros(native.shape, bool); base[25:35, 40:50] = True
    paths = {}
    for name, array in (('mask', native), ('valid', valid), ('base', base)):
        path = tmp_path/(name+'.npy'); np.save(path, array); paths[name] = path
    paths['mosaic'] = tmp_path/'mosaic.png'
    Image.new('RGB', (500, 400), (220, 180, 170)).save(paths['mosaic'])
    return paths, native, valid, base


def prepare(inputs, tmp_path, **kwargs):
    paths, *_ = inputs
    return R.prepare_review(paths['mask'], paths['valid'], paths['base'], paths['mosaic'],
                            tmp_path/'reviews', **kwargs)


def test_proposal_never_approves_and_preview_coordinates_are_source_bound(inputs, tmp_path):
    paths, *_ = inputs
    before = {key: sha(path) for key, path in paths.items()}
    proposal = prepare(inputs, tmp_path)
    record = R.load_review(proposal['review_dir'])
    assert record['status'] == proposal['status'] == 'needs_review'
    assert record['human_review_required'] is True
    assert record['automatic_proposal']['status'] == 'selected'
    assert record['rectangles'] == [{'x1':168, 'y1':130, 'x2':332, 'y2':270}]
    assert (record['image_width'], record['image_height']) == (500, 400)
    assert record['base_exclusion_mask'] == record['sources']['exclusions']
    assert record['preview_semantics']['protected_base_exclusions_editable'] is False
    assert record['preview_semantics']['protected_base_excluded_valid_px'] == 100
    assert record['sources']['mosaic']['sha256'] == before['mosaic']
    for key in ('optical', 'segmentation'):
        preview = Path(proposal['preview_paths'][key])
        assert preview.is_file()
        assert sha(preview) == record['previews'][key]['sha256']
        with Image.open(preview) as image:
            assert image.size == (record['previews'][key]['width'], record['previews'][key]['height'])
            if key == 'segmentation':
                pixels = np.asarray(image)
                assert pixels[30,45].tolist() == [150,150,150]
                assert pixels[0,0].tolist() == [255,255,255]
    assert {key:sha(path) for key,path in paths.items()} == before
    assert not list(Path(proposal['review_dir']).glob('*confirmed*'))


def test_human_may_change_boxes_and_confirm_base_union_without_source_mutation(inputs, tmp_path):
    paths, native, valid, base = inputs
    proposal = prepare(inputs, tmp_path)
    proposal_sha = sha(proposal['proposal_path'])
    before = {key:sha(path) for key,path in paths.items()}
    chosen = [{'x1': 180, 'y1': 140, 'x2': 320, 'y2': 260}]
    confirmation = R.confirm_review(proposal['review_dir'], chosen, tmp_path/'confirmed')
    record = confirmation['record']
    assert record['status'] == 'confirmed' and record['human_confirmed'] is True
    assert record['human_changed_proposal'] is True
    expected = base.copy(); expected[140:260,180:320] = True
    np.testing.assert_array_equal(np.load(confirmation['combined_mask_path']), expected)
    labels, revalid, bitmap, rectangles, sources, _ = _load_sources(paths['mask'], paths['valid'], confirmation['exclusion_path'])
    np.testing.assert_array_equal(bitmap, base)
    assert rectangles == chosen
    assert sources['base_exclusions'] == {'path':str(paths['base']), 'sha256':before['base']}
    *_, actual = analyze(labels, rectangles, 5, 4, revalid, bitmap)
    assert actual['excluded_px'] == int((valid & expected).sum())
    assert sha(proposal['proposal_path']) == proposal_sha
    assert {key:sha(path) for key,path in paths.items()} == before
    with pytest.raises(ValueError, match='exists'):
        R.confirm_review(proposal['review_dir'], chosen, tmp_path/'confirmed')


def test_empty_confirmed_rectangles_is_explicit_retain_decision_not_automatic(inputs, tmp_path):
    proposal = prepare(inputs, tmp_path)
    confirmation = R.confirm_review(proposal['review_dir'], [], tmp_path/'retain')
    assert confirmation['record']['human_confirmed'] is True
    assert confirmation['rectangles'] == []
    np.testing.assert_array_equal(np.load(confirmation['combined_mask_path']), inputs[3])


@pytest.mark.parametrize('changed', ['mask', 'valid', 'base', 'mosaic'])
def test_stale_originals_require_new_review(inputs, tmp_path, changed):
    proposal = prepare(inputs, tmp_path)
    with inputs[0][changed].open('ab') as handle:
        handle.write(b'synthetic change')
    with pytest.raises(ValueError, match='Source changed'):
        R.confirm_review(proposal['review_dir'], [], tmp_path/'stale')
    assert not (tmp_path/'stale').exists()


@pytest.mark.parametrize('changed', ['proposal', 'optical', 'segmentation'])
def test_modified_proposals_or_previews_are_rejected(inputs, tmp_path, changed):
    proposal = prepare(inputs, tmp_path)
    path = Path(proposal['proposal_path'] if changed == 'proposal' else proposal['preview_paths'][changed])
    with path.open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError, match='changed'):
        R.confirm_review(proposal['review_dir'], [], tmp_path/'tampered')
    assert not (tmp_path/'tampered').exists()


@pytest.mark.parametrize('boxes', [None, [{}], [{'x1':1.1,'y1':0,'x2':5,'y2':5}],
    [{'x1':True,'y1':0,'x2':5,'y2':5}], [{'x1':5,'y1':1,'x2':5,'y2':5}],
    [{'x1':0,'y1':0,'x2':501,'y2':5}], [{'x1':0,'y1':-1,'x2':5,'y2':5}],
    [{'x1':0,'y1':0,'x2':5,'y2':5}]*101])
def test_invalid_or_zero_area_boxes_never_create_confirmation(inputs, tmp_path, boxes):
    proposal = prepare(inputs, tmp_path)
    with pytest.raises(ValueError):
        R.confirm_review(proposal['review_dir'], boxes, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_disabled_or_absent_auto_candidate_still_allows_manual_review(inputs, tmp_path):
    disabled = prepare(inputs, tmp_path, automatic=False)
    assert disabled['rectangles'] == []
    assert disabled['record']['automatic_proposal']['status'] == 'disabled'
    paths, native, *_ = inputs
    native[:] = 0; np.save(paths['mask'], native)
    absent = prepare(inputs, tmp_path)
    assert absent['rectangles'] == [] and absent['status'] == 'needs_review'
    assert absent['record']['automatic_proposal']['status'] == 'needs_review'
    manual = [{'x1':10,'y1':10,'x2':30,'y2':40}]
    result = R.confirm_review(absent['review_dir'], manual, tmp_path/'manual')
    assert result['rectangles'] == manual


def test_existing_json_rectangles_are_editable_but_indirect_bitmap_is_protected(inputs, tmp_path):
    paths, *_ = inputs
    original = {'x1':10,'y1':10,'x2':30,'y2':40}
    existing = tmp_path/'existing.json'
    existing.write_text(json.dumps({'image_width':500,'image_height':400,
        'excluded_rectangles':[original],
        'base_exclusion_mask':{'path':str(paths['base']), 'sha256':sha(paths['base'])}}))
    proposal = R.prepare_review(paths['mask'],paths['valid'],existing,paths['mosaic'],tmp_path/'reviews',automatic=False)
    assert proposal['rectangles'] == [original]
    confirmed = R.confirm_review(proposal['review_dir'], [], tmp_path/'edited')
    np.testing.assert_array_equal(np.load(confirmed['combined_mask_path']), inputs[3])
    assert confirmed['record']['human_changed_proposal']


def test_invalid_binary_support_and_mismatched_optical_are_rejected(inputs, tmp_path):
    paths, _, valid, _ = inputs
    bad = valid.astype(np.uint8); bad[0,0] = 2; np.save(paths['valid'],bad)
    with pytest.raises(ValueError,match='binary'):
        prepare(inputs,tmp_path)
    np.save(paths['valid'], valid)
    Image.new('RGB',(10,10)).save(paths['mosaic'])
    with pytest.raises(ValueError,match='dimensions'):
        prepare(inputs,tmp_path)
    assert not list((tmp_path/'reviews').glob('.review-staging-*'))


def test_repeated_preparations_create_separate_immutable_review_folders(inputs,tmp_path):
    first = prepare(inputs,tmp_path)
    before = Path(first['proposal_path']).read_bytes()
    second = prepare(inputs,tmp_path)
    assert first['review_id'] != second['review_id']
    assert first['review_dir'] != second['review_dir']
    assert Path(first['proposal_path']).read_bytes() == before
