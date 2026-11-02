"""Offline HTTP-action boundaries for human-reviewed spatial exclusions."""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from tools.analysis_workbench.test_workflow_server import wb, fake_manager


@pytest.fixture
def spatial_inputs(tmp_path):
    native = np.full((240, 160), 2, dtype=np.uint8)
    native[85:155, 50:110] = 3
    mask, valid, photo = [tmp_path / name for name in ('mask.npy', 'valid.npy', 'photo.png')]
    np.save(mask, native)
    np.save(valid, np.ones(native.shape, dtype=bool))
    Image.new('RGB', (160, 240), (230, 187, 189)).save(photo)
    return {'mask_path': str(mask), 'valid_path': str(valid), 'mosaic_path': str(photo),
            'exclusion_path': '', 'sample': 'synthetic-review', 'auto_central_exclusion': True}


def test_proposal_preview_and_confirmation_are_separate_actions(wb, spatial_inputs):
    manager = fake_manager(wb, {'jobs': [], 'prepared': []})
    manager.register_plan = Mock(return_value={'id': 'reviewed-job'})
    manager.start = Mock(return_value={'id': 'reviewed-job', 'status': 'starting'})
    result = wb.workflow_action('spatial-review', spatial_inputs)
    assert result['status'] == 'needs_review' and len(result['rectangles']) == 1
    manager.register_plan.assert_not_called()
    png = wb.spatial_review_artifact(result['review_id'], 'optical_preview.png')
    assert png.startswith(b'\x89PNG')
    with pytest.raises(ValueError, match='Unknown'):
        wb.spatial_review_artifact(result['review_id'], '../proposal.json')
    with pytest.raises(ValueError, match='confirm'):
        wb.workflow_action('spatial', spatial_inputs)
    manager.register_plan.assert_not_called()
    # A human explicitly clears the suggested rectangle, retaining all regions.
    body = {**spatial_inputs, 'review_id': result['review_id'],
            'rectangles': [], 'review_confirmed': True}
    final = wb.workflow_action('spatial', body)
    assert final['ok'] is True
    plan = manager.register_plan.call_args[0][0]
    assert '--auto-central-exclusion' not in plan['argv']
    path = Path(plan['source_binding']['sources']['exclusions']['path'])
    confirmed = json.loads(path.read_text())
    assert confirmed['excluded_rectangles'] == []
    assert confirmed['review']['human_confirmed'] is True
    assert json.loads(Path(result['proposal_path']).read_text())['status'] == 'needs_review'


def test_changed_input_or_batch_cannot_reuse_review(wb, spatial_inputs, tmp_path):
    manager = fake_manager(wb, {'jobs': [], 'prepared': []})
    manager.register_plan = Mock()
    result = wb.workflow_action('spatial-review', spatial_inputs)
    other = tmp_path / 'another.png'
    Image.new('RGB', (160, 240)).save(other)
    body = {**spatial_inputs, 'review_id': result['review_id'], 'rectangles': result['rectangles'],
            'review_confirmed': True, 'mosaic_path': str(other)}
    with pytest.raises(ValueError, match='differ from the reviewed'):
        wb.workflow_action('spatial', body)
    manager.register_plan.assert_not_called()
    Image.new('RGB', (160, 240), (1, 2, 3)).save(spatial_inputs['mosaic_path'])
    with pytest.raises(ValueError, match='Source changed'):
        wb.workflow_action('spatial', {**body, 'mosaic_path': spatial_inputs['mosaic_path']})


def test_review_id_cannot_escape_memory(wb):
    for value in ('../outside', '', None, 'a' * 31, 'a' * 33):
        with pytest.raises(ValueError, match='identifier'):
            wb.spatial_review_directory(value)
