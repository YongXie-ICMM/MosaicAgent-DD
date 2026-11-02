"""Immutable, source-bound human review before spatial exclusions are applied.

Preview images are for editing native-pixel rectangles only. Exact counts
always use the original semantic labels and support; no model is called here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
from PIL import Image

from flakepipeline.spatial_maps import _load_sources
from flakepipeline.spatial_maps.figure4 import automatic_central_exclusion, CentralExclusionNeedsReview
from flakepipeline.spatial_maps.stats import RGB, require, sha


PROPOSAL_FILENAME = 'proposal.json'
PROPOSAL_HASH_FILENAME = 'proposal.sha256'
MAX_RECTANGLES = 100
MAX_JSON_BYTES = 8 * 1024 * 1024


def _json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,
            'Missing or oversized review JSON record')
    def reject(value):
        raise ValueError('Nonfinite JSON value in review record')
    result = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject)
    require(isinstance(result, dict), 'Review record must be a JSON object')
    return result


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _source(path):
    path = Path(path).expanduser().resolve()
    require(path.is_file() and path.name != '.env', 'Missing input file; credentials are not analysis inputs')
    return {'path': str(path), 'sha256': sha(path)}


def _verify_sources(sources):
    require(isinstance(sources, dict) and 'mask' in sources and 'mosaic' in sources,
            'Review source bindings are incomplete')
    for name, record in sources.items():
        require(isinstance(record, dict) and isinstance(record.get('path'), str)
                and isinstance(record.get('sha256'), str), 'Invalid review source binding')
        path = Path(record['path'])
        require(path.is_file() and path.name != '.env', f'Missing review source: {name}')
        require(sha(path) == record['sha256'],
                f'Source changed since review preparation: {name}; prepare a new review')


def _rectangles(rectangles, width, height):
    require(isinstance(rectangles, (list, tuple)) and len(rectangles) <= MAX_RECTANGLES,
            f'Rectangles must be a list with at most {MAX_RECTANGLES} entries')
    result = []
    for item in rectangles:
        require(isinstance(item, dict) and all(k in item for k in ('x1', 'y1', 'x2', 'y2')),
                'Each rectangle requires x1,y1,x2,y2')
        values = [item[k] for k in ('x1', 'y1', 'x2', 'y2')]
        require(all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in values),
                'Rectangle bounds must be integer native saved-mask pixels')
        x1, y1, x2, y2 = map(int, values)
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
                'Rectangle must have positive area and fit the native saved mask')
        result.append(dict(x1=x1, y1=y1, x2=x2, y2=y2))
    return result


def _binary(array, shape, name):
    array = np.asarray(array)
    require(array.shape == shape and array.dtype.kind in 'biuf'
            and np.all((array == 0) | (array == 1)), f'{name} must be a matching binary mask')
    return array.astype(bool, copy=False)


def load_review(review_dir, verify=True):
    """Load a proposal, normally verifying its immutable content and inputs."""
    review_dir = Path(review_dir).expanduser().resolve()
    path = review_dir / PROPOSAL_FILENAME
    record = _json(path)
    require(record.get('schema_version') == 1
            and record.get('kind') == 'spatial_exclusion_review_proposal'
            and record.get('status') == 'needs_review', 'Not an unconfirmed spatial review proposal')
    require(type(record.get('image_width')) is int and type(record.get('image_height')) is int
            and record['image_width'] > 0 and record['image_height'] > 0,
            'Invalid review image dimensions')
    _rectangles(record.get('rectangles'), record['image_width'], record['image_height'])
    if verify:
        digest_path = review_dir / PROPOSAL_HASH_FILENAME
        require(digest_path.is_file() and digest_path.read_text(encoding='ascii').strip() == sha(path),
                'Review proposal changed; prepare a new review')
        _verify_sources(record.get('sources'))
        for key in ('optical', 'segmentation'):
            preview = record.get('previews', {}).get(key)
            require(isinstance(preview, dict) and isinstance(preview.get('path'), str),
                    'Review preview binding is missing')
            preview_path = review_dir / preview['path']
            require(preview_path.resolve().parent == review_dir and preview_path.is_file()
                    and sha(preview_path) == preview.get('sha256'),
                    'Review preview changed; prepare a new review')
    return record


def prepare_review(mask_path, valid_path, exclusion_path, mosaic_path, directory, *, automatic=True):
    """Save an immutable proposal and aligned previews, requiring human review.

    ``automatic`` enables a deterministic geometric suggestion only. Both its
    success and failure remain ``needs_review`` until the user confirms boxes.
    Existing bitmap exclusions remain protected; existing rectangles are editable.
    No final exclusion or classification is changed at this stage.
    """
    require(type(automatic) is bool, 'automatic must be a boolean')
    _source(mask_path)
    for optional in (valid_path, exclusion_path):
        if optional is not None:
            _source(optional)
    labels, valid, excluded, existing, sources, encoding = _load_sources(mask_path, valid_path, exclusion_path)
    height, width = labels.shape
    valid = _binary(valid, labels.shape, 'Valid support')
    excluded = _binary(excluded, labels.shape, 'Existing exclusions')
    existing = _rectangles(existing, width, height)
    sources['mosaic'] = _source(mosaic_path)
    rectangles = list(existing)
    automatic_proposal = {'status': 'disabled', 'rectangles': [],
                          'reason': 'No automatic suggestion requested; existing rectangles remain editable'}
    if automatic:
        try:
            automatic_proposal = automatic_central_exclusion(labels, valid)
        except CentralExclusionNeedsReview as error:
            automatic_proposal = dict(error.audit, reason=str(error))
        for rectangle in automatic_proposal.get('rectangles', []):
            if rectangle not in rectangles:
                rectangles.append(rectangle)
    base = None
    if 'base_exclusions' in sources:
        base = sources['base_exclusions']
    elif 'exclusions' in sources and Path(sources['exclusions']['path']).suffix.lower() == '.npy':
        base = sources['exclusions']
    require(base is not None or not excluded.any(),
            'Existing bitmap exclusions lack a source binding')
    _verify_sources(sources)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    review_id = uuid.uuid4().hex
    target = directory / review_id
    staging = Path(tempfile.mkdtemp(prefix='.review-staging-', dir=directory))
    try:
        stride = max(1, int(np.ceil(max(width / 1100., height / 1750.))))
        pw, ph = int(np.ceil(width / stride)), int(np.ceil(height / stride))
        with Image.open(sources['mosaic']['path']) as photo:
            require(photo.size == (width, height),
                    'Optical image dimensions must match the native saved-mask coordinates')
            preview = photo.convert('RGB').resize((pw, ph), Image.Resampling.LANCZOS)
            preview.save(staging / 'optical_preview.png')
        # Nearest-neighbour sampling is only a preview; native arrays stay intact.
        xs = np.minimum(width - 1, ((np.arange(pw) + .5) * width / pw).astype(int))
        ys = np.minimum(height - 1, ((np.arange(ph) + .5) * height / ph).astype(int))
        index = np.ix_(ys, xs)
        prediction = RGB[labels[index]].copy()
        # Protected legacy bitmap omissions are distinct from editable boxes.
        # Only this review preview is painted; source/native predictions and
        # the optical evidence remain byte-identical.
        prediction[valid[index] & excluded[index]] = (150, 150, 150)
        prediction[~valid[index]] = 255
        Image.fromarray(prediction).save(staging / 'segmentation_preview.png')
        previews = {key: {'path': name, 'sha256': sha(staging / name), 'width': pw, 'height': ph}
                    for key, name in (('optical', 'optical_preview.png'), ('segmentation', 'segmentation_preview.png'))}
        record = {
            'schema_version': 1, 'kind': 'spatial_exclusion_review_proposal',
            'review_id': review_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'needs_review', 'human_review_required': True,
            'image_width': width, 'image_height': height,
            'coordinates': 'x1,y1,x2,y2; half-open native saved-mask pixels; top-left origin',
            'sources': sources, 'input_encoding': encoding, 'previews': previews,
            'rectangles': rectangles, 'existing_rectangles': existing,
            'base_exclusion_mask': base, 'automatic_proposal': automatic_proposal,
            'preview_semantics': {
                'optical': 'Unpainted optical evidence; editable rectangles are an interface overlay',
                'segmentation': 'Native predicted colours; white unsupported; neutral grey protected base-bitmap exclusions',
                'protected_base_exclusion_rgb': [150, 150, 150],
                'protected_base_exclusions_editable': False,
                'protected_base_excluded_valid_px': int(np.count_nonzero(valid & excluded)),
                'editable_rectangles_baked_into_preview': False,
            },
            'interpretation': 'Geometric suggestions only. No chemical identity or accepted physical layer inferred.',
            'next_action': 'Review, edit or remove rectangles on both previews, then explicitly confirm. No exclusion applied yet.',
        }
        _write_json(staging / PROPOSAL_FILENAME, record)
        (staging / PROPOSAL_HASH_FILENAME).write_text(sha(staging / PROPOSAL_FILENAME) + '\n', encoding='ascii')
        _verify_sources(sources)
        require(not target.exists(), 'Review output already exists')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'review_id': review_id, 'review_dir': str(target), 'proposal_path': str(target / PROPOSAL_FILENAME),
            'status': 'needs_review', 'rectangles': rectangles, 'image_width': width, 'image_height': height,
            'preview_paths': {key: str(target / preview['path']) for key, preview in previews.items()},
            'proposal': record, 'record': record}


def confirm_review(review_dir, rectangles, output_dir):
    """Persist explicitly human-confirmed boxes in a new output directory.

    Passing an empty list is an explicit retain-all-rectangles decision; an
    existing immutable bitmap exclusion is preserved. This function must be
    called only after the caller receives the user's confirmation action.
    It neither accepts a model recommendation nor edits the proposal.
    """
    review_dir = Path(review_dir).expanduser().resolve()
    record = load_review(review_dir, verify=True)
    width, height = record['image_width'], record['image_height']
    rectangles = _rectangles(rectangles, width, height)
    output_dir = Path(output_dir).expanduser().resolve()
    require(not output_dir.exists(), 'Confirmation output exists; choose a new directory')
    sources = record['sources']
    labels, valid, excluded, _, reloaded_sources, _ = _load_sources(
        sources['mask']['path'], sources.get('valid_support', {}).get('path'),
        sources.get('exclusions', {}).get('path'))
    require(labels.shape == (height, width), 'Source dimensions changed since review')
    _binary(valid, labels.shape, 'Valid support')
    combined = _binary(excluded, labels.shape, 'Existing exclusions').copy()
    for rectangle in rectangles:
        combined[rectangle['y1']:rectangle['y2'], rectangle['x1']:rectangle['x2']] = True
    # Revalidate all sources after decoding, including indirect bitmap sources.
    _verify_sources(sources)
    for name, binding in reloaded_sources.items():
        require(name in sources and binding == sources[name], 'Reloaded source bindings disagree with proposal')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.confirmation-staging-', dir=output_dir.parent))
    try:
        mask_name = 'reviewed_exclusion_mask.npy'
        np.save(staging / mask_name, combined, allow_pickle=False)
        exclusion_record = {
            'image_width': width, 'image_height': height, 'excluded_rectangles': rectangles,
            'base_exclusion_mask': record.get('base_exclusion_mask'),
            'review': {'review_id': record['review_id'], 'human_confirmed': True,
                       'proposal_path': str(review_dir / PROPOSAL_FILENAME),
                       'proposal_sha256': sha(review_dir / PROPOSAL_FILENAME)},
        }
        if exclusion_record['base_exclusion_mask'] is None:
            del exclusion_record['base_exclusion_mask']
        _write_json(staging / 'confirmed_exclusions.json', exclusion_record)
        confirmation = {
            'schema_version': 1, 'kind': 'spatial_exclusion_human_confirmation',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'review_id': record['review_id'],
            'human_confirmed': True, 'status': 'confirmed', 'sources': sources,
            'proposal': {'path': str(review_dir / PROPOSAL_FILENAME), 'sha256': sha(review_dir / PROPOSAL_FILENAME)},
            'rectangles': rectangles, 'proposal_rectangles': record['rectangles'],
            'human_changed_proposal': rectangles != record['rectangles'],
            'image_width': width, 'image_height': height, 'coordinates': record['coordinates'],
            'base_exclusion_mask': record.get('base_exclusion_mask'),
            'outputs': [{'path': name, 'sha256': sha(staging / name)}
                        for name in ('confirmed_exclusions.json', mask_name)],
            'mask_semantics': 'Boolean union of protected base bitmap and human-confirmed rectangles; valid support remains separate',
        }
        _write_json(staging / 'confirmation.json', confirmation)
        _verify_sources(sources)
        require(not output_dir.exists(), 'Confirmation output appeared; choose a new directory')
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'exclusion_path': str(output_dir / 'confirmed_exclusions.json'),
            'combined_mask_path': str(output_dir / 'reviewed_exclusion_mask.npy'),
            'confirmation_path': str(output_dir / 'confirmation.json'),
            'rectangles': rectangles, 'record': confirmation}
