"""Prepare bounded, source-bound DD layer/spatial jobs without loading a model.

Preparation validates inputs and builds fixed-script argument arrays. It never
starts a process, calls a model, edits a teacher configuration or overwrites a
result. Relative teacher-config paths follow the pipeline CLI's repository cwd.
Spatial fractions/figures remain the existing spatial_maps implementation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys

MAX_JSON = 8 * 1024 * 1024
_UNSET = object()
SAFE_NAME = re.compile(r'^[^<>:"/\\|?*\x00-\x1f]{1,120}$')
GLOBAL_KEYS = {
    'work_root', 'mosaic_agent', 'weights', 'samples', 'workflow_validation',
    'tile', 'overlap', 'batch', 'threads', 'border_trim_px', 'filler', 'votes',
    'workers', 'no_ai', 'no_repair', 'no_refine', 'device', 'seed', 'full_res',
    'region_scale_div', 'min_region_pct', 'max_regions', 'coarse_close_frac',
    'refine_rounds', 'refine_min_pct', 'map_preview_div', 'min_confidence',
    'exclusion_rule', 'sensitivity_alarm_pct', 'inference_geometry', 'spatial_maps',
}
SAMPLE_KEYS = {'name', 'mosaic_in'}


class AnalysisPlanError(ValueError):
    """Actionable preparation or result-validation error, before model execution."""


def _require(condition, message):
    if not condition:
        raise AnalysisPlanError(message)


def _read_json(path):
    path = Path(path)
    _require(path.is_file(), f'找不到记录 / Missing record: {path}')
    _require(path.stat().st_size <= MAX_JSON, 'JSON记录过大 / JSON record is too large')
    def invalid(value):
        raise AnalysisPlanError('JSON含非有限数字 / Nonfinite JSON number')
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisPlanError(f'无法读取JSON记录 / Invalid JSON record: {path}') from error
    _require(isinstance(value, dict), '需要JSON对象 / Expected a JSON object')
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file(path, base=None):
    _require(isinstance(path, (str, Path)) and bool(str(path)), '缺少输入路径 / Missing input path')
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = (Path(base) if base is not None else Path.cwd()) / value
    value = value.resolve()
    _require(value.is_file(), f'找不到输入文件 / Missing input file: {value}')
    _require(value.name != '.env', '此流程不读取凭据 / This workflow does not read credentials')
    return value


def _safe_sample(name):
    _require(isinstance(name, str) and SAFE_NAME.fullmatch(name) is not None
             and name not in ('.', '..') and not name.endswith((' ', '.')),
             '样品名称不能包含路径或特殊字符 / Sample name must be a safe filename label')
    _require(name.split('.')[0].upper() not in {
        'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)],
        *[f'LPT{i}' for i in range(1, 10)]},
        '样品名称为Windows保留名称 / Sample name is reserved on Windows')
    return name


def _new_output(output_dir):
    output = Path(output_dir).expanduser().resolve()
    _require(not output.exists(), '输出目录已存在，请使用新目录保留旧结果 / Output exists; choose a new directory')
    return output


def _records(paths):
    return [{'path': str(p), 'sha256': _sha(p)} for p in dict.fromkeys(paths)]


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def prepare_layers(config_path, output_dir, repo, expected_mosaic=None):
    """Bind an explicit teacher configuration to a new offline layer-analysis job.

    The teacher must explicitly confirm sample/model domain and geometry. The
    selected checkpoint must match the supplied original0409 asset manifest.
    If a current stitched mosaic is supplied, only that exact single sample is
    accepted. No teacher/sample parameters are inferred from image colour.
    """
    repo = Path(repo).resolve()
    config_path = _file(config_path)
    _require(config_path.suffix.lower() == '.json', '请选择老师的JSON配置 / Select the teacher JSON config')
    config_sha = _sha(config_path)
    config = _read_json(config_path)
    _require(not (set(config) - GLOBAL_KEYS), '老师配置包含未支持字段，请核对 / Unsupported teacher configuration fields')
    confirmation = config.get('workflow_validation')
    _require(isinstance(confirmation, dict)
             and confirmation.get('sample_model_confirmed') is True
             and confirmation.get('geometry_confirmed') is True,
             '需要老师确认样品/0409模型适用性和推理尺度：workflow_validation中的sample_model_confirmed、geometry_confirmed必须为true / Teacher confirmation of sample/model domain and geometry is required')
    asset_path = repo / 'data/demo/assets.json'
    _require(asset_path.is_file(), '缺少原始0409资源清单data/demo/assets.json，请老师恢复可信清单；尚未加载模型 / Restore the original0409 asset manifest; no model loaded')
    asset_sha = _sha(asset_path)
    assets = _read_json(asset_path)
    record = assets.get('weights')
    _require(type(assets.get('schema_version')) is int and assets['schema_version'] == 1 and isinstance(record, dict)
             and record.get('path') == 'weights/model_0409_all.pth'
             and isinstance(record.get('sha256'), str)
             and re.fullmatch(r'[0-9a-fA-F]{64}', record['sha256']) is not None,
             '0409模型资源清单无效 / Invalid original0409 checkpoint manifest')
    weights = _file(config.get('weights'), repo)
    _require(_sha(weights) == record['sha256'].lower(), '所选权重不是清单绑定的0409原模型，请老师核对 / Checkpoint does not match the original0409 asset binding')
    samples = config.get('samples')
    _require(isinstance(samples, list) and 1 <= len(samples) <= 100, '老师配置需要1至100个明确样品 / Teacher config needs1–100 explicit samples')
    names, generated_samples, mosaics = set(), [], []
    for sample in samples:
        _require(isinstance(sample, dict) and not (set(sample) - SAMPLE_KEYS),
                 '样品只允许name和mosaic_in，不能覆盖模型或执行策略 / Samples may only bind name and mosaic_in')
        name = _safe_sample(sample.get('name'))
        _require(name.casefold() not in names, '样品名称重复 / Duplicate sample name')
        names.add(name.casefold())
        mosaic = _file(sample.get('mosaic_in'), repo)
        _require(mosaic.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'},
                 '层数输入必须为拼接图像 / Layer input must be a mosaic image')
        from PIL import Image
        try:
            with Image.open(mosaic) as image:
                _require(image.width > 0 and image.height > 0, '拼接图像尺寸无效 / Invalid mosaic dimensions')
        except (OSError, ValueError) as error:
            raise AnalysisPlanError('无法读取拼接图像 / Cannot read mosaic image') from error
        mosaics.append(mosaic)
        generated_samples.append({'name': name, 'mosaic_in': str(mosaic)})
    if expected_mosaic is not None:
        expected = _file(expected_mosaic)
        _require(len(mosaics) == 1 and mosaics[0] == expected,
                 '老师配置与当前选定拼接结果不一致，请重新绑定 / Teacher config differs from the selected stitched mosaic')
    script = _file(repo / 'flakepipeline/run.py')
    output = _new_output(output_dir)
    derived = {k: value for k, value in config.items() if k != 'workflow_validation'}
    derived.update(work_root=str(output / 'run'), mosaic_agent=str(repo), weights=str(weights),
                   no_ai=True, no_repair=True, samples=generated_samples)
    # Spatial export is an explicit later workflow step, never silently mixed in.
    spatial = derived.get('spatial_maps', {})
    _require(isinstance(spatial, dict), 'spatial_maps必须是对象 / spatial_maps must be an object')
    derived['spatial_maps'] = {**spatial, 'enabled': False}
    implementation = sorted((repo / 'flakepipeline').rglob('*.py'))
    input_records = _records([config_path, asset_path.resolve(), weights, *mosaics, script, *implementation])
    digests = {item['path']: item['sha256'] for item in input_records}
    _require(digests[str(config_path)] == config_sha
             and digests[str(asset_path.resolve())] == asset_sha
             and digests[str(weights)] == record['sha256'].lower(),
             '准备过程中输入已改变，请重试 / Inputs changed while preparing; retry')
    output.mkdir(parents=True, exist_ok=False)
    control = output / 'control'
    control.mkdir()
    generated_config = control / 'layers_config.json'
    _write_new(generated_config, derived)
    files = [*input_records, *_records([generated_config])]
    plan = {'kind': 'layers', 'argv': [sys.executable, str(script), '--config', str(generated_config), '--no-ai', '--no-repair'],
            'output_dir': str(output), 'files': files,
            'source_binding': {'teacher_config': str(config_path), 'teacher_config_sha256': config_sha,
                               'sample_model_confirmed': True, 'geometry_confirmed': True,
                               'weights': str(weights), 'weights_sha256': record['sha256'].lower(),
                               'mosaics': [{'sample': s['name'], 'path': s['mosaic_in'], 'sha256': _sha(s['mosaic_in'])} for s in generated_samples],
                               'no_ai': True, 'no_repair': True,
                               'predictions_are_measurements': False},
            'expected': {'report_path': str(output / 'run/report.json'),
                         'samples': [{'sample': s['name'], 'manifest_path': str(output / 'run' / s['name'] / 'manifest.json')} for s in generated_samples]}}
    return plan


def _load_npy(path, label):
    import numpy as np
    try:
        array = np.load(path, allow_pickle=False, mmap_mode='r')
    except (OSError, ValueError, TypeError) as error:
        raise AnalysisPlanError(f'{label}不是安全数值NPY / Invalid numeric NPY: {path}') from error
    _require(isinstance(array, np.ndarray) and array.ndim == 2, f'{label}必须为二维数组 / Expected a2D array')
    return array


def _check_values(array, allowed, label):
    import numpy as np
    rows = max(1, (2 * 1024 * 1024) // max(1, array.shape[1] * array.dtype.itemsize))
    for start in range(0, array.shape[0], rows):
        _require(np.all(np.isin(array[start:start + rows], allowed)),
                 f'{label}含未知像素值 / Unknown pixel values in {label}')


def _mask_sources(mask_path, valid_path, exclusion_path, whole_canvas_supported):
    """Validate existing encoding/support using the spatial core's exact conventions.

    NPY values are checked in bounded memory without loading a model or creating
    canonical copies. Exact RGB/P-mode decoding uses the existing core directly.
    No counting or normalization is reimplemented here.
    """
    import numpy as np
    mask = _file(mask_path)
    if mask.suffix.lower() == '.npy':
        labels = _load_npy(mask, 'Native layer mask')
        _require(np.issubdtype(labels.dtype, np.integer), '原生mask必须是整数类别 / Native mask needs integer class IDs')
        _check_values(labels, [0, 1, 2, 3], 'Native layer mask')
        shape = labels.shape
        encoding = 'MosaicAgent native IDs:0 substrate,1=2L,2=1L,3=TL'
    else:
        _require(mask.suffix.lower() == '.png', '请选择原生NPY或精确颜色PNG mask / Use native NPY or exact-colour PNG mask')
        from flakepipeline.spatial_maps.stats import load_labels
        try:
            labels = load_labels(mask)
        except (OSError, ValueError) as error:
            raise AnalysisPlanError(str(error)) from error
        shape, encoding = labels.shape, 'Exact RGB colours, independent of palette indices'
    _require(shape[0] >= 15 and shape[1] >= 8, 'mask过小，不能用于8×15和6×11网格 / Mask is too small for the requested grids')
    sources = {'mask': mask}
    if valid_path is None:
        _require(whole_canvas_supported is True,
                 '缺少有效扫描范围mask，请明确确认整个画布均被扫描；不能把空白补边自动视作扫描范围 / Provide valid support or explicitly confirm the whole canvas is supported')
    else:
        valid = _file(valid_path)
        _require(valid.suffix.lower() == '.npy', '扫描范围必须为原生NPY / Valid support must be native NPY')
        array = _load_npy(valid, 'Valid support')
        _require(array.shape == shape and array.dtype.kind in 'biuf', '扫描范围尺寸或类型不匹配 / Invalid support shape or dtype')
        _check_values(array, [0, 1], 'Valid support')
        sources['valid_support'] = valid
    if exclusion_path is not None:
        exclusion = _file(exclusion_path)
        if exclusion.suffix.lower() == '.json':
            record = _read_json(exclusion)
            _require(type(record.get('image_height')) is int and type(record.get('image_width')) is int
                     and (record['image_height'], record['image_width']) == shape,
                     '排除区尺寸与原始mask不一致 / Exclusion metadata dimensions differ')
            rectangles = record.get('excluded_rectangles')
            _require(isinstance(rectangles, list) and len(rectangles) <= 100000,
                     '排除区记录无效 / Invalid exclusion rectangle list')
            for rect in rectangles:
                _require(isinstance(rect, dict) and all(type(rect.get(k)) is int for k in ('x1', 'y1', 'x2', 'y2')),
                         '排除区必须为整数像素坐标 / Exclusions require integer pixel coordinates')
                _require(0 <= rect['x1'] < rect['x2'] <= shape[1] and 0 <= rect['y1'] < rect['y2'] <= shape[0],
                         '排除区超出mask / Exclusion rectangle outside mask')
        else:
            _require(exclusion.suffix.lower() == '.npy', '排除区应为NPY或矩形JSON / Exclusions must be NPY or rectangle JSON')
            array = _load_npy(exclusion, 'Exclusion mask')
            _require(array.shape == shape and array.dtype.kind in 'biuf', '排除区尺寸或类型不匹配 / Invalid exclusion shape or dtype')
            _check_values(array, [0, 1], 'Exclusion mask')
        sources['exclusions'] = exclusion
    return sources, shape, encoding


def prepare_spatial(mask_path, valid_path, exclusion_path, sample, output_dir, repo,
                    whole_canvas_supported=False):
    """Plan both saved-mask figure versions, with local MoS2 normalization unchanged."""
    repo = Path(repo).resolve()
    # Import only existing mask validation; never segmentation or torch.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sample = _safe_sample(sample)
    sources, shape, encoding = _mask_sources(mask_path, valid_path, exclusion_path, whole_canvas_supported)
    output = _new_output(output_dir)
    script = _file(repo / 'tools/layer_spatial_maps.py')
    argv = [sys.executable, str(script), '--mask', str(sources['mask'])]
    for key, flag in [('valid_support', '--valid-mask'), ('exclusions', '--exclusions')]:
        if key in sources:
            argv.extend([flag, str(sources[key])])
    argv.extend(['--sample', sample, '--output', str(output), '--heatmap-grid', '8', '15', '--combined-grid', '6', '11'])
    binding = {key: {'path': str(path), 'sha256': _sha(path)} for key, path in sources.items()}
    return {'kind': 'spatial', 'argv': argv, 'output_dir': str(output),
            'files': _records([*sources.values(), script, *sorted((repo / 'flakepipeline/spatial_maps').glob('*.py'))]),
            'source_binding': {'sources': binding, 'input_encoding': encoding,
                               'image_shape_hw': list(shape), 'whole_canvas_supported': valid_path is None and whole_canvas_supported is True,
                               'coordinates': 'Mask pixels; no physical millimetre scale inferred',
                               'composition_denominator': 'Retained classified MoS2 area inside each analysis cell'},
            'expected': {'manifest_path': str(output / 'spatial_manifest.json'), 'kind': 'layer_spatial_maps', 'sample': sample,
                         'grids': {'heatmaps': [8, 15], 'combined': [6, 11]}}}


def _verify_sources(plan):
    _require(isinstance(plan.get('files'), list) and bool(plan['files']), '缺少输入绑定 / Missing input bindings')
    for record in plan['files']:
        path = _file(record.get('path'))
        _require(_sha(path) == record.get('sha256'), f'输入已改变，请重新准备 / Input changed; prepare again: {path}')


def _checked_output(path, root, expected_sha=_UNSET):
    value = _file(path)
    _require(_inside(value, root), '输出记录越过本次目录 / Output escaped this job directory')
    if expected_sha is not _UNSET:
        _require(isinstance(expected_sha, str)
                 and re.fullmatch(r'[0-9a-f]{64}', expected_sha) is not None
                 and _sha(value) == expected_sha,
                 f'结果文件校验不一致 / Output hash mismatch: {value}')
    return value


def validate_plan_result(plan):
    """Validate saved completion evidence; exit0 alone never validates a result."""
    _verify_sources(plan)
    root = Path(plan['output_dir']).resolve()
    expected = plan['expected']
    if plan['kind'] == 'spatial':
        path = _checked_output(expected['manifest_path'], root)
        record = _read_json(path)
        _require(record.get('schema_version') == 1 and record.get('kind') == 'layer_spatial_maps'
                 and record.get('sample') == expected['sample'] and record.get('grids') == expected['grids'],
                 '空间统计清单不对应本次请求 / Spatial manifest does not match this job')
        _require(record.get('sources') == plan['source_binding']['sources'], '空间统计来源绑定不一致 / Spatial input source bindings differ')
        outputs = record.get('outputs')
        _require(isinstance(outputs, list) and 1 <= len(outputs) <= 128, '空间统计产物清单无效 / Invalid spatial output inventory')
        artifacts = [str(path)]
        seen = set()
        for item in outputs:
            _require(isinstance(item, dict) and isinstance(item.get('path'), str), '空间统计文件记录无效 / Invalid output record')
            relative = Path(item['path'])
            _require(not relative.is_absolute() and '\\' not in item['path'] and '..' not in relative.parts,
                     '空间统计文件路径无效 / Unsafe spatial output path')
            target = _checked_output(path.parent / relative, root, item.get('sha256'))
            _require(target not in seen, '产物记录重复 / Duplicate output record')
            seen.add(target)
            artifacts.append(str(target))
        _require(any(Path(p).suffix.lower() == '.svg' for p in artifacts)
                 and any(Path(p).suffix.lower() == '.csv' for p in artifacts),
                 '缺少空间分布SVG或统计CSV / Missing spatial SVG or statistics CSV')
        return {'status': 'validated', 'output_path': str(path), 'artifacts': artifacts,
                'review_required': True, 'coordinates': 'Mask pixels; no physical scale inferred'}
    _require(plan['kind'] == 'layers', '未知分析步骤 / Unknown analysis kind')
    report_path = _checked_output(expected['report_path'], root)
    report = _read_json(report_path)
    rows = report.get('samples')
    names = [item['sample'] for item in expected['samples']]
    _require(isinstance(rows, list) and sorted(r.get('sample') for r in rows) == sorted(names),
             '层数分析汇总样品不匹配 / Layer report sample inventory differs')
    samples, artifacts, quality_flags = [], [str(report_path)], []
    for item in expected['samples']:
        name = item['sample']
        manifest_path = _checked_output(item['manifest_path'], root)
        manifest = _read_json(manifest_path)
        _require(manifest.get('sample') == name, '样品清单不匹配 / Layer sample manifest differs')
        stages = manifest.get('stages', {})
        _require(isinstance(stages, dict) and all(k in stages for k in ('stitch', 'segment', 'audit', 'regions', 'stats')),
                 '层数分析步骤未完成 / Layer pipeline stages are incomplete')
        source = next(m for m in plan['source_binding']['mosaics'] if m['sample'] == name)
        stitch = stages['stitch'].get('result', {})
        _require(stitch.get('source') and Path(stitch['source']).resolve() == Path(source['path']).resolve(),
                 '层数分析使用的拼接图与绑定来源不一致 / Layer pipeline used a different mosaic source')
        stitched_copy = _checked_output(stitch.get('mosaic'), root, source['sha256'])
        segment = stages['segment'].get('result', {})
        _require(segment.get('mask') and segment.get('valid_mask'), '缺少原生类别或扫描范围mask / Missing native mask or valid support')
        files = {}
        for field in ('mask', 'valid_mask'):
            value = _checked_output(segment[field], root)
            _require(value.suffix.lower() == '.npy', '层数结果必须保留原生NPY / Native layer output must be NPY')
            expected_hash = stages['segment'].get('output_sha256', {}).get(str(value))
            _require(isinstance(expected_hash, str), '层数产物没有文件校验值 / Missing layer output hash')
            _checked_output(value, root, expected_hash)
            files[field] = str(value)
        exclusion = stages['regions'].get('result', {}).get('exclusion_mask')
        if exclusion:
            exclusion = str(_checked_output(exclusion, root))
        _, mask_shape, _ = _mask_sources(files['mask'], files['valid_mask'], exclusion, False)
        from PIL import Image
        with Image.open(stitched_copy) as image:
            _require(mask_shape == (image.height, image.width),
                     '原生mask尺寸与所绑定拼接图不一致 / Native mask dimensions differ from the bound mosaic')
        for stage_name, stage in stages.items():
            _require(isinstance(stage, dict), '分析步骤记录无效 / Invalid stage record')
            for output, sha in stage.get('output_sha256', {}).items():
                _checked_output(output, root, sha)
            if stage.get('ok') is not True or stage.get('escalate'):
                quality_flags.append({'sample': name, 'stage': stage_name,
                                      'reason': str(stage.get('escalate_reason', 'Needs review'))[:1000]})
        _require(stages['stats'].get('ok') is True, '面积统计未成功完成 / Area statistics did not complete successfully')
        samples.append({'sample': name, 'manifest': str(manifest_path), **files,
                        'exclusion_mask': exclusion})
        artifacts.extend([str(manifest_path), *files.values()])
        if exclusion:
            artifacts.append(exclusion)
    return {'status': 'validated', 'output_path': str(report_path), 'artifacts': artifacts,
            'samples': samples, 'review_required': True, 'quality_flags': quality_flags,
            'interpretation': 'Saved model predictions require human review; no independent accuracy claim'}
