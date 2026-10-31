"""Offline planning/integrity checks; no checkpoints, model calls or subprocesses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location('workflow_analysis_under_test', Path(__file__).with_name('workflow_analysis.py'))
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WorkflowAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / 'repo with spaces'
        for folder in ('flakepipeline', 'tools', 'weights', 'data/demo'):
            (self.repo / folder).mkdir(parents=True, exist_ok=True)
        for script in ('flakepipeline/run.py', 'tools/layer_spatial_maps.py'):
            (self.repo / script).write_text('# Fixed CLI fixture; never executed\n')
        self.weights = self.repo / 'weights/model_0409_all.pth'
        self.weights.write_bytes(b'checkpoint fixture, not a model')
        (self.repo / 'data/demo/assets.json').write_text(json.dumps({
            'schema_version': 1, 'weights': {'path': 'weights/model_0409_all.pth', 'sha256': digest(self.weights)}}))
        self.mosaic = self.repo / 'mosaic ; data.png'
        Image.new('RGB', (32, 48), (180, 140, 120)).save(self.mosaic)
        self.config = self.root / 'teacher.json'
        self.cfg = {'weights': 'weights/model_0409_all.pth', 'work_root': 'original output',
                    'mosaic_agent': '.', 'tile': 512, 'overlap': 64,
                    'workflow_validation': {'sample_model_confirmed': True, 'geometry_confirmed': True},
                    'samples': [{'name': 'sample-A', 'mosaic_in': self.mosaic.name}]}
        self.write_config()
        self.mask = self.root / 'mask.npy'
        self.valid = self.root / 'valid.npy'
        self.exclude = self.root / 'excluded.npy'
        labels = np.indices((30, 16)).sum(axis=0).astype(np.uint8) % 4
        np.save(self.mask, labels)
        np.save(self.valid, np.ones_like(labels, dtype=bool))
        np.save(self.exclude, np.zeros_like(labels, dtype=bool))

    def write_config(self):
        self.config.write_text(json.dumps(self.cfg))

    def layers(self, **kw):
        return W.prepare_layers(self.config, self.root / 'new layers', self.repo, **kw)

    def spatial(self, **kw):
        defaults = dict(mask_path=self.mask, valid_path=self.valid, exclusion_path=self.exclude,
                        sample='sample-A', output_dir=self.root / 'new spatial', repo=self.repo)
        defaults.update(kw)
        return W.prepare_spatial(**defaults)

    def test_layers_fixed_argv_originals_unchanged_and_offline_enforced(self):
        torch_before = sys.modules.get('torch')
        before = self.config.read_bytes()
        self.cfg.update(no_ai=False, no_repair=False, spatial_maps={'enabled': True})
        self.write_config()
        before = self.config.read_bytes()
        with patch('subprocess.run', side_effect=AssertionError('No child process')), \
                patch('subprocess.Popen', side_effect=AssertionError('No child process')):
            plan = self.layers(expected_mosaic=self.mosaic)
        generated = json.loads(Path(plan['argv'][3]).read_text())
        self.assertEqual(plan['argv'][:2], [sys.executable, str(self.repo / 'flakepipeline/run.py')])
        self.assertEqual(plan['argv'][-2:], ['--no-ai', '--no-repair'])
        self.assertTrue(generated['no_ai'] and generated['no_repair'])
        self.assertFalse(generated['spatial_maps']['enabled'])
        self.assertEqual(generated['samples'][0]['mosaic_in'], str(self.mosaic))
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.repo / 'original output').exists())
        self.assertIs(sys.modules.get('torch'), torch_before)
        W._verify_sources(plan)

    def test_missing_teacher_confirmation_writes_nothing(self):
        del self.cfg['workflow_validation']
        self.write_config()
        with self.assertRaisesRegex(W.AnalysisPlanError, '老师确认'):
            self.layers()
        self.assertFalse((self.root / 'new layers').exists())

    def test_checkpoint_hash_and_asset_identity_fail_closed(self):
        self.weights.write_bytes(b'changed weights')
        with self.assertRaisesRegex(W.AnalysisPlanError, '0409'):
            self.layers()
        self.assertFalse((self.root / 'new layers').exists())

    def test_missing_original_asset_manifest_is_not_synthesized(self):
        (self.repo / 'data/demo/assets.json').unlink()
        with self.assertRaisesRegex(W.AnalysisPlanError, '原始0409资源清单'):
            self.layers()
        self.assertFalse((self.repo / 'data/demo/assets.json').exists())

    def test_duplicate_sample_names_and_sample_policy_override_rejected(self):
        self.cfg['samples'].append({'name': 'SAMPLE-a', 'mosaic_in': self.mosaic.name})
        self.write_config()
        with self.assertRaisesRegex(W.AnalysisPlanError, '重复'):
            self.layers()
        self.cfg['samples'] = [{'name': 'sample-A', 'mosaic_in': self.mosaic.name, 'no_ai': False}]
        self.write_config()
        with self.assertRaisesRegex(W.AnalysisPlanError, '不能覆盖'):
            self.layers()

    def test_selected_mosaic_mismatch_and_existing_output_rejected(self):
        other = self.repo / 'other.png'
        Image.new('RGB', (20, 20)).save(other)
        with self.assertRaisesRegex(W.AnalysisPlanError, '当前选定'):
            self.layers(expected_mosaic=other)
        (self.root / 'new layers').mkdir()
        with self.assertRaisesRegex(W.AnalysisPlanError, '输出目录已存在'):
            self.layers()

    def test_spatial_native_class_order_fixed_commands_and_no_precreated_output(self):
        plan = self.spatial()
        self.assertEqual(plan['kind'], 'spatial')
        self.assertEqual(plan['argv'][1], str(self.repo / 'tools/layer_spatial_maps.py'))
        self.assertEqual(plan['argv'][-7:], ['--output', str(self.root / 'new spatial'), '--heatmap-grid', '8', '15', '--combined-grid', '6', '11'][-7:])
        self.assertFalse(Path(plan['output_dir']).exists())
        self.assertIn('1=2L', plan['source_binding']['input_encoding'])
        self.assertIn('no physical', plan['source_binding']['coordinates'])
        W._verify_sources(plan)

    def test_spatial_support_requires_explicit_whole_canvas_confirmation(self):
        with self.assertRaisesRegex(W.AnalysisPlanError, '明确确认'):
            self.spatial(valid_path=None)
        plan = self.spatial(valid_path=None, whole_canvas_supported=True)
        self.assertNotIn('--valid-mask', plan['argv'])
        self.assertTrue(plan['source_binding']['whole_canvas_supported'])

    def test_native_unknown_class_or_object_and_mismatched_support_rejected(self):
        np.save(self.mask, np.full((30, 16), 4, dtype=np.uint8))
        with self.assertRaisesRegex(W.AnalysisPlanError, '未知像素值'):
            self.spatial()
        np.save(self.mask, np.zeros((30, 16), dtype=np.uint8))
        np.save(self.valid, np.ones((29, 16), dtype=bool))
        with self.assertRaisesRegex(W.AnalysisPlanError, '不匹配'):
            self.spatial()
        np.save(self.mask, np.array([['unsafe']], dtype=object))
        with self.assertRaisesRegex(W.AnalysisPlanError, '数值NPY'):
            self.spatial()

    def test_exact_colour_png_uses_existing_decoder_and_rejects_nearest_colour(self):
        png = self.root / 'mask.png'
        Image.new('RGB', (16, 30), (0, 128, 0)).save(png)
        self.spatial(mask_path=png)
        Image.new('RGB', (16, 30), (0, 127, 0)).save(png)
        with self.assertRaisesRegex(W.AnalysisPlanError, 'Unknown mask colours'):
            self.spatial(mask_path=png)

    def test_exclusion_rectangles_use_original_dimensions_and_integer_bounds(self):
        rect = self.root / 'rectangles.json'
        rect.write_text(json.dumps({'image_height': 30, 'image_width': 16,
                                    'excluded_rectangles': [{'x1': 2, 'y1': 3, 'x2': 9, 'y2': 25}]}))
        self.spatial(exclusion_path=rect)
        record = json.loads(rect.read_text())
        record['excluded_rectangles'][0]['x1'] = 2.5
        rect.write_text(json.dumps(record))
        with self.assertRaisesRegex(W.AnalysisPlanError, '整数像素'):
            self.spatial(exclusion_path=rect)

    def test_changed_source_refuses_completion_even_with_exit0(self):
        plan = self.spatial()
        np.save(self.mask, np.ones((30, 16), dtype=np.uint8))
        with self.assertRaisesRegex(W.AnalysisPlanError, '输入已改变'):
            W.validate_plan_result(plan)

    def spatial_manifest(self, plan):
        out = Path(plan['output_dir'])
        out.mkdir()
        (out / 'map.svg').write_text('<svg/>')
        (out / 'counts.csv').write_text('class,count\n1L,20\n')
        record = {'schema_version': 1, 'kind': 'layer_spatial_maps', 'sample': 'sample-A',
                  'sources': plan['source_binding']['sources'], 'grids': plan['expected']['grids'],
                  'outputs': [{'path': n, 'sha256': digest(out / n)} for n in ['map.svg', 'counts.csv']]}
        path = out / 'spatial_manifest.json'
        path.write_text(json.dumps(record))
        return path, record

    def test_spatial_completion_binds_sources_hashes_and_required_outputs(self):
        plan = self.spatial()
        path, record = self.spatial_manifest(plan)
        result = W.validate_plan_result(plan)
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(len(result['artifacts']), 3)
        self.assertTrue(result['review_required'])
        (path.parent / 'counts.csv').write_text('edited')
        with self.assertRaisesRegex(W.AnalysisPlanError, '校验不一致'):
            W.validate_plan_result(plan)

    def test_spatial_completion_rejects_missing_hash_and_path_escape(self):
        plan = self.spatial()
        path, record = self.spatial_manifest(plan)
        del record['outputs'][0]['sha256']
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(W.AnalysisPlanError, '校验不一致'):
            W.validate_plan_result(plan)
        record['outputs'][0] = {'path': '../outside.svg', 'sha256': '0' * 64}
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(W.AnalysisPlanError, '路径无效'):
            W.validate_plan_result(plan)

    def test_real_existing_spatial_export_roundtrip_uses_unchanged_counting(self):
        plan = self.spatial(repo=REPO)
        from flakepipeline.spatial_maps import export_spatial_maps
        original = {p: p.read_bytes() for p in [self.mask, self.valid, self.exclude]}
        result = export_spatial_maps(self.mask, self.valid, self.exclude,
                                     Path(plan['output_dir']), 'sample-A')
        verified = W.validate_plan_result(plan)
        self.assertEqual(verified['output_path'], result['manifest'])
        manifest = json.loads(Path(result['manifest']).read_text())
        self.assertEqual(manifest['global']['retained_canvas_px'], 480)
        self.assertEqual(manifest['coordinates'],
                         'Mask pixels; top-left origin; analysis tiles, not camera frames; no physical scale inferred')
        self.assertTrue(any('combined_stacks' in path or 'stacks' in path for path in verified['artifacts']))
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_layer_completion_requires_native_hash_bound_masks_and_all_stages(self):
        plan = self.layers()
        root = Path(plan['output_dir'])
        sample_root = root / 'run/sample-A'
        sample_root.mkdir(parents=True)
        for name, source in [('mask.npy', self.mask), ('valid.npy', self.valid), ('excluded.npy', self.exclude)]:
            (sample_root / name).write_bytes(source.read_bytes())
        mask, valid, exclusion = (sample_root / n for n in ['mask.npy', 'valid.npy', 'excluded.npy'])
        np.save(mask, np.indices((48, 32)).sum(axis=0).astype(np.uint8) % 4)
        np.save(valid, np.ones((48, 32), dtype=bool))
        np.save(exclusion, np.zeros((48, 32), dtype=bool))
        mosaic_copy = sample_root / 'mosaic.png'
        mosaic_copy.write_bytes(self.mosaic.read_bytes())
        stages = {name: {'ok': True, 'output_sha256': {}, 'result': {}} for name in ['stitch', 'segment', 'audit', 'regions', 'stats']}
        stages['stitch'].update(result={'mosaic': str(mosaic_copy), 'source': str(self.mosaic)},
                                output_sha256={str(mosaic_copy): digest(mosaic_copy)})
        stages['segment'].update(result={'mask': str(mask), 'valid_mask': str(valid)},
                                 output_sha256={str(mask): digest(mask), str(valid): digest(valid)})
        stages['regions'].update(result={'exclusion_mask': str(exclusion)}, output_sha256={str(exclusion): digest(exclusion)})
        (sample_root / 'manifest.json').write_text(json.dumps({'sample': 'sample-A', 'stages': stages}))
        (root / 'run/report.json').write_text(json.dumps({'samples': [{'sample': 'sample-A'}]}))
        result = W.validate_plan_result(plan)
        self.assertEqual(result['samples'][0]['mask'], str(mask))
        np.save(mask, np.zeros((30, 16), dtype=np.uint8))
        with self.assertRaisesRegex(W.AnalysisPlanError, '校验不一致'):
            W.validate_plan_result(plan)


if __name__ == '__main__':
    unittest.main()
