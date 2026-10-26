"""Read-only spatial-map review, using temporary saved outputs only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'workbench_spatial_under_test', Path(__file__).with_name('server.py'))
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class SpatialResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.results = self.root / 'results'
        self.results.mkdir()
        config = self.root / 'config.json'
        config.write_text(json.dumps({'project_root': str(self.root),
                                      'inference_results': str(self.results)}))
        self.workbench = server.Workbench(config)
        for target in ('run', 'Popen'):
            handle = patch.object(server.subprocess, target,
                                  side_effect=AssertionError('No child process'))
            handle.start()
            self.addCleanup(handle.stop)
        for module, name in ((server.urllib.request, 'urlopen'), (server.webbrowser, 'open')):
            handle = patch.object(module, name, side_effect=AssertionError('No network/browser'))
            handle.start()
            self.addCleanup(handle.stop)

    def export(self, relative='', **changes):
        run = self.results / relative
        run.mkdir(parents=True, exist_ok=True)
        records = []
        for name, role, payload in (
                ('heatmaps.png', 'preview', b'PNG fixture'),
                ('combined.svg', 'figure', b'<svg xmlns="http://www.w3.org/2000/svg"/>'),
                ('counts.csv', 'data', b'layer,count\n1L,5\n'),
                ('settings.json', 'data', b'{"fixture": true}'),
                ('arrays.npz', 'data', b'NPZ fixture')):
            (run / name).write_bytes(payload)
            records.append({'path': name, 'role': role, 'sha256': hashlib.sha256(payload).hexdigest()})
        manifest = {'schema_version': 1, 'kind': 'layer_spatial_maps', 'sample': 'sample-A',
                    'grids': {'heatmaps': [8, 15], 'combined': [6, 11]}, 'outputs': records}
        manifest.update(changes)
        path = run / 'spatial_manifest.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    def collect(self):
        return [item for item in self.workbench.collect() if item['tool'] == 'layers']

    def test_standalone_export_has_figures_data_provenance_and_bilingual_details(self):
        manifest_path, _ = self.export()
        before = {p: p.read_bytes() for p in self.results.rglob('*') if p.is_file()}
        items = self.collect()
        self.assertEqual(len(items), 6)
        self.assertEqual({item['role'] for item in items}, {'report', 'data'})
        self.assertEqual({item['mime'] for item in items if item['role'] == 'report'},
                         {'image/png', 'image/svg+xml'})
        for item in items:
            self.assertTrue(item['label']['zh'])
            self.assertTrue(item['label']['en'])
            contents = self.workbench.artifact(item['id'])[0]
            self.assertEqual(contents, Path(item['path']).read_bytes())
        details = ' '.join(note['en'] for note in self.workbench.spatial_details)
        self.assertIn('8 × 15', details)
        self.assertIn('6 × 11', details)
        self.assertIn('coverage is separate', details)
        self.assertEqual(before, {p: p.read_bytes() for p in self.results.rglob('*') if p.is_file()})
        self.assertFalse((manifest_path.parent / 'results/run_manifest.json').exists())
        self.assertEqual(self.workbench.errors, [])

    def test_inference_and_pipeline_runs_are_discovered_without_inference_manifest(self):
        self.export('spatial_maps/20261026/first', sample='first')
        self.export('sample-A/04_stats/spatial_maps/second', sample='second')
        items = self.collect()
        self.assertEqual(len(items), 12)
        self.assertEqual({item['source_run'] for item in items},
                         {'spatial_maps/20261026/first', 'sample-A/04_stats/spatial_maps/second'})

    def test_only_configured_results_tree_is_discovered(self):
        path, _ = self.export('saved')
        outside = self.root / 'outside_saved'
        path.parent.rename(outside)
        self.assertEqual(self.collect(), [])

    def test_invalid_output_inventory_does_not_expose_partially_valid_files(self):
        path, manifest = self.export()
        for bad in ('../outside.csv', '/absolute.csv', 'sub/../../counts.csv',
                    'C:\\outside.csv', 'sub//counts.csv', './counts.csv'):
            with self.subTest(path=bad):
                manifest['outputs'][-1]['path'] = bad
                path.write_text(json.dumps(manifest))
                self.assertEqual(self.collect(), [])
                self.assertIn('Spatial-map manifest', self.workbench.errors[0]['en'])

    def test_sibling_export_symlink_is_rejected_even_inside_configured_results(self):
        path, manifest = self.export('one')
        sibling = self.results / 'sibling.csv'
        sibling.write_bytes(b'not from this run')
        linked = path.parent / 'linked.csv'
        try:
            linked.symlink_to(sibling)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        manifest['outputs'].append({'path': linked.name, 'role': 'data',
                                    'sha256': hashlib.sha256(sibling.read_bytes()).hexdigest()})
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.collect(), [])
        self.assertIn('own run directory', self.workbench.errors[0]['en'])

    def test_discovery_skips_symlink_directories_and_manifests(self):
        path, _ = self.export('saved')
        outside = self.root / 'outside_saved'
        path.parent.rename(outside)
        try:
            (self.results / 'linked').symlink_to(outside, target_is_directory=True)
            (self.results / 'spatial_manifest.json').symlink_to(outside / path.name)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        self.assertEqual(self.collect(), [])

    def test_output_changed_after_registration_fails_checksum(self):
        path, _ = self.export()
        item = next(item for item in self.collect() if item['path'].endswith('counts.csv'))
        (path.parent / 'counts.csv').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.workbench.artifact(item['id'])

    def test_manifest_changed_after_registration_fails_checksum(self):
        path, _ = self.export()
        item = next(item for item in self.collect() if item['path'] == str(path))
        path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.workbench.artifact(item['id'])

    def test_output_replaced_by_sibling_symlink_is_rechecked_on_read(self):
        path, _ = self.export('saved')
        item = next(item for item in self.collect() if item['path'].endswith('counts.csv'))
        original = path.parent / 'counts.csv'
        sibling = self.results / 'sibling.csv'
        sibling.write_bytes(original.read_bytes())  # Identical hash cannot bypass run scope.
        original.unlink()
        try:
            original.symlink_to(sibling)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        with self.assertRaisesRegex(ValueError, 'own run directory'):
            self.workbench.artifact(item['id'])

    def test_wrong_digest_is_not_silently_served(self):
        path, manifest = self.export()
        manifest['outputs'][0]['sha256'] = '0' * 64
        path.write_text(json.dumps(manifest))
        item = next(item for item in self.collect() if item['path'].endswith('.png'))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.workbench.artifact(item['id'])

    def test_schema_grid_digest_and_role_validation(self):
        for changes in ({'schema_version': True}, {'kind': 'other'}, {'sample': []},
                        {'grids': {'heatmaps': [True, 15]}}, {'outputs': []}):
            with self.subTest(changes=changes):
                self.export(**changes)
                self.assertEqual(self.collect(), [])
                self.assertTrue(self.workbench.errors)
        for key, value in (('role', 'original'), ('sha256', '123'), ('path', 'code.py')):
            with self.subTest(key=key):
                path, manifest = self.export()
                manifest['outputs'][0][key] = value
                path.write_text(json.dumps(manifest))
                self.assertEqual(self.collect(), [])

    def test_discovery_budget_and_depth_limit_warn_instead_of_unbounded_search(self):
        self.export('a/b/c/d')
        with patch.object(server, 'SPATIAL_MAX_DEPTH', 1):
            self.assertEqual(self.collect(), [])
            self.assertIn('limit', self.workbench.errors[0]['en'])
        with patch.object(server, 'SPATIAL_MAX_ENTRIES', 1):
            self.assertEqual(self.collect(), [])
            self.assertIn('limit', self.workbench.errors[0]['en'])
        with patch.object(server, 'SPATIAL_MAX_DIRECTORIES', 1):
            self.assertEqual(self.collect(), [])
            self.assertIn('limit', self.workbench.errors[0]['en'])

    def test_reconnect_clears_spatial_details_and_artifacts(self):
        self.export()
        self.assertTrue(self.collect())
        empty = self.root / 'empty'
        empty.mkdir()
        self.workbench.configure({'inference_results': str(empty)})
        self.assertEqual(self.collect(), [])
        self.assertEqual(self.workbench.spatial_details, [])


if __name__ == '__main__':
    unittest.main()
