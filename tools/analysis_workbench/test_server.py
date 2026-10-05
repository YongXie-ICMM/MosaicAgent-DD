"""Offline workbench contracts: temporary data, no sockets, models, or devices."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("workbench_server_under_test", HERE / "server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.scan = self.root / "scanner"
        self.scan.mkdir()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps({
            "project_root": str(self.project), "scan_repo": str(self.scan),
            "inference_results": "",
        }), encoding="utf-8")
        self.workbench = server.Workbench(self.config_path)
        self.bridge = types.ModuleType("scan_bridge")
        self.bridge.inspect_scanner = Mock(return_value={
            "exists": True, "platform_supported": False, "revision": "fixture",
            "version_note": server.words("测试资料", "Fixture only"),
        })
        self.bridge.scanner_history = Mock(return_value=[])
        self.bridge.history_signature = Mock(return_value=())
        self.bridge.last_launch = Mock(return_value=None)
        self.bridge.launch_scanner = Mock(side_effect=AssertionError("Scanner must not launch"))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"scan_bridge": self.bridge}).start()
        self.urlopen = patch.object(
            server.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network request")
        ).start()
        self.run = patch.object(
            server.subprocess, "run", side_effect=AssertionError("Unexpected child process")
        ).start()
        self.popen = patch.object(
            server.subprocess, "Popen", side_effect=AssertionError("Unexpected child process")
        ).start()
        patch.object(server.webbrowser, "open", side_effect=AssertionError("Unexpected browser launch")).start()

    def write(self, relative, data=PNG):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_json(self, relative, value):
        return self.write(relative, json.dumps(value).encode("utf-8"))

    def registered(self, path, **kwargs):
        result = self.workbench.register(path, "images", "original", server.words("原图", "Original"), **kwargs)
        self.assertIsNotNone(result)
        return result["id"]

    def source_archive(self, member="native/original.png"):
        archive = self.project / "source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(member, PNG)
        return archive, member

    def image_manifest(self, samples):
        return self.write_json(server.IMAGE_RUN + "/results/run_manifest.json", {"samples": samples})

    def test_new_config_never_autoloads_private_project_and_prefers_bundled_scanner(self):
        checkout = self.root / "public_checkout"
        checkout.mkdir()
        with patch.object(server, "REPO", checkout):
            first = server.Workbench(self.root / "default_one.json")
            self.assertEqual(first.config["project_root"], "")
            self.assertIsNone(first.inference_results)
            self.assertEqual(Path(first.config["scan_repo"]), checkout.parent / "AmScope-Camera")
            launcher = checkout / "acquisition/Auto_Scan/launch_scan.py"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("# fixture only")
            second = server.Workbench(self.root / "default_two.json")
            self.assertEqual(second.config["project_root"], "")
            self.assertEqual(Path(second.config["scan_repo"]), checkout / "acquisition")
            self.assertEqual(second.collect(), [])

    def test_regular_artifact_requires_manifest_hash(self):
        path = self.write("original.png")
        aid = self.registered(path, expected_hash=hashlib.sha256(PNG).hexdigest())
        self.assertEqual(self.workbench.artifact(aid)[:2], (PNG, "image/png"))
        path.write_bytes(PNG + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.workbench.artifact(aid)

    def test_wrong_manifest_hash_is_rejected(self):
        aid = self.registered(self.write("original.png"), expected_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.workbench.artifact(aid)

    def test_zip_member_original_is_read_without_extraction_or_file_changes(self):
        archive, member = self.source_archive()
        self.image_manifest([{
            "sample_id": "case_one", "archive": str(archive), "member": member,
            "member_sha256": hashlib.sha256(PNG).hexdigest(), "outputs": {},
        }])
        originals = [item for item in self.workbench.collect() if item["role"] == "original" and item["tool"] == "images"]
        self.assertEqual(len(originals), 1)
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(zipfile.ZipFile, "extract", side_effect=AssertionError("No extraction")), \
                patch.object(zipfile.ZipFile, "extractall", side_effect=AssertionError("No extraction")):
            data, mime, _ = self.workbench.artifact(originals[0]["id"])
        self.assertEqual((data, mime), (PNG, "image/png"))
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)
        self.assertFalse((self.project / member).exists())

    def test_zip_member_hash_mismatch_is_rejected(self):
        archive, member = self.source_archive()
        aid = self.registered(archive, member=member, expected_hash="f" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.workbench.artifact(aid)

    def test_outside_file_and_symlink_are_not_registered(self):
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        link = self.project / "linked.png"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("Symlinks unavailable: " + str(exc))
        for candidate in [outside, link]:
            with self.subTest(path=candidate):
                self.assertFalse(server.inside(candidate, self.project))
                self.assertIsNone(self.workbench.register(candidate, "images", "original", {}))
        self.assertEqual(self.workbench.artifacts, {})

    def test_file_replaced_with_outside_symlink_is_rejected_on_read(self):
        path = self.write("original.png")
        aid = self.registered(path)
        outside = self.root / "outside.png"
        outside.write_bytes(b"private external data")
        path.unlink()
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("Symlinks unavailable: " + str(exc))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.workbench.artifact(aid)

    def test_invalid_configuration_is_atomic(self):
        file_path = self.write("not_a_directory.txt", b"fixture")
        valid_new_project = self.root / "another_project"
        valid_new_project.mkdir()
        invalid_values = [
            {"unknown_key": str(self.project)},
            {"project_root": str(self.root / "missing")},
            {"project_root": str(file_path)},
            {"project_root": 123},
            {"project_root": "x" * 2049},
            {"project_root": str(valid_new_project), "scan_repo": str(self.root / "missing")},
            [], None,
        ]
        old_config = dict(self.workbench.config)
        old_bytes = self.config_path.read_bytes()
        old_events = self.workbench.events()
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.workbench.configure(values)
            self.assertEqual(self.workbench.config, old_config)
            self.assertEqual(self.config_path.read_bytes(), old_bytes)
            self.assertEqual(self.workbench.events(), old_events)

    def test_empty_project_does_not_invent_artifacts(self):
        self.assertEqual(self.workbench.collect(), [])
        state = self.workbench.state()
        for item in state["tools"]:
            if item["id"] != "scan":
                self.assertFalse(item["available"])
                self.assertFalse(item["launchable"])
        self.assertEqual(state["artifacts"], [])

    def test_manifest_does_not_invent_missing_saved_predictions(self):
        archive, member = self.source_archive()
        self.image_manifest([{
            "sample_id": "case_one", "archive": str(archive), "member": member,
            "member_sha256": hashlib.sha256(PNG).hexdigest(),
            "outputs": {"mask_color.png": {"sha256": "0" * 64}},
        }])
        initial = self.workbench.collect()
        self.assertEqual({item["role"] for item in initial}, {"original", "data"})
        saved = self.write(server.IMAGE_RUN + "/results/case_one/mask_color.png")
        listed = self.workbench.collect()
        self.assertEqual(sum(item["role"] == "prediction" for item in listed), 2)
        self.assertFalse(any(item["role"] == "overlay" for item in listed))
        saved.unlink()
        self.assertFalse(any(item["role"] == "prediction" for item in self.workbench.collect()))

    def test_unknown_artifact_is_not_read_as_a_path(self):
        for value in ["missing", "../outside.png", str(self.config_path)]:
            with self.subTest(value=value), self.assertRaises(FileNotFoundError):
                self.workbench.artifact(value)

    def test_layers_reviews_saved_inference_without_network_or_subprocess(self):
        archive, member = self.source_archive()
        digest = hashlib.sha256(PNG).hexdigest()
        self.image_manifest([{
            "sample_id": "case_one", "archive": str(archive), "member": member,
            "member_sha256": digest, "outputs": {"mask_color.png": {"sha256": digest}},
        }])
        self.write(server.IMAGE_RUN + "/results/case_one/mask_color.png")
        state = self.workbench.state()
        layer = next(item for item in state["tools"] if item["id"] == "layers")
        self.assertTrue(layer["launchable"])
        self.assertNotIn("secondary_actions", layer)
        artifacts = [item for item in state["artifacts"] if item["tool"] == "layers"]
        self.assertEqual({item["role"] for item in artifacts}, {"original", "prediction", "data"})
        images = [item for item in artifacts if item["role"] != "data"]
        self.assertEqual({item["case_id"] for item in images}, {"case_one"})
        for item in images:
            self.assertEqual(self.workbench.artifact(item["id"])[0], PNG)
        self.assertEqual(state["counts"]["images"], 1)
        result = self.workbench.action("layers")
        self.assertTrue(result["ok"])
        self.assertNotIn("url", result)
        self.assertEqual(self.workbench.events()[-1]["kind"], "review_opened")
        self.urlopen.assert_not_called()
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_removed_views_fail_without_launch_or_event(self):
        before = self.workbench.events()
        for view in ("outlines", "layers", "instances", "touching"):
            with self.subTest(view=view), self.assertRaisesRegex(ValueError, "Unknown tool view"):
                self.workbench.action("layers", view)
        self.assertEqual(self.workbench.events(), before)
        self.urlopen.assert_not_called()
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_layers_exposes_saved_counting_support_and_denominators(self):
        self.write("06_analysis/Figure3_current/fig3_recount_overview.png")
        record = {"source_region": "test-region", "numerator": 20, "denominator": 100}
        self.write_json("06_analysis/Figure3_current/fig3_recount.json", record)
        artifacts = [item for item in self.workbench.collect() if item["tool"] == "layers"]
        self.assertEqual({item["role"] for item in artifacts}, {"report", "data"})
        data = next(item for item in artifacts if item["role"] == "data")
        self.assertEqual(json.loads(self.workbench.artifact(data["id"])[0]), record)

    def test_configured_inference_results_is_read_without_private_tool_assets(self):
        external = self.root / "saved_layer_run"
        external.mkdir()
        self.workbench.configure({"inference_results": str(external)})
        archive, member = self.source_archive()
        results = external / "results"
        results.mkdir()
        (results / "run_manifest.json").write_text(json.dumps({"samples": [{
            "sample_id": "example", "archive": str(archive), "member": member,
            "member_sha256": hashlib.sha256(PNG).hexdigest(), "outputs": {},
        }]}))
        artifacts = [item for item in self.workbench.collect() if item["tool"] == "layers"]
        self.assertEqual({item["role"] for item in artifacts}, {"original", "data"})
        self.assertTrue(self.workbench.action("layers")["ok"])
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_legacy_config_is_ignored_without_rewriting_source_file(self):
        saved = {"project_root": str(self.project), "scan_repo": str(self.scan),
                 "annotation_results": "/old/private/research"}
        self.config_path.write_text(json.dumps(saved))
        original = self.config_path.read_bytes()
        migrated = server.Workbench(self.config_path)
        self.assertNotIn("annotation_results", migrated.config)
        self.assertEqual(self.config_path.read_bytes(), original)
        self.assertEqual(migrated.collect(), [])
        with self.assertRaisesRegex(ValueError, "Unsupported settings"):
            migrated.configure({"annotation_results": "/old/private/research"})

    def test_scan_unavailable_on_non_windows_without_launch_attempt(self):
        with patch.object(server.platform, "system", return_value="Darwin"):
            scan = next(item for item in self.workbench.state()["tools"] if item["id"] == "scan")
            self.assertFalse(scan["launchable"])
            self.assertEqual(scan["status"], "platform_unavailable")
            before = self.workbench.events()
            with self.assertRaisesRegex(ValueError, "Windows"):
                self.workbench.action("scan")
        self.assertEqual(self.workbench.events(), before)
        self.bridge.launch_scanner.assert_not_called()
        self.urlopen.assert_not_called()
        self.run.assert_not_called()
        self.popen.assert_not_called()

    def test_construction_writes_no_opened_event(self):
        """'opened' is written by main() once the port is actually served, so a second
        launch that only reuses a running workbench leaves no false row."""
        self.assertEqual(self.workbench.events(), [])

    def test_invalid_student_package_has_disabled_launch_and_bilingual_reason(self):
        self.bridge.inspect_scanner.return_value.update(platform_supported=True, launch_blocked=True)
        scan = next(item for item in self.workbench.state()["tools"] if item["id"] == "scan")
        self.assertEqual(scan["status"], "package_mismatch")
        self.assertFalse(scan["launchable"])
        with self.assertRaises(server.WorkbenchError) as caught:
            self.workbench.action("scan")
        self.assertIn("delivery manifest", caught.exception.message["en"])
        self.assertIn("交付清单", caught.exception.message["zh"])
        self.bridge.launch_scanner.assert_not_called()

    def test_errors_are_bilingual_for_missing_layer_data(self):
        self.workbench.config["project_root"] = ""
        with self.assertRaises(server.WorkbenchError) as caught:
            self.workbench.action("layers")
        self.assertEqual(set(caught.exception.message), {"zh", "en"})
        self.assertIn("saved", caught.exception.message["en"])

    def test_foreign_absolute_manifest_paths_are_relocated_under_the_project(self):
        """Manifests written on another computer carry that computer's absolute paths."""
        archive, member = self.source_archive()
        foreign = "/Users/someone/elsewhere/" + self.project.name + "/" + archive.name
        self.image_manifest([{
            "sample_id": "case_one", "archive": foreign, "member": member,
            "member_sha256": hashlib.sha256(PNG).hexdigest(), "outputs": {},
        }])
        originals = [item for item in self.workbench.collect() if item["role"] == "original" and item["tool"] == "images"]
        self.assertEqual(len(originals), 1)
        self.assertEqual(self.workbench.artifact(originals[0]["id"])[:2], (PNG, "image/png"))
        self.assertIsNone(self.workbench.relocate("/Users/someone/elsewhere/" + self.project.name + "/missing.zip"))
        self.assertIsNone(self.workbench.relocate("/Users/someone/other_project/" + archive.name))

    def test_windows_manifest_paths_relocate_on_any_host(self):
        archive, _ = self.source_archive()
        foreign = "D:\\experiments\\" + self.project.name + "\\" + archive.name
        self.assertEqual(self.workbench.relocate(foreign), archive)

    def test_relocation_never_escapes_configured_folders(self):
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        self.assertIsNone(self.workbench.register("/elsewhere/" + self.project.name + "/../outside.png", "images", "original", {}))
        self.assertEqual(self.workbench.artifacts, {})

    def test_scanner_exit_is_recorded_as_error_with_code_and_log_tail(self):
        self.workbench._scanner_exited({"pid": 5, "returncode": 1, "log_path": "x.log", "entrypoint": "e",
                                        "started_at_utc": "t0", "ended_at_utc": "t1",
                                        "log_tail": ["Console history: x", "", "Missing dependencies: numpy"]})
        event = self.workbench.events()[-1]
        self.assertEqual((event["tool"], event["kind"]), ("scan", "error"))
        self.assertIn("exited with code 1", event["message"]["en"])
        self.assertIn("Missing dependencies: numpy", event["message"]["en"])
        self.assertIn("退出码 1", event["message"]["zh"])
        self.assertEqual(event["process"]["returncode"], 1)
        self.workbench._scanner_exited({"pid": 6, "returncode": 0, "log_path": "y.log", "log_tail": []})
        event = self.workbench.events()[-1]
        self.assertEqual(event["kind"], "process_exited")
        self.assertIn("scanner's own history", event["message"]["en"])

    def test_scanner_history_is_reparsed_only_when_files_change(self):
        self.bridge.history_signature.return_value = ("a",)
        self.workbench.state()
        self.workbench.state()
        self.assertEqual(self.bridge.scanner_history.call_count, 1)
        self.bridge.history_signature.return_value = ("b",)
        with patch.object(server, "HISTORY_TTL_S", 0.0):
            self.workbench.state()
        self.assertEqual(self.bridge.scanner_history.call_count, 2)
        self.bridge.history_signature.return_value = ("c",)
        self.workbench.state()   # changed again, but within the TTL: the previous rows are reused
        self.assertEqual(self.bridge.scanner_history.call_count, 2)
        self.assertEqual(self.bridge.inspect_scanner.call_count, 1)

    def test_scanner_inspection_refreshes_when_the_folder_setting_changes(self):
        self.workbench.state()
        other = self.root / "other_scanner"
        other.mkdir()
        self.workbench.configure({"scan_repo": str(other)})
        self.workbench.state()
        self.assertEqual(self.bridge.inspect_scanner.call_count, 2)
        self.assertEqual(Path(self.bridge.inspect_scanner.call_args[0][0]), other)

    def test_package_revision_and_last_launch_are_shown_in_scan_details(self):
        self.bridge.inspect_scanner.return_value = {
            "exists": True, "platform_supported": False, "revision": "f409384fcf7d99b938dabe2b51", "revision_source": "delivery_manifest",
            "layout": "package", "version_note": server.words("学生包", "Student package"),
        }
        self.bridge.last_launch.return_value = {"pid": 12, "status": "exited", "returncode": 1}
        scan = next(item for item in self.workbench.state()["tools"] if item["id"] == "scan")
        self.assertEqual(scan["revision"], "f409384fcf7d (delivery manifest)")
        details = " ".join(note["en"] for note in scan["details"])
        self.assertIn("delivery_manifest.json", details)
        self.assertIn("Last launch: PID 12, status exited, exit code 1", details)


if __name__ == "__main__":
    unittest.main()
