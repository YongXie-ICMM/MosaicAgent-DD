"""Offline adapter tests: no imported scanner, GUI, device, or launched child."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import scan_bridge as bridge


HERE = Path(__file__).resolve().parent


def fake_git(answers):
    """Answer _git(repo, *args) by the first argument after -C: 'rev-parse --show-toplevel',
    'rev-parse HEAD' or 'status'."""
    def _git(repo, *args):
        if args[:2] == ("rev-parse", "--show-toplevel"):
            return answers.get("toplevel", str(repo))
        if args[:2] == ("rev-parse", "HEAD"):
            return answers.get("head")
        if args[:1] == ("status",):
            return answers.get("status")
        raise AssertionError(f"unexpected git call {args}")
    return _git


class ScanBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "scanner"
        self.folder = self.repo / "Auto_Scan"
        self.folder.mkdir(parents=True)
        (self.folder / "launch_scan.py").write_text("raise AssertionError('must not execute')\n")
        bridge._PROCESSES.clear()

    def tearDown(self):
        bridge._PROCESSES.clear()
        self.temp.cleanup()

    def event(self, number, event, **fields):
        return dict(schema_version=1, history_id="history-1", event_id=f"event-{number}",
                    sequence=number, timestamp_utc=f"2026-10-04T10:00:{number:02d}+00:00",
                    event=event, **fields)

    def history(self, name="20261004_100000", events=None, journal=True, folder=None):
        events = events or [self.event(1, "history_started", metadata={})]
        directory = (folder or self.folder) / "shared_history" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "history.json"
        value = dict(schema_version=1, history_id="history-1", events=events,
                     event_count=len(events), snapshot_sequence=len(events),
                     is_closed=events[-1]["event"] == "history_closed")
        path.write_text(json.dumps(value), encoding="utf-8")
        if journal:
            path.with_name("events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return path

    def package(self, name="自动扫描试跑_v3", commit="f409384fcf7d99b938dabe2b513f4d848248eb3c", tamper=False, manifest=True):
        """A flat student package: launch_scan.py and the main program beside delivery_manifest.json."""
        folder = self.root / name
        folder.mkdir()
        (folder / "launch_scan.py").write_text("raise AssertionError('must not execute')\n")
        main = folder / bridge._MAIN_PROGRAM
        main.write_bytes(b"# student-tested acquisition core\n")
        digest = hashlib.sha256(main.read_bytes()).hexdigest()
        if tamper:
            main.write_bytes(b"# edited after packaging\n")
        if manifest:
            (folder / "delivery_manifest.json").write_text(json.dumps({
                "release": "954-base-L1", "source_repo_commit": commit, "local_changes": False,
                "hardware_validated": False, "external_model_calls": 0,
                "files": [{"path": bridge._MAIN_PROGRAM, "sha256": digest, "bytes": main.stat().st_size},
                          {"path": "launch_scan.py", "sha256": hashlib.sha256((folder / "launch_scan.py").read_bytes()).hexdigest()}],
            }), encoding="utf-8")
        return folder

    # ------------------------------------------------------------ inspection
    def test_inspect_reports_revision_local_changes_and_version_limit(self):
        with patch.object(bridge, "_git", fake_git({"head": "8a7cd29abc", "status": "?? local.txt"})), \
                patch.object(bridge.platform, "system", return_value="Windows"):
            result = bridge.inspect_scanner(self.repo)
        self.assertTrue(result["exists"])
        self.assertTrue(result["dirty"])
        self.assertTrue(result["platform_supported"])
        self.assertEqual(result["revision_source"], "git")
        self.assertEqual(result["layout"], "checkout")
        self.assertIn("Historical typography revision 8a7cd29", result["version_note"]["en"])
        self.assertIn("superseded by 954-1080p-A1", result["version_note"]["en"])
        self.assertIn("Windows instrument UI validation", result["version_note"]["en"])
        self.assertIn("pending", result["version_note"]["en"])
        self.assertIn("no new 954-point instrument validation", result["version_note"]["en"])
        self.assertIn("Local changes", result["version_note"]["en"])

    def test_current_1080p_release_retains_pending_instrument_validation_scope(self):
        with patch.object(bridge, "_git", fake_git({"head": "25a8392abc", "status": ""})), \
                patch.object(bridge.platform, "system", return_value="Windows"):
            result = bridge.inspect_scanner(self.repo)
        self.assertEqual(result["revision"], "25a8392abc")
        self.assertEqual(result["revision_source"], "git")
        self.assertFalse(result["dirty"])
        self.assertTrue(result["platform_supported"])
        note = result["version_note"]["en"]
        for phrase in ("954-1080p-A1", "received 1920 × 1080 image checks", "Claude typography",
                       "Offline checks completed", "instrument/stitching trials are pending",
                       "original 954-point result does not validate this release"):
            self.assertIn(phrase, note)

    def test_unknown_git_state_is_not_clean(self):
        with patch.object(bridge, "_git", return_value=None):
            result = bridge.inspect_scanner(self.repo)
        self.assertIsNone(result["dirty"])
        self.assertIsNone(result["revision"])
        self.assertIn("cannot be confirmed", result["version_note"]["en"])

    def test_git_toplevel_elsewhere_is_not_this_scanner(self):
        """A package extracted inside another repository must not inherit its HEAD."""
        with patch.object(bridge, "_git", fake_git({"toplevel": str(self.root / "elsewhere"), "head": "deadbeef"})):
            result = bridge.inspect_scanner(self.repo)
        self.assertIsNone(result["revision"])
        self.assertIsNone(result["dirty"])

    def test_flat_student_package_is_recognized_from_delivery_manifest(self):
        folder = self.package()
        with patch.object(bridge, "_git", return_value=None), patch.object(bridge.platform, "system", return_value="Windows"):
            result = bridge.inspect_scanner(folder)
        self.assertTrue(result["exists"])
        self.assertEqual(result["layout"], "package")
        self.assertEqual(Path(result["entrypoint"]), folder / "launch_scan.py")
        self.assertEqual(result["revision_source"], "delivery_manifest")
        self.assertTrue(result["revision"].startswith("f409384"))
        self.assertIsNone(result["dirty"])
        self.assertTrue(result["package"]["main_program_matches_manifest"])
        self.assertIn("Historical L1 baseline", result["version_note"]["en"])
        self.assertIn("superseded by 954-1080p-A1", result["version_note"]["en"])
        self.assertIn("not the added language UI", result["version_note"]["en"])
        self.assertIn("954-base-L1", result["version_note"]["en"])
        self.assertIn("matches the delivery manifest", result["version_note"]["en"])
        self.assertIn("一致", result["version_note"]["zh"])

    def test_tampered_package_main_program_is_reported(self):
        folder = self.package(tamper=True)
        with patch.object(bridge, "_git", return_value=None):
            result = bridge.inspect_scanner(folder)
        self.assertIs(result["package"]["main_program_matches_manifest"], False)
        self.assertIn("does NOT match", result["version_note"]["en"])
        self.assertIn("不一致", result["version_note"]["zh"])

    def test_package_without_manifest_or_git_is_unverified(self):
        folder = self.package(manifest=False)
        with patch.object(bridge, "_git", return_value=None):
            result = bridge.inspect_scanner(folder)
        self.assertTrue(result["exists"])
        self.assertIsNone(result["revision"])
        self.assertIsNone(result["package"])
        self.assertIn("delivery_manifest.json", result["version_note"]["en"])

    def test_changed_helper_is_blocked_even_if_main_program_matches(self):
        folder = self.package()
        helper = folder / "camera_session_history.py"
        helper.write_bytes(b"# expected camera discovery\n")
        manifest = folder / "delivery_manifest.json"
        data = json.loads(manifest.read_text())
        data["files"].append({"path": helper.name, "sha256": hashlib.sha256(helper.read_bytes()).hexdigest()})
        manifest.write_text(json.dumps(data))
        helper.write_bytes(b"# replaced camera discovery\n")
        with patch.object(bridge, "_git", return_value=None), patch.object(bridge.platform, "system", return_value="Windows"), \
                patch.object(bridge.subprocess, "Popen") as popen:
            info = bridge.inspect_scanner(folder)
            self.assertTrue(info["package"]["main_program_matches_manifest"])
            self.assertFalse(info["package"]["package_files_match_manifest"])
            self.assertTrue(info["launch_blocked"])
            self.assertIn(helper.name, info["version_note"]["en"])
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                bridge.launch_scanner(folder, self.root / "logs", "python")
            popen.assert_not_called()

    def test_malformed_or_unsafe_manifest_is_reported_without_crashing(self):
        folder = self.package()
        manifest = folder / "delivery_manifest.json"
        original = json.loads(manifest.read_text())
        invalid_lists = [3, {}, [], [{"path": "../outside.py", "sha256": "0" * 64}],
                         original["files"] + [original["files"][0]],
                         [{"path": "C:\\elsewhere.py", "sha256": "0" * 64}]]
        for files in invalid_lists:
            with self.subTest(files=files):
                manifest.write_text(json.dumps(dict(original, files=files)))
                with patch.object(bridge, "_git", return_value=None):
                    info = bridge.inspect_scanner(folder)
                self.assertTrue(info["launch_blocked"])
                self.assertTrue(info["package"]["error"])

    def test_manifest_error_without_revision_is_visible(self):
        folder = self.package()
        (folder / "delivery_manifest.json").write_text('{"broken":')
        with patch.object(bridge, "_git", return_value=None):
            info = bridge.inspect_scanner(folder)
        self.assertTrue(info["launch_blocked"])
        self.assertIn("verification failed", info["version_note"]["en"])

    def test_windows_manifest_helper_paths_are_checked_on_any_host(self):
        folder = self.package()
        helper = folder / "drivers" / "camera.dll"
        helper.parent.mkdir()
        helper.write_bytes(b"fixture binary")
        manifest = folder / "delivery_manifest.json"
        data = json.loads(manifest.read_text())
        data["files"].append({"path": "drivers\\camera.dll", "sha256": hashlib.sha256(helper.read_bytes()).hexdigest()})
        manifest.write_text(json.dumps(data))
        self.assertTrue(bridge.read_delivery_manifest(folder)["package_files_match_manifest"])

    def test_revision_notes_come_from_the_json_table(self):
        table = json.loads(bridge._REVISIONS_FILE.read_text(encoding="utf-8"))
        prefixes = [p for entry in table["revisions"] for p in entry["prefixes"]]
        self.assertIn("46f48ac", prefixes)
        self.assertIn("8a7cd29", prefixes)
        with patch.object(bridge, "_git", fake_git({"head": "0123456789", "status": ""})):
            note = bridge.inspect_scanner(self.repo)["version_note"]
        self.assertEqual(note["en"], table["unknown"]["en"])

    # ------------------------------------------------------------ launching
    def test_non_windows_cannot_launch(self):
        with patch.object(bridge.platform, "system", return_value="Darwin"), patch.object(bridge.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                bridge.launch_scanner(self.repo, self.root / "logs", "python")
            popen.assert_not_called()
        self.assertFalse((self.root / "logs").exists())

    def test_launch_exact_command_and_duplicate_rejected_without_killing(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen", return_value=process) as popen:
            result = bridge.launch_scanner(self.repo, self.root / "logs", "configured-python")
            self.assertEqual(result["pid"], 1234)
            args, kwargs = popen.call_args
            self.assertEqual(args[0], ["configured-python", "-u", str(self.folder / "launch_scan.py")])
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["cwd"], str(self.folder))
            self.assertTrue(Path(result["log_path"]).is_file())
            with self.assertRaisesRegex(RuntimeError, "already running"):
                bridge.launch_scanner(self.repo, self.root / "logs", "configured-python")
            self.assertEqual(popen.call_count, 1)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        active = json.loads((self.root / "logs" / "active_scanner.json").read_text(encoding="utf-8"))
        self.assertEqual((active["pid"], active["status"]), (1234, "running"))

    def test_flat_package_launches_from_its_own_folder(self):
        folder = self.package()
        process = Mock(pid=9)
        process.poll.return_value = None
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen", return_value=process) as popen:
            bridge.launch_scanner(folder, self.root / "logs", "py")
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["py", "-u", str(folder / "launch_scan.py")])
        self.assertEqual(kwargs["cwd"], str(folder))

    def test_exit_watcher_reports_returncode_and_log_tail(self):
        gate = threading.Event()
        process = Mock(pid=55)
        process.poll.return_value = None
        process.wait.side_effect = lambda: (gate.wait(5), 1)[1]
        seen, done = [], threading.Event()

        def on_exit(record):
            seen.append(record)
            done.set()

        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen", return_value=process):
            result = bridge.launch_scanner(self.repo, self.root / "logs", "python", on_exit=on_exit)
            Path(result["log_path"]).write_text("Console history: x\nMissing dependencies: numpy\n", encoding="utf-8")
            gate.set()
            self.assertTrue(done.wait(5), "watcher did not report the exit")
        record = seen[0]
        self.assertEqual(record["returncode"], 1)
        self.assertEqual(record["pid"], 55)
        self.assertIn("Missing dependencies: numpy", record["log_tail"])
        active = json.loads((self.root / "logs" / "active_scanner.json").read_text(encoding="utf-8"))
        self.assertEqual((active["status"], active["returncode"]), ("exited", 1))
        self.assertEqual(bridge.last_launch(self.root / "logs")["returncode"], 1)
        process.terminate.assert_not_called()

    def test_existing_lock_is_not_removed(self):
        lock = self.folder / ".mosaic_stage.lock"
        lock.write_text("42")
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "lock exists"):
                bridge.launch_scanner(self.repo, self.root / "logs", "python")
            popen.assert_not_called()
        self.assertEqual(lock.read_text(), "42")

    def test_lock_message_distinguishes_running_and_stale_pid(self):
        lock = self.folder / ".mosaic_stage.lock"
        lock.write_text(str(os.getpid()))
        self.assertIn("appears to be running", bridge.describe_lock(lock))
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        lock.write_text(str(child.pid))
        with patch.object(bridge, "pid_alive", return_value=False):
            message = bridge.describe_lock(lock)
        self.assertIn("stale", message)
        self.assertIn("remove it by hand", message)
        self.assertIn(str(lock), message)
        lock.write_text("not a pid")
        self.assertIn("no readable PID", bridge.describe_lock(lock))
        self.assertTrue(lock.exists())

    def test_pid_alive_never_signals_and_answers_for_this_process(self):
        self.assertTrue(bridge.pid_alive(os.getpid()))
        self.assertIsNone(bridge.pid_alive(0))
        self.assertIsNone(bridge.pid_alive("x"))
        if os.name == "nt":
            with patch.object(bridge.os, "kill", side_effect=AssertionError("os.kill terminates on Windows")):
                self.assertTrue(bridge.pid_alive(os.getpid()))

    def test_missing_entry_and_failed_creation_do_not_register_process(self):
        entry = self.folder / "launch_scan.py"
        entry.unlink()
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen") as popen:
            with self.assertRaises(FileNotFoundError):
                bridge.launch_scanner(self.repo, self.root / "logs", "python")
            popen.assert_not_called()
        entry.write_text("# fixture")
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen", side_effect=OSError("not executable")):
            with self.assertRaises(OSError):
                bridge.launch_scanner(self.repo, self.root / "logs", "python")
        self.assertFalse(bridge._PROCESSES)

    def test_finished_managed_process_can_be_launched_again(self):
        old = Mock()
        old.poll.return_value = 0
        bridge._PROCESSES[str(self.folder / "launch_scan.py")] = old
        new = Mock(pid=77)
        with patch.object(bridge.platform, "system", return_value="Windows"), patch.object(bridge.subprocess, "Popen", return_value=new):
            result = bridge.launch_scanner(self.repo, self.root / "logs", "python")
        self.assertEqual(result["pid"], 77)
        old.kill.assert_not_called()

    # ------------------------------------------------------------ histories
    def test_closed_history_is_not_scan_or_image_verification(self):
        events = [self.event(1, "history_started", metadata={}),
                  self.event(2, "candidate_accepted", human_confirmed=True),
                  self.event(3, "capture_success"),
                  self.event(4, "session_finished", status="aborted", photos_saved=1),
                  self.event(5, "history_closed", status="closed")]
        path = self.history(events=events)
        before = {p: p.read_bytes() for p in path.parent.iterdir()}
        result = bridge.scanner_history(self.repo)[0]
        self.assertIn("latest scan finish: aborted", result["message"]["en"])
        self.assertIn("have not been verified", result["message"]["en"])
        self.assertIn("accepted images: 1", result["message"]["en"])
        self.assertEqual(before, {p: p.read_bytes() for p in path.parent.iterdir()})

    def test_string_false_does_not_count_as_human_confirmation(self):
        self.history(events=[self.event(1, "history_started", metadata={}),
                             self.event(2, "candidate_accepted", human_confirmed="False")])
        self.assertIn("human acceptances: 0", bridge.scanner_history(self.repo)[0]["message"]["en"])

    def test_pending_verified_and_mismatched_1080p_contracts_stay_distinct(self):
        contract = {"schema_version": 1, "expected_image_size": [1920, 1080], "actual_image_size": None,
                    "verification_status": "pending_pre_scan", "verification_phase": "pre_scan",
                    "calibration_status": "unverified_for_current_mode", "raw_images_resized": False}
        events = [self.event(1, "history_started", metadata={}),
                  self.event(2, "session_started", session={"scan_config": {"acquisition_contract": contract}})]
        self.history(events=events)
        row = bridge.scanner_history(self.repo)[0]
        self.assertIn("verification pending", row["message"]["en"])
        verified = dict(contract, actual_image_size=[1920, 1080], verification_status="verified_received_frame")
        events.append(self.event(3, "acquisition_mode_verified", acquisition_contract=verified))
        self.history(events=events)
        row = bridge.scanner_history(self.repo)[0]
        self.assertIn("received-frame size verified", row["message"]["en"])
        self.assertIn("unverified_for_current_mode", row["message"]["en"])
        mismatch = dict(verified, actual_image_size=[640, 480], verification_status="mismatch", verification_phase="resume")
        events.append(self.event(4, "acquisition_mode_mismatch", acquisition_contract=mismatch))
        self.history(events=events)
        row = bridge.scanner_history(self.repo)[0]
        self.assertEqual(row["kind"], "scanner_history_warning")
        self.assertEqual(row["acquisition_contract"], mismatch)
        self.assertIn("640 × 480", row["message"]["en"])

    def test_previous_scan_verification_cannot_verify_new_scan(self):
        verified = {"schema_version": 1, "expected_image_size": [1920, 1080], "actual_image_size": [1920, 1080],
                    "verification_status": "verified_received_frame", "verification_phase": "pre_scan"}
        events = [self.event(1, "history_started", metadata={}),
                  self.event(2, "acquisition_mode_verified", acquisition_contract=verified),
                  self.event(3, "session_started", session={"scan_config": {}})]
        self.history(events=events)
        row = bridge.scanner_history(self.repo)[0]
        self.assertIsNone(row["acquisition_contract"])
        self.assertNotIn("received-frame size verified", row["message"]["en"])

    def test_journal_lag_and_uncommitted_tail_are_visible_read_only(self):
        path = self.history()
        journal = path.with_name("events.jsonl")
        with journal.open("a") as stream:
            stream.write(json.dumps(self.event(2, "capture_success")) + "\n" + '{"event":')
        before = journal.read_bytes()
        row = bridge.scanner_history(self.repo)[0]
        self.assertEqual(row["kind"], "scanner_history_warning")
        self.assertIn("snapshot_lag=1", row["message"]["en"])
        self.assertIn("uncommitted_trailing_line", row["message"]["en"])
        self.assertIn("capture-success events: 1", row["message"]["en"])
        self.assertEqual(before, journal.read_bytes())

    def test_snapshot_without_journal_warns(self):
        self.history(journal=False)
        self.assertIn("snapshot_only", bridge.scanner_history(self.repo)[0]["message"]["en"])

    def test_tampered_journal_does_not_produce_success_summary(self):
        path = self.history()
        event = self.event(1, "history_started", metadata={"changed": True})
        path.with_name("events.jsonl").write_text(json.dumps(event) + "\n")
        result = bridge.scanner_history(self.repo)[0]
        self.assertEqual(result["kind"], "scanner_history_error")
        self.assertIn("prefix", result["message"]["en"])

    def test_invalid_snapshot_and_read_size_limit_surface_errors(self):
        path = self.history()
        path.write_text("{broken")
        self.assertEqual(bridge.scanner_history(self.repo)[0]["kind"], "scanner_history_error")
        self.history()
        with patch.object(bridge, "_MAX_BYTES", 10):
            self.assertIn("read limit", bridge.scanner_history(self.repo)[0]["message"]["en"])

    def test_limit_sorting_and_missing_root(self):
        self.assertEqual(bridge.scanner_history(self.repo), [])
        self.history("20261003")
        newest = self.history("20261004")
        result = bridge.scanner_history(self.repo, limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], str(newest))
        for value in (0, 21, True, "5"):
            with self.assertRaises(ValueError):
                bridge.scanner_history(self.repo, value)

    def test_flat_package_history_is_read_beside_its_main_program(self):
        folder = self.package()
        self.history(folder=folder, events=[self.event(1, "history_started", metadata={}),
                                            self.event(2, "capture_success")])
        rows = bridge.scanner_history(folder)
        self.assertEqual(len(rows), 1)
        self.assertIn("capture-success events: 1", rows[0]["message"]["en"])
        self.assertEqual(bridge.history_root(folder), folder / "shared_history")

    def test_signature_changes_only_when_history_files_change(self):
        self.assertEqual(bridge.history_signature(self.repo), ())
        path = self.history()
        first = bridge.history_signature(self.repo)
        self.assertEqual(first, bridge.history_signature(self.repo))
        with path.with_name("events.jsonl").open("a") as stream:
            stream.write(json.dumps(self.event(2, "capture_success")) + "\n")
        self.assertNotEqual(first, bridge.history_signature(self.repo))

    def test_synthetic_scanner_schema_history_parses(self):
        """Generated schema-v1 records exercise the scanner adapter without raw lab logs."""
        events = [
            self.event(1, "history_started", metadata={"purpose": "synthetic offline test", "stage_simulated": False}),
            self.event(2, "camera_connected", camera={"actual_resolution": [1920, 1080]}),
            self.event(3, "session_started", session={"session_id": "synthetic-session", "scan_config": {}}),
            self.event(4, "move_requested", row=0, column=0),
            self.event(5, "position_verified", position=[0, 0]),
            self.event(6, "capture_success", filename="synthetic_0_0.png"),
            self.event(7, "candidate_accepted", human_confirmed=True),
            self.event(8, "move_requested", row=0, column=1),
            self.event(9, "capture_success", filename="synthetic_0_1.png"),
            self.event(10, "candidate_accepted", human_confirmed=True),
            self.event(11, "session_finished", status="failed"),
        ]
        events.extend(self.event(number, "status_checked", status="idle") for number in range(12, 41))
        path = self.history(name="synthetic_schema_test", events=events)
        before = path.read_bytes()
        rows = bridge.scanner_history(self.repo)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "scanner_history")
        self.assertIn("Application history open; latest scan finish: failed", rows[0]["message"]["en"])
        self.assertIn("accepted images: 2", rows[0]["message"]["en"])
        self.assertIn("capture-success events: 2", rows[0]["message"]["en"])
        self.assertIn("human acceptances: 2", rows[0]["message"]["en"])
        self.assertNotIn("XY stage was simulated", rows[0]["message"]["en"])
        self.assertIn("1920 × 1080", rows[0]["message"]["en"])
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
