"""Public source bundle integration; no SDK, camera, GUI or child process."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import scan_bridge as bridge


REPO = Path(__file__).resolve().parents[2]
ACQUISITION = REPO / "acquisition"
BUNDLE = ACQUISITION / "Auto_Scan"


class BundledScannerTests(unittest.TestCase):
    def test_delivery_manifest_covers_exact_exported_runtime(self):
        source = json.loads((ACQUISITION / "acquisition_source.json").read_text())
        delivery = json.loads((BUNDLE / "delivery_manifest.json").read_text())
        expected = {str(Path(row["path"]).relative_to("Auto_Scan")): row["sha256"]
                    for row in source["files"] if row["role"] == "runtime"}
        actual = {row["path"]: row["sha256"] for row in delivery["files"]}
        self.assertEqual(actual, expected)
        self.assertEqual(delivery["source_repo_commit"], source["source_commit"])
        self.assertFalse(delivery["hardware_validated"])
        self.assertFalse(delivery["vendor_sdk_included"])
        for relative, digest in actual.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((BUNDLE / relative).read_bytes()).hexdigest(), digest)
        self.assertFalse(list(BUNDLE.rglob("*.dll")))
        self.assertFalse((BUNDLE / "amcam.py").exists())

    def test_source_bundle_uses_scanner_revision_not_parent_git_head(self):
        def outer_git(root, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(REPO)
            raise AssertionError("The enclosing DD repository must not supply the scanner HEAD")

        with patch.object(bridge, "_git", side_effect=outer_git):
            result = bridge.inspect_scanner(ACQUISITION)
        self.assertEqual(result["revision"], "25a839261ae033f068bdcc84507a8cb0e749f760")
        self.assertEqual(result["revision_source"], "delivery_manifest")
        self.assertIsNone(result["dirty"])
        self.assertTrue(result["exists"])
        self.assertFalse(result["launch_blocked"])
        self.assertTrue(result["package"]["package_files_match_manifest"])
        self.assertEqual(result["package"]["checked_files"], 16)
        self.assertIn("Instrument validation of this delivery is pending", result["version_note"]["en"])

    def test_exported_helper_change_blocks_launch_without_sdk_or_process(self):
        with tempfile.TemporaryDirectory() as temp:
            acquisition = Path(temp) / "acquisition"
            bundle = acquisition / "Auto_Scan"
            bundle.mkdir(parents=True)
            data = json.loads((BUNDLE / "delivery_manifest.json").read_text())
            for row in data["files"]:
                shutil.copyfile(BUNDLE / row["path"], bundle / row["path"])
            shutil.copyfile(BUNDLE / "delivery_manifest.json", bundle / "delivery_manifest.json")
            # A mixed helper must be caught even though the main program is intact.
            helper = bundle / "gui_theme.py"
            helper.write_bytes(helper.read_bytes() + b"\n# changed after export\n")
            with patch.object(bridge, "_git", return_value=None), \
                    patch.object(bridge.platform, "system", return_value="Windows"), \
                    patch.object(bridge.subprocess, "Popen") as popen:
                result = bridge.inspect_scanner(acquisition)
                self.assertTrue(result["package"]["main_program_matches_manifest"])
                self.assertTrue(result["launch_blocked"])
                self.assertIn("gui_theme.py", result["version_note"]["en"])
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    bridge.launch_scanner(acquisition, Path(temp) / "logs", "python")
                popen.assert_not_called()
                self.assertFalse((Path(temp) / "logs").exists())


if __name__ == "__main__":
    unittest.main()
