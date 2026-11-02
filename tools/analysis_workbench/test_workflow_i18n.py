"""Offline synthetic UI checks; no instrument, network or model access."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

HERE = Path(__file__).resolve().parent


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = {"tag": "document", "attrs": {}, "children": [], "text": ""}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": dict(attrs), "children": [], "text": ""}
        self.stack[-1]["children"].append(node)
        if tag not in {"input", "img", "meta", "link", "br", "hr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if self.stack[-1]["tag"] == tag:
            self.stack.pop()

    def handle_data(self, data):
        self.stack[-1]["text"] += data


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for the offline UI fixture")
class WorkflowLanguageTests(unittest.TestCase):
    def test_language_switch_keeps_paths_review_state_and_submission(self):
        parser = PageParser()
        parser.feed((HERE / "static/workflow.html").read_text())
        root = next(node for node in parser.root["children"] if node["tag"] == "html")
        self.assertEqual(root["attrs"]["lang"], "en")
        keys = []

        def inspect(node):
            keys.extend(value for attr, value in node["attrs"].items() if attr.startswith("data-i18n"))
            if re.search(r"[\u4e00-\u9fff]", node["text"]):
                self.assertEqual(node["text"].strip(), "中文", "Default page contains untranslated visible copy")
            for child in node["children"]:
                inspect(child)

        inspect(root)
        keys.extend(re.findall(r"\bt\('([^']+)'", (HERE / "static/workflow.js").read_text()))
        result = subprocess.run([shutil.which("node"), str(HERE / "test_workflow_ui.js")],
                                input=json.dumps({"document": root, "translationKeys": keys}),
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Offline UI checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
