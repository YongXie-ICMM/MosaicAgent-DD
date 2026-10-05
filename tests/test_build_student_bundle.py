"""The student bundle builder packs tracked sources plus verified assets and checks itself."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_student_bundle as bundle  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_repo(tmp_path: Path, *, break_weights=False) -> Path:
    repo = tmp_path / "repo"
    for launcher in bundle.REQUIRED_LAUNCHERS:
        (repo / launcher).parent.mkdir(parents=True, exist_ok=True)
        (repo / launcher).write_text("echo demo\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "student_demo.py").write_text("print('demo')\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "STUDENT_GUIDE_zh.md").write_text("# guide\n", encoding="utf-8")
    (repo / "docs" / "STUDENT_GUIDE_zh.pdf").write_bytes(b"%PDF-1.4 fake manual\n")
    (repo / "outputs" / "demo").mkdir(parents=True)
    (repo / "outputs" / "demo" / "latest.json").write_text("{}", encoding="utf-8")       # must not be packed
    (repo / "tools" / "__pycache__").mkdir()
    (repo / "tools" / "__pycache__" / "x.pyc").write_bytes(b"\0")                         # must not be packed
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")  # must not be packed
    (repo / "data" / "demo").mkdir(parents=True)
    images = b"PK\x05\x06" + bytes(18)                                                      # empty zip archive
    (repo / "data" / "demo" / "source_images.zip").write_bytes(images)
    (repo / "weights").mkdir()
    weights = b"weights-bytes" * 100
    (repo / "weights" / "model_0409_all.pth").write_bytes(weights)
    assets = {"schema_version": 1,
              "archive": {"path": "data/demo/source_images.zip", "sha256": sha(images)},
              "weights": {"path": "weights/model_0409_all.pth", "sha256": sha(b"other" if break_weights else weights)},
              "samples": []}
    (repo / "data" / "demo" / "assets.json").write_text(json.dumps(assets), encoding="utf-8")
    return repo


def test_build_packs_sources_and_verified_assets_and_checks_itself(tmp_path):
    repo = make_repo(tmp_path)
    result = bundle.build(repo, tmp_path / "dist")
    zip_path = Path(result["bundle"])
    assert zip_path.name == bundle.BUNDLE_NAME
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert f"{bundle.TOP_LEVEL}/weights/model_0409_all.pth" in names
        assert f"{bundle.TOP_LEVEL}/data/demo/source_images.zip" in names
        assert f"{bundle.TOP_LEVEL}/data/demo/assets.json" in names
        assert f"{bundle.TOP_LEVEL}/docs/STUDENT_GUIDE_zh.md" in names
        assert f"{bundle.TOP_LEVEL}/00_STUDENT_GUIDE_zh.pdf" in names, "manual PDF is surfaced at the top level"
        assert archive.read(f"{bundle.TOP_LEVEL}/00_STUDENT_GUIDE_zh.pdf") == archive.read(f"{bundle.TOP_LEVEL}/docs/STUDENT_GUIDE_zh.pdf")
        assert f"{bundle.TOP_LEVEL}/01_install.command" in names
        assert not any("outputs/" in n or "__pycache__" in n or ".github" in n for n in names)
        command = archive.getinfo(f"{bundle.TOP_LEVEL}/01_install.command")
        assert (command.external_attr >> 16) & 0o111, "macOS launchers must stay executable"
        manifest = json.loads(archive.read(f"{bundle.TOP_LEVEL}/{bundle.MANIFEST_NAME}"))
    assert manifest["kind"] == "student_bundle_manifest"
    assert manifest["file_count"] == len(manifest["files"]) == len(names) - 1
    assert all(len(r["sha256"]) == 64 for r in manifest["files"])
    copies = [r for r in manifest["files"] if r.get("source")]
    assert copies == [{"path": "00_STUDENT_GUIDE_zh.pdf", "source": "docs/STUDENT_GUIDE_zh.pdf",
                       "sha256": sha(b"%PDF-1.4 fake manual\n"), "bytes": len(b"%PDF-1.4 fake manual\n")}]
    assert {a["path"] for a in manifest["assets"]["files"]} == {"data/demo/source_images.zip", "weights/model_0409_all.pth"}
    assert (tmp_path / "dist" / (bundle.BUNDLE_NAME + ".sha256")).read_text().split()[0] == result["sha256"]
    assert bundle.check_bundle(zip_path)["file_count"] == manifest["file_count"]


def test_asset_hash_mismatch_stops_before_writing(tmp_path):
    repo = make_repo(tmp_path, break_weights=True)
    with pytest.raises(bundle.BundleError, match="does not match"):
        bundle.build(repo, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


def test_missing_assets_are_an_explicit_error(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "weights" / "model_0409_all.pth").unlink()
    with pytest.raises(bundle.BundleError, match="Missing asset"):
        bundle.build(repo, tmp_path / "dist")


def test_check_detects_a_tampered_member(tmp_path):
    repo = make_repo(tmp_path)
    zip_path = Path(bundle.build(repo, tmp_path / "dist")["bundle"])
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zip_path) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info)
            if info.filename.endswith("student_demo.py"):
                data = b"print('changed')\n"
            target.writestr(info, data)
    with pytest.raises(bundle.BundleError, match="hash mismatch"):
        bundle.check_bundle(tampered)


def test_cli_check_reports_ok(tmp_path, capsys):
    repo = make_repo(tmp_path)
    zip_path = bundle.build(repo, tmp_path / "dist")["bundle"]
    assert bundle.main(["--check", zip_path]) == 0
    assert capsys.readouterr().out.startswith("OK")
    assert bundle.main(["--check", str(tmp_path / "missing.zip")]) == 1


def test_real_repository_selection_excludes_outputs_and_includes_launchers():
    files, _ = bundle.select_files(ROOT)
    assert all(not f.startswith(("outputs/", ".github/", "dist/")) for f in files)
    assert all(l in files for l in bundle.REQUIRED_LAUNCHERS)
    assert "tools/student_demo.py" in files and "acquisition/Auto_Scan/wb_calibrate.py" in files
    assert all(not f.endswith(".pyc") for f in files)


def test_public_bundle_never_contains_an_env_file_even_if_one_lies_in_the_checkout(tmp_path):
    repo = make_repo(tmp_path)
    (repo / ".env").write_text("KIMI_API_KEY=secret\n", encoding="utf-8")
    result = bundle.build(repo, tmp_path / "dist")
    with zipfile.ZipFile(result["bundle"]) as archive:
        assert not any(n.endswith("/.env") for n in archive.namelist())
    assert result["credentials_included"] is False


def test_with_env_builds_a_separately_named_internal_bundle(tmp_path):
    repo = make_repo(tmp_path)
    env_file = tmp_path / "lab.env"
    env_file.write_text("KIMI_API_KEY=secret-key\nKIMI_BASE_URL=https://api.moonshot.cn/v1\n", encoding="utf-8")
    result = bundle.build(repo, tmp_path / "dist", env_file)
    assert Path(result["bundle"]).name == bundle.INTERNAL_BUNDLE_NAME
    assert result["credentials_included"] is True
    with zipfile.ZipFile(result["bundle"]) as archive:
        assert archive.read(f"{bundle.TOP_LEVEL}/.env") == env_file.read_bytes()
        manifest = json.loads(archive.read(f"{bundle.TOP_LEVEL}/{bundle.MANIFEST_NAME}"))
    assert manifest["credentials_included"] and manifest["distribution"] == "internal_group_only"
    assert any("INTERNAL" in n for n in manifest["notes"])
    env_record = [r for r in manifest["files"] if r["path"] == ".env"][0]
    assert env_record["credentials"] is True and "secret-key" not in json.dumps(manifest)
    assert bundle.check_bundle(result["bundle"])["credentials_included"] is True
    # the public bundle built from the same checkout is untouched and key-free
    public = bundle.build(repo, tmp_path / "dist")
    assert Path(public["bundle"]).name == bundle.BUNDLE_NAME and public["credentials_included"] is False


def test_with_env_requires_a_real_key(tmp_path):
    repo = make_repo(tmp_path)
    empty = tmp_path / "empty.env"
    empty.write_text("KIMI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="no KIMI_API_KEY"):
        bundle.build(repo, tmp_path / "dist", empty)
    with pytest.raises(bundle.BundleError, match="not found"):
        bundle.build(repo, tmp_path / "dist", tmp_path / "missing.env")
