#!/usr/bin/env python3
"""Build (or check) the complete one-click student bundle ``MosaicAgent-DD-student-complete.zip``.

The Git repository is source only. The bundle adds the trusted assets that students need
to run the launchers without any other download:

    data/demo/source_images.zip   two original 1920 x 1080 camera frames
    data/demo/assets.json         SHA-256 bindings of the images and the checkpoint
    weights/model_0409_all.pth    the unchanged 0409 checkpoint

Everything tracked by Git is included (code, launchers, docs, tests, acquisition
helpers); generated outputs, virtual environments and caches are not. Every asset is
verified against ``data/demo/assets.json`` before packing, every packed file is listed
with its SHA-256 in ``bundle_manifest.json`` inside the archive, and the archive is
re-read and checked after writing. ``--check`` verifies an existing bundle the same way.

Teacher's tool: run from the repository root with the assets in place:

    python tools/build_student_bundle.py                 # -> dist/MosaicAgent-DD-student-complete.zip
    python tools/build_student_bundle.py --check dist/MosaicAgent-DD-student-complete.zip

Owner decision 2026-10-05: the group's students run the stitching with Kimi, as in the
original pipeline, so an internal variant can carry the group's ``.env``:

    python tools/build_student_bundle.py --with-env /path/to/5_AI_SYSTEM/.env
        # -> dist/MosaicAgent-DD-student-complete-with-kimi.zip  (credentials_included: true)

That archive is for the group only: never attach it to a GitHub release or send it outside.
The public bundle stays key-free; ``.env`` remains gitignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "MosaicAgent-DD-student-complete.zip"
INTERNAL_BUNDLE_NAME = "MosaicAgent-DD-student-complete-with-kimi.zip"   # carries .env; internal distribution only
TOP_LEVEL = "MosaicAgent-DD"
MANIFEST_NAME = "bundle_manifest.json"
ASSET_MANIFEST = "data/demo/assets.json"
REQUIRED_LAUNCHERS = ("01_install.bat", "01_install.command", "02_run_layer_demo.bat", "02_run_layer_demo.command",
                      "03_open_workbench.bat", "03_open_workbench.command")
# Tracked files also surfaced at the top level of the archive so students see them first.
TOP_LEVEL_COPIES = {"00_STUDENT_GUIDE_zh.pdf": "docs/STUDENT_GUIDE_zh.pdf"}
EXCLUDED_TOP_LEVEL = {".github", ".gitattributes"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "outputs", "dist", "_work", "_stitch_work",
                 "cache", "kimi_cache", "_kimi_cache", "node_modules", "workbench_history", "history", "scan_runs",
                 "drivers", "camera_history", "colour_calibration", "shared_history"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".orig", ".rej", ".DS_Store"}


def is_secret_file(name: str) -> bool:
    """Credential files are never packed by enumeration; only --with-env adds one deliberately."""
    if name == ".env.example":
        return False
    return name == ".env" or name.endswith(".env") or name.startswith(".env.") or name.endswith(".local.json")
STORED_SUFFIXES = {".zip", ".pth", ".pt", ".png", ".jpg", ".jpeg", ".pdf"}
SCHEMA_VERSION = 1


class BundleError(ValueError):
    """An actionable problem; nothing was written when it is raised before packing."""


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=repo, capture_output=True, text=True,
                                timeout=60, env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def tracked_files(repo: Path) -> list[str] | None:
    listing = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if listing is None:
        return None
    return sorted(p for p in listing.split("\0") if p)


def walked_files(repo: Path) -> list[str]:
    found = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix in EXCLUDED_SUFFIXES or name.endswith("~"):
                continue
            found.append(path.relative_to(repo).as_posix())
    return sorted(found)


def select_files(repo: Path) -> tuple[list[str], str]:
    """Source files to pack and how they were enumerated."""
    files = tracked_files(repo)
    source = "git ls-files (tracked and untracked-not-ignored)"
    if files is None:
        files, source = walked_files(repo), "directory walk with exclusions (git unavailable)"
    selected = []
    for relative in files:
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(part in EXCLUDED_DIRS for part in parts[:-1]):
            continue
        if PurePosixPath(relative).suffix in EXCLUDED_SUFFIXES or is_secret_file(parts[-1]):
            continue
        if not (repo / relative).is_file():
            continue
        selected.append(relative)
    return sorted(set(selected)), source


def verify_assets(repo: Path) -> dict:
    """The three trusted assets must exist and match data/demo/assets.json."""
    manifest_path = repo / ASSET_MANIFEST
    if not manifest_path.is_file():
        raise BundleError(f"Missing {ASSET_MANIFEST}: this checkout has no student assets; the bundle cannot be built here.")
    assets = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    report = {"manifest": ASSET_MANIFEST, "manifest_sha256": sha256_file(manifest_path), "files": []}
    for key in ("archive", "weights"):
        record = assets.get(key) or {}
        relative, expected = record.get("path"), record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise BundleError(f"{ASSET_MANIFEST} has no usable '{key}' record")
        path = repo / relative
        if not path.is_file():
            raise BundleError(f"Missing asset {relative} (listed in {ASSET_MANIFEST})")
        actual = sha256_file(path)
        if actual != expected:
            raise BundleError(f"Asset {relative} does not match {ASSET_MANIFEST}: {actual} != {expected}")
        report["files"].append({"path": relative, "sha256": actual, "bytes": path.stat().st_size, "verified": True})
    return report


def build_manifest(repo: Path, files: list[str], source: str, assets: dict, with_env: Path | None = None) -> dict:
    head = (git(repo, "rev-parse", "HEAD") or "").strip() or None
    branch = (git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
    status = git(repo, "status", "--porcelain")
    dirty = None if status is None else bool(status.strip())
    records = []
    total = 0
    for relative in files:
        path = repo / relative
        size = path.stat().st_size
        total += size
        records.append({"path": relative, "sha256": sha256_file(path), "bytes": size})
    for alias, source in sorted(TOP_LEVEL_COPIES.items()):
        if source in files and alias not in files:
            path = repo / source
            size = path.stat().st_size
            total += size
            records.append({"path": alias, "source": source, "sha256": sha256_file(path), "bytes": size})
    if with_env is not None:
        size = with_env.stat().st_size
        total += size
        records.append({"path": ".env", "source": str(with_env), "sha256": sha256_file(with_env), "bytes": size,
                        "credentials": True})
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "student_bundle_manifest",
        "bundle_name": INTERNAL_BUNDLE_NAME if with_env is not None else BUNDLE_NAME,
        "credentials_included": with_env is not None,
        "distribution": "internal_group_only" if with_env is not None else "public_release",
        "top_level_folder": TOP_LEVEL,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head, "source_branch": branch, "working_tree_dirty": dirty,
        "file_enumeration": source,
        "builder": {"path": "tools/build_student_bundle.py", "sha256": sha256_file(Path(__file__)),
                    "python": platform.python_version(), "os": platform.platform()},
        "assets": assets,
        "file_count": len(records), "total_bytes": total,
        "files": records,
        "notes": [
            "Everything tracked by Git plus the verified trusted assets; outputs, environments and caches are not packed.",
            "Records with a 'source' field are top-level copies of tracked files (the student manual PDF).",
        ] + ([
            "INTERNAL: this archive contains the group's Kimi credentials (.env) so that the stitching runs with Kimi "
            "as in the original pipeline. Never attach it to a GitHub release or send it outside the group.",
        ] if with_env is not None else []) + [
            "Extract the whole folder; run 01_install, then 02_run_layer_demo, then 03_open_workbench (see 00_STUDENT_GUIDE_zh.pdf).",
            "Scanner runtime files inside acquisition/Auto_Scan are unchanged and hash-checked by delivery_manifest.json.",
        ],
    }


def write_bundle(repo: Path, out_path: Path, manifest: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".building")
    if temp.exists():
        temp.unlink()
    stamp = datetime.now(timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for record in manifest["files"]:
            relative = record["path"]
            source = Path(record.get("source", relative))
            if not source.is_absolute():
                source = repo / source
            info = zipfile.ZipInfo(f"{TOP_LEVEL}/{relative}", date_time=stamp)
            info.compress_type = zipfile.ZIP_STORED if PurePosixPath(relative).suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            mode = 0o100755 if relative.endswith((".command", ".sh")) else 0o100644
            info.external_attr = mode << 16
            with source.open("rb") as stream, archive.open(info, "w") as target:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    target.write(chunk)
        info = zipfile.ZipInfo(f"{TOP_LEVEL}/{MANIFEST_NAME}", date_time=stamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, out_path)


def check_bundle(zip_path: Path) -> dict:
    """Re-read an archive: CRC check, manifest present, every listed file present with its hash."""
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise BundleError(f"Bundle not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise BundleError(f"Corrupt member in {zip_path.name}: {bad}")
        names = set(archive.namelist())
        manifest_member = f"{TOP_LEVEL}/{MANIFEST_NAME}"
        if manifest_member not in names:
            raise BundleError(f"{MANIFEST_NAME} is missing from {zip_path.name}")
        manifest = json.loads(archive.read(manifest_member).decode("utf-8"))
        problems = []
        for record in manifest["files"]:
            member = f"{TOP_LEVEL}/{record['path']}"
            if member not in names:
                problems.append(f"missing: {record['path']}")
                continue
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                problems.append(f"hash mismatch: {record['path']}")
        listed = {f"{TOP_LEVEL}/{r['path']}" for r in manifest["files"]} | {manifest_member}
        extra = sorted(n for n in names if not n.endswith("/") and n not in listed)
        if extra:
            problems.append("unlisted members: " + ", ".join(extra[:5]))
        for launcher in REQUIRED_LAUNCHERS:
            if f"{TOP_LEVEL}/{launcher}" not in names:
                problems.append(f"launcher missing: {launcher}")
        for asset in manifest.get("assets", {}).get("files", []):
            if f"{TOP_LEVEL}/{asset['path']}" not in names:
                problems.append(f"asset missing: {asset['path']}")
        if problems:
            raise BundleError(f"{zip_path.name} failed verification: " + "; ".join(problems))
    return {"bundle": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path),
            "file_count": manifest["file_count"], "source_commit": manifest.get("source_commit"),
            "built_at_utc": manifest.get("built_at_utc"),
            "credentials_included": bool(manifest.get("credentials_included"))}


def verify_env_file(path: Path) -> Path:
    """The .env must exist and define a non-empty KIMI_API_KEY; the value is never printed."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise BundleError(f"--with-env file not found: {path}")
    keys = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            keys[key.strip()] = value.strip().strip('"').strip("'")
    if not keys.get("KIMI_API_KEY"):
        raise BundleError(f"--with-env file has no KIMI_API_KEY: {path}")
    return path


def build(repo: Path = REPO, out_dir: Path | None = None, with_env: Path | None = None) -> dict:
    repo = Path(repo).resolve()
    out_dir = Path(out_dir) if out_dir is not None else repo / "dist"
    env_path = verify_env_file(with_env) if with_env is not None else None
    assets = verify_assets(repo)
    files, source = select_files(repo)
    for relative in (ASSET_MANIFEST, *(a["path"] for a in assets["files"])):
        if relative not in files:
            files.append(relative)
    files = sorted(set(files))
    missing = [l for l in REQUIRED_LAUNCHERS if l not in files]
    if missing:
        raise BundleError("Launchers missing from the checkout: " + ", ".join(missing))
    manifest = build_manifest(repo, files, source, assets, env_path)
    name = manifest["bundle_name"]
    out_path = out_dir / name
    write_bundle(repo, out_path, manifest)
    result = check_bundle(out_path)
    (out_dir / (name + ".sha256")).write_text(f"{result['sha256']}  {name}\n", encoding="utf-8")
    (out_dir / name.replace(".zip", ".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                                    encoding="utf-8")
    result["credentials_included"] = manifest["credentials_included"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/dist")
    parser.add_argument("--check", type=Path, default=None, metavar="ZIP", help="verify an existing bundle instead of building")
    parser.add_argument("--with-env", type=Path, default=None, metavar="ENV_FILE",
                        help="also pack this .env (Kimi credentials) as MosaicAgent-DD/.env into the INTERNAL "
                             "variant MosaicAgent-DD-student-complete-with-kimi.zip; never publish that archive")
    args = parser.parse_args(argv)
    try:
        if args.check is not None:
            result = check_bundle(args.check)
            print(f"OK  {result['bundle']}  {result['bytes']:,} bytes  sha256 {result['sha256']}  "
                  f"{result['file_count']} files  commit {result['source_commit']}"
                  + ("  [INTERNAL: credentials included]" if result["credentials_included"] else ""))
            return 0
        result = build(args.repo, args.out_dir, args.with_env)
        print(f"Built {result['bundle']}  {result['bytes']:,} bytes  sha256 {result['sha256']}  "
              f"{result['file_count']} files  commit {result['source_commit']}")
        if result["credentials_included"]:
            print("INTERNAL ARCHIVE: contains .env with Kimi credentials. Share only inside the group; "
                  "do not attach to a GitHub release.")
        return 0
    except BundleError as exc:
        print(f"Bundle not built: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
