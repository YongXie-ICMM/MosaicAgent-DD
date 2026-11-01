"""Source-bound automatic stitching colour policy; no source writes or camera calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ALGORITHM = "registered-overlap-rgb-tile-v1"


def apply_policy(resolved, mode, data_dir, inventory):
    """Verify an existing derived dataset before deciding not to correct it twice.

    Merely finding a filename is not sufficient: every selected image must match
    the derived manifest, and the manifest must describe exactly that image set.
    Explicit ``off`` retains the historical render path for developer comparisons.
    """
    if mode not in ("auto", "off"):
        raise ValueError(f"Unknown colour correction policy: {mode}")
    resolved["colour_correction"] = mode
    if mode == "off":
        resolved["colour_input"] = {"state": "disabled_by_explicit_request"}
        return resolved
    resolved["colour_algorithm"] = ALGORITHM
    path = Path(data_dir) / "colour_match_manifest.json"
    if not path.exists():
        resolved["colour_input"] = {"state": "raw"}
        return resolved
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Invalid colour manifest; restore the derived dataset or choose the raw scan.")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
        if (manifest.get("kind") != "colour_matched_grid"
                or manifest.get("schema_version") != 2):
            raise ValueError("unsupported colour manifest schema")
        records = manifest["tiles"]
        listed = {}
        for record in records:
            name, sha = record["filename"], record["sha256"]
            if (not isinstance(name, str) or Path(name).name != name
                    or name in listed or not isinstance(sha, str) or len(sha) != 64):
                raise ValueError("invalid or duplicate tile identity")
            listed[name] = sha
        selected = {}
        folder = Path(data_dir).resolve()
        for record in inventory["tiles"]:
            if Path(record["container"]).resolve() != folder or "sha256" not in record:
                raise ValueError("a derived colour manifest must bind loose image files in its own directory")
            if record["inner"] in selected:
                raise ValueError("duplicate selected tile")
            selected[record["inner"]] = record["sha256"]
        if not listed or selected != listed:
            raise ValueError("selected images do not match the colour manifest hashes")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Colour manifest verification failed: {exc}; "
                         "restore/rebuild the derived dataset before stitching.") from exc
    resolved["colour_input"] = {"state": "already_corrected", "manifest": str(path.resolve()),
                                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                                "verified_tiles": len(selected)}
    return resolved
