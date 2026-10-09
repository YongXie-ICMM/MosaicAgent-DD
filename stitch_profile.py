"""Validated, run-scoped stitching geometry and read-only image preflight.

All vectors are in original image pixels: (dx_v, dy_v, dx_h, dy_h).
Image byte size is inventory evidence only; it never determines geometry.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import re
from pathlib import Path
import zipfile

from PIL import Image

HISTORICAL_SIZE = (3840, 2160)
HISTORICAL_VECTORS = (-74.6, 1904.56, 3583.58, 153.4)
RESCALED_SOURCE = "same-field-of-view-rescaled"
CONFIRMATIONS = ("same_optics", "same_field_of_view", "same_stage_steps")


class ProfileError(ValueError):
    """An actionable preflight failure, before cache generation or inference."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in value)):
        raise ProfileError("image_size must be two positive integers [width, height].")
    return tuple(value)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{label} must be a finite number.")
    return float(value)


def historical_profile():
    return {"schema_version": 1, "name": "historical-3840x2160",
            "image_size": list(HISTORICAL_SIZE),
            "nominal_vectors": list(HISTORICAL_VECTORS),
            "geometry_source": "historical-calibration"}


def rescaled_historical_profile(image_size, *, same_optics=False,
                                 same_field_of_view=False, same_stage_steps=False,
                                 name="rescaled-historical"):
    """Rescale only after all three acquisition facts were explicitly confirmed."""
    confirmations = dict(zip(CONFIRMATIONS, (same_optics, same_field_of_view, same_stage_steps)))
    if any(value is not True for value in confirmations.values()):
        raise ProfileError("Historical rescaling requires explicit confirmation of same optics, "
                           "same field of view (no crop), and same stage steps.")
    w, h = _size(image_size)
    if w * HISTORICAL_SIZE[1] != h * HISTORICAL_SIZE[0]:
        raise ProfileError("Historical rescaling requires the same aspect ratio (pure resolution change). "
                           "Use measured vectors for a crop or changed camera geometry.")
    sx, sy = w / HISTORICAL_SIZE[0], h / HISTORICAL_SIZE[1]
    v = HISTORICAL_VECTORS
    return validate_profile({"schema_version": 1, "name": name, "image_size": [w, h],
                             "nominal_vectors": [v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy],
                             "geometry_source": RESCALED_SOURCE,
                             "confirmations": confirmations})


def validate_profile(profile):
    """Return a canonical JSON-safe profile or reject ambiguous/invalid geometry."""
    if not isinstance(profile, dict) or type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ProfileError("Stitch profile must be an object with schema_version: 1.")
    allowed = {"schema_version", "name", "image_size", "nominal_vectors", "geometry_source",
               "scale_div", "out_scale", "confirmations"}
    if set(profile) - allowed:
        raise ProfileError(f"Unknown stitch profile fields: {sorted(set(profile) - allowed)}")
    for key in ("name", "geometry_source"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ProfileError(f"Profile {key} must be a nonempty string.")
    w, h = _size(profile.get("image_size"))
    values = profile.get("nominal_vectors")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ProfileError("nominal_vectors must be [dx_v, dy_v, dx_h, dy_h] in original pixels.")
    dxv, dyv, dxh, dyh = [_number(x, "nominal_vectors") for x in values]
    if dyv <= 0 or dxh <= 0:
        raise ProfileError("dy_v and dx_h must be positive for down-row/right-column progression.")
    if abs(dxv) >= w or dyv >= h or dxh >= w or abs(dyh) >= h:
        raise ProfileError("Both nominal neighbor vectors must retain positive image overlap on both axes.")
    if dxh * dyv - dxv * dyh <= 1e-9:
        raise ProfileError("Nominal row/column vectors must form a nondegenerate forward grid.")
    result = {"schema_version": 1, "name": profile["name"].strip(), "image_size": [w, h],
              "nominal_vectors": [dxv, dyv, dxh, dyh],
              "geometry_source": profile["geometry_source"].strip()}
    if result["geometry_source"] == RESCALED_SOURCE:
        if w * HISTORICAL_SIZE[1] != h * HISTORICAL_SIZE[0]:
            raise ProfileError("Rescaled historical profile must retain the same aspect ratio; "
                               "use measured vectors for changed camera geometry.")
        confirmations = profile.get("confirmations", {})
        if not isinstance(confirmations, dict) or any(confirmations.get(k) is not True for k in CONFIRMATIONS):
            raise ProfileError("Rescaled historical profiles require all three explicit confirmations: "
                               + ", ".join(CONFIRMATIONS))
        sx, sy = w / HISTORICAL_SIZE[0], h / HISTORICAL_SIZE[1]
        expected = [HISTORICAL_VECTORS[0] * sx, HISTORICAL_VECTORS[1] * sy,
                    HISTORICAL_VECTORS[2] * sx, HISTORICAL_VECTORS[3] * sy]
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
               for a, b in zip(result["nominal_vectors"], expected)):
            raise ProfileError("Rescaled historical vectors do not match the stated image dimensions.")
        result["confirmations"] = {k: True for k in CONFIRMATIONS}
    elif "confirmations" in profile:
        raise ProfileError("confirmations are only used for same-field-of-view-rescaled profiles.")
    if result["geometry_source"] == "historical-calibration" and (
            (w, h) != HISTORICAL_SIZE or tuple(result["nominal_vectors"]) != HISTORICAL_VECTORS):
        raise ProfileError("Historical calibration is only valid for its original 3840x2160 geometry.")
    for key in ("scale_div", "out_scale"):
        if key in profile:
            result[key] = profile[key]
    _scales(result, result.get("scale_div", 1), result.get("out_scale", 1 / 8), full=True)
    return result


def load_profile(path):
    try:
        return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Cannot read stitch profile {path}: {exc}") from exc


def _scales(profile, scale_div, out_scale, full):
    w, h = profile["image_size"]
    if type(scale_div) is not int or scale_div < 1:
        raise ProfileError("scale_div must be a positive integer.")
    if w % scale_div or h % scale_div:
        raise ProfileError("scale_div must divide both image dimensions exactly; choose a divisor "
                           "(for example 1) to preserve the registration coordinate scale.")
    if min(w // scale_div, h // scale_div) < 8:
        raise ProfileError("scale_div produces tiles smaller than 8 pixels; choose a smaller divisor.")
    out_scale = _number(out_scale, "out_scale")
    if not 0 < out_scale <= 1:
        raise ProfileError("out_scale must be greater than 0 and at most 1.")
    if min(w, h) * out_scale < 1:
        raise ProfileError("out_scale would produce a tile smaller than one output pixel.")
    if not full and out_scale > 1 / scale_div + 1e-12:
        raise ProfileError("out_scale exceeds cached resolution; reduce it or use --full.")
    return scale_div, out_scale


def resolve_profile(profile, inventory, scale_div=None, out_scale=None, full=False):
    actual = _size(inventory.get("image_size"))
    if profile is None:
        if actual != HISTORICAL_SIZE:
            raise ProfileError(f"Tiles are {actual[0]}x{actual[1]}; historical geometry is 3840x2160. "
                               "Provide --stitch-profile with measured vectors, or explicitly confirm "
                               "same optics/field of view/stage steps before rescaling. File MB cannot calibrate geometry.")
        profile = historical_profile()
    profile = validate_profile(profile)
    if tuple(profile["image_size"]) != actual:
        raise ProfileError(f"Profile image_size {profile['image_size']} does not match actual tile headers {list(actual)}.")
    sd = scale_div if scale_div is not None else profile.get("scale_div", 8)
    scale = out_scale if out_scale is not None else profile.get("out_scale", 1 / 8)
    sd, scale = _scales(profile, sd, scale, full)
    warnings = []
    if profile["geometry_source"] in ("historical-calibration", RESCALED_SOURCE):
        warnings.append("Historical calibration is a search prior, not a measurement of this acquisition; "
                        "verify alignment on representative adjacent raw tiles.")
    min_strip = max(8, 0.03 * max(actual) / sd)
    dxv, dyv, dxh, dyh = profile["nominal_vectors"]
    if min(actual[0] - abs(dxv), actual[1] - dyv,
           actual[0] - dxh, actual[1] - abs(dyh)) / sd < min_strip:
        warnings.append("Nominal overlap at this cache scale is thinner than the registration reliability threshold; "
                        "use a smaller scale_div or acquire more overlap and verify pair alignment.")
    return {"profile": profile, "scale_div": sd, "out_scale": scale, "full": bool(full), "warnings": warnings}


def scan_grid_tiles(data_dir, *, ignored_paths=()):
    """Recognize the scanner's flat physical-coordinate grid without any AI.

    Returns None when this convention is absent. A partially recognized grid is
    an error, never permission to omit unrecognized images or missing cells.
    Tile.direction is logical physical ordering, not reconstructed acquisition order.
    """
    import tiles as T
    folder = Path(data_dir).resolve()
    rx = re.compile(r"^mosaic_r(?P<row>\d+)_c(?P<col>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$", re.I)
    images = [p for p in sorted(folder.iterdir()) if p.is_file()
              and not p.name.startswith(".") and p.suffix.lower().lstrip(".") in T.IMAGE_EXTS]
    matched = [(p, rx.fullmatch(p.name)) for p in images]
    if not any(match for _, match in matched):
        return None
    ignored = {Path(p).resolve() for p in ignored_paths}
    for entry in folder.iterdir():
        if (entry.name.startswith((".", "_")) or entry.name == "review_candidates"
                or entry.resolve() in ignored):
            continue
        if entry.is_file() and entry.suffix.lower() == ".zip":
            raise ProfileError(f"ZIP container {entry.name} alongside flat-grid images; "
                               "select one acquisition in a separate directory.")
        if entry.is_dir():
            nested, _ = T._list_container(str(entry), False)
            if any(Path(name).suffix.lower().lstrip(".") in T.IMAGE_EXTS for name, _, _ in nested):
                raise ProfileError(f"Unexpected nested image container {entry.name} alongside flat-grid images. "
                                   "Only documented review_candidates are excluded from the acquisition.")
    unknown = [p.name for p, match in matched if match is None]
    if unknown:
        raise ProfileError("Unrecognized images alongside mosaic_r<row>_c<col> tiles: "
                           + ", ".join(unknown[:10]) + ". Use a directory containing only this acquisition.")
    grid = {}
    for path, match in matched:
        coord = (int(match["row"]), int(match["col"]))
        if coord in grid:
            raise ProfileError(f"Duplicate grid coordinate {coord}: {grid[coord].name}, {path.name}")
        grid[coord] = path
    rows, cols = sorted({r for r, _ in grid}), sorted({c for _, c in grid})
    if len(rows) < 2 or len(cols) < 2:
        raise ProfileError("Flat-grid stitching requires at least 2 rows and 2 columns of adjacent raw tiles.")
    if rows != list(range(rows[0], rows[-1] + 1)) or cols != list(range(cols[0], cols[-1] + 1)):
        raise ProfileError("Grid row/column indices are not contiguous; restore missing raw tiles before stitching.")
    missing = [(r, c) for c in cols for r in rows if (r, c) not in grid]
    if missing:
        raise ProfileError(f"Incomplete raw tile grid; missing coordinates: {missing[:20]}")
    result = []
    for ci, col in enumerate(cols):
        for ri, row in enumerate(rows):
            path = grid[row, col]
            result.append(T.Tile(col=col, col_idx=ci, direction="down", order=ri,
                                 key=float(row), name=path.name, zip_path=str(folder),
                                 inner=path.name, nominal_row=ri, nbytes=path.stat().st_size))
    return result


def _stat(path):
    s = path.stat()
    return {"path": str(path.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns, "device": s.st_dev, "inode": s.st_ino}


def _container_stat(path):
    evidence = _stat(path)
    if path.is_dir():
        # Creating a work subdirectory changes directory timestamps, not raw images.
        return {key: evidence[key] for key in ("path", "device", "inode")}
    return evidence


def _header(stream):
    with Image.open(stream) as image:
        size, fmt, mode = list(image.size), image.format, image.mode
        if mode not in ("RGB", "L"):
            raise ProfileError(f"Unsupported tile pixel mode {mode!r}. This pipeline decodes to 8-bit color; "
                               "use consistently prepared 8-bit RGB or grayscale tiles and preserve the originals.")
        image.verify()
    return size, fmt, mode


def inspect_tiles(tiles):
    """Inspect every selected image in ZIPs/directories; never decode into a cache.

    Directory images are SHA-256 hashed. ZIP inventory includes member CRC and
    container path/stat identity. Evidence also binds tile order and grid indices.
    """
    tiles = list(tiles)
    if not tiles:
        raise ProfileError("No raw tiles selected. A prestitched mosaic cannot calibrate tile geometry.")
    records, sizes, modes, containers, archives = [], {}, {}, {}, {}
    seen = set()
    try:
        for tile in tiles:
            path = Path(tile.zip_path).resolve()
            label = f"{path.name}/{tile.inner}"
            tid = str(tile.tid)
            if tid in seen:
                raise ProfileError(f"Duplicate tile/cache identity {tid}; correct the layout before stitching.")
            seen.add(tid)
            try:
                if str(path) not in containers:
                    containers[str(path)] = _container_stat(path)
                record = {"tid": tid, "container": str(path), "inner": tile.inner,
                          "grid": {k: getattr(tile, k) for k in ("col", "col_idx", "direction", "order", "key", "nominal_row")
                                   if hasattr(tile, k)}}
                if path.is_dir():
                    source = (path / tile.inner).resolve()
                    if not source.is_relative_to(path):
                        raise ProfileError(f"Tile escapes its source directory: {label}")
                    before = _stat(source)
                    sha = hashlib.sha256()
                    with source.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            sha.update(chunk)
                        stream.seek(0)
                        size, fmt, mode = _header(stream)
                    if _stat(source) != before:
                        raise ProfileError(f"Input changed during preflight: {label}; retry after acquisition stops.")
                    record.update(source=before, sha256=sha.hexdigest())
                else:
                    if str(path) not in archives:
                        archives[str(path)] = zipfile.ZipFile(path)
                    archive = archives[str(path)]
                    info = archive.getinfo(tile.inner)
                    if sum(entry.filename == tile.inner for entry in archive.infolist()) != 1:
                        raise ProfileError(f"Duplicate ZIP member: {label}")
                    with archive.open(info) as stream:
                        size, fmt, mode = _header(stream)
                    record.update(crc=info.CRC, uncompressed_bytes=info.file_size,
                                  compressed_bytes=info.compress_size)
                record.update(image_size=size, format=fmt, image_mode=mode)
                records.append(record)
                sizes.setdefault(tuple(size), []).append(label)
                modes.setdefault(mode, []).append(label)
            except ProfileError:
                raise
            except Exception as exc:
                raise ProfileError(f"Unreadable tile image {label}: {exc}") from exc
        for key, evidence in containers.items():
            if _container_stat(Path(key)) != evidence:
                raise ProfileError(f"Input container changed during preflight: {key}; retry after acquisition stops.")
    finally:
        for archive in archives.values():
            archive.close()
    if len(sizes) != 1:
        detail = "; ".join(f"{w}x{h}: {', '.join(names[:3])}" for (w, h), names in sizes.items())
        raise ProfileError(f"Mixed tile resolutions are not supported: {detail}. Split acquisitions into separate runs.")
    if len(modes) != 1:
        detail = "; ".join(f"{mode}: {', '.join(names[:3])}" for mode, names in modes.items())
        raise ProfileError(f"Mixed tile pixel modes are not supported: {detail}. Use a consistent acquisition.")
    evidence = {"image_size": list(next(iter(sizes))), "image_mode": next(iter(modes)), "tile_count": len(records),
                "containers": list(containers.values()), "tiles": records}
    return {**evidence, "fingerprint": _digest(evidence)}


def make_run_binding(resolved, inventory, acquisition=None):
    values = {key: resolved[key] for key in ("profile", "scale_div", "out_scale", "full")}
    values.update(schema_version=1, input_fingerprint=inventory["fingerprint"])
    # Keep legacy bindings compatible, but never reuse them for grid protection.
    if resolved.get("grid_policy", "legacy") != "legacy":
        values["grid_policy"] = resolved["grid_policy"]
    # Preserve historical bindings byte-for-byte when no acquisition record exists.
    if acquisition and acquisition.get("present"):
        values["acquisition_fingerprint"] = _digest(acquisition)
    return {**values, "fingerprint": _digest(values)}



def check_input_preflight(path, resolved, inventory, acquisition):
    """Bind a prepared command to its inspected inputs, metadata and geometry."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ProfileError("Input preflight must be a prepared JSON report smaller than 32 MiB.")
    def invalid(value):
        raise ProfileError(f"Non-finite value in input preflight: {value}")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Cannot read input preflight: {exc}") from exc
    if not isinstance(saved, dict) or saved.get("schema_version") != 1 or saved.get("status") != "profile_prepared":
        raise ProfileError("--input-preflight needs the prepared bundle's preflight.json.")
    previous_inventory = saved.get("input_inventory")
    if not isinstance(previous_inventory, dict) or previous_inventory.get("fingerprint") != inventory["fingerprint"]:
        raise ProfileError("Raw inputs changed after profile preparation. Inspect and prepare again.")
    previous = saved.get("resolved")
    if not isinstance(previous, dict) or any(previous.get(key) != resolved[key]
            for key in ("profile", "scale_div", "out_scale", "full")):
        raise ProfileError("Profile or scale settings changed after preparation. Inspect and prepare again.")
    inspection = saved.get("inspection")
    if not isinstance(inspection, dict) or inspection.get("acquisition") != acquisition:
        raise ProfileError("Acquisition metadata changed after preparation. Inspect and prepare again.")


def check_work_binding(work, binding):
    """Reject stale/legacy populated work directories; --force is no bypass."""
    work = Path(work)
    if not work.exists():
        return
    state_path = work / "state.json"
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8")).get("run_binding")
        except (OSError, ValueError, AttributeError) as exc:
            raise ProfileError(f"Cannot validate existing state in {work}: {exc}. Choose a new --work directory.") from exc
        if not isinstance(previous, dict) or previous != binding:
            raise ProfileError("Existing work state is legacy/unbound or its profile, scale, rendering mode, "
                               "or input inventory changed. Choose a new --work directory; --force cannot bypass this check.")
    else:
        # Configuration and read-only preflight reports may precede a first run.
        allowed = {"layout.json", "layout_rejected.json", "stitch_profile.json", "profile.json", "preflight.json"}
        if any(p.name not in allowed for p in work.iterdir()):
            raise ProfileError("Populated work directory has no bound state. Choose a new --work directory "
                               "to avoid reusing legacy caches; --force cannot bypass this check.")


@contextmanager
def geometry_context(profile):
    """Apply validated geometry to all legacy import copies, restoring on exit.

    Intended for the single-run CLI. Concurrent jobs must use separate processes.
    """
    profile = validate_profile(profile)
    import tiles
    import register
    import blend
    w, h = profile["image_size"]
    dxv, dyv, dxh, dyh = profile["nominal_vectors"]
    updates = {tiles: {"TILE_W": w, "TILE_H": h, "NOMINAL_PITCH_X": dxh,
                       "NOMINAL_PITCH_Y": dyv, "SHEAR_PER_ROW": -dxv},
               register: {"TILE_W": w, "TILE_H": h, "NOMINAL_FULL": tuple(profile["nominal_vectors"]),
                          "SHEAR_X_PER_ROW": dxv, "SHEAR_Y_PER_COL": dyh},
               blend: {"TILE_W": w, "TILE_H": h}}
    previous = [(module, key, getattr(module, key)) for module, attrs in updates.items() for key in attrs]
    try:
        for module, attrs in updates.items():
            for key, value in attrs.items():
                setattr(module, key, value)
        yield
    finally:
        for module, key, value in previous:
            setattr(module, key, value)
