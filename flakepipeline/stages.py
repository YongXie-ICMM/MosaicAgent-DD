"""Implementations of the pipeline stages. Each stage is a pure function: it reads the
paths out of ctx, writes its products to disk, and returns a manifest."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from agents import Result          # the shared contract: every stage returns one

Image.MAX_IMAGE_PIXELS = None

# First four entries of the VOC palette. index -> (name, RGB)
CLASSES = {0: ("bare", (0, 0, 0)), 1: ("2L", (128, 0, 0)),
           2: ("1L", (0, 128, 0)), 3: ("TL", (128, 128, 0))}
PALETTE = np.array([c[1] for c in CLASSES.values()], np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def votes_setting(ctx) -> tuple:
    """Return (adaptive, n) from the votes configuration.

    An integer selects a fixed vote count. The string "adaptive" selects additional
    votes for invalid, uncertain or split decisions, with a maximum of five."""
    v = ctx.get("votes", 3)
    if isinstance(v, str) and v.strip().lower() == "adaptive":
        return True, 3
    return False, max(1, int(v))


# ============================================================ stitch
def stage_stitch(ctx):
    """Call MosaicAgent to stitch the raw tiles into one large map. Skipped when ctx
    already points at a finished mosaic."""
    out = ctx["work"] / "01_mosaic" / f"{ctx['sample']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if ctx.get("mosaic_in"):
        src = Path(ctx["mosaic_in"])
        if out.resolve() != src.resolve() and not out.exists():
            os.link(src, out) if src.stat().st_dev == out.parent.stat().st_dev \
                else __import__("shutil").copy2(src, out)
        return Result(ok=out.exists(), confidence=1.0,
                      data={"skipped": "使用已有拼接图", "mosaic": str(out),
                            "source": str(src)},
                      evidence=f"直接使用已有拼接图 {src.name}，{out.stat().st_size/1e6:.1f} MB",
                      escalate=not out.exists(),
                      escalate_reason="" if out.exists() else "拼接图不存在")

    repo = Path(ctx["mosaic_agent"])
    cmd = [sys.executable, str(repo / "run_stitch.py"),
           "--data", str(ctx["raw_dir"]), "--work", str(ctx["work"] / "01_mosaic" / "_cache"),
           "--out", str(out), "--votes", str(votes_setting(ctx)[1])]
    if ctx.get("no_ai"):
        cmd.append("--no-ai")
    if ctx.get("full_res"):
        cmd.append("--full")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    (ctx["work"] / "01_mosaic" / "stitch.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"MosaicAgent 拼接失败（退出码 {p.returncode}），"
                           f"日志见 {ctx['work'] / '01_mosaic' / 'stitch.log'}")
    return Result(ok=out.exists(), confidence=1.0,
                  data={"mosaic": str(out), "cmd": " ".join(cmd),
                        "seconds": round(time.time() - t0, 1)},
                  evidence=f"MosaicAgent 退出码 0，产出 {out.name}",
                  escalate=not out.exists(),
                  escalate_reason="" if out.exists() else "拼接完成但产物缺失")


# ============================================================ segment
def _valid_region(photo, filler="white", tol=5):
    """Exclude pixels matching the configured white or black filler colour.

    This is a colour-based support heuristic, not independently measured scan
    coverage. Real sample pixels matching the filler can also be excluded; inspect
    the support mask and retain its definition with the resulting statistics."""
    if filler == "white":
        bad = np.all(photo >= 255 - tol, axis=2)
    elif filler == "black":
        bad = np.all(photo <= tol, axis=2)
    else:
        bad = np.zeros(photo.shape[:2], bool)
    return ~bad


def _segment_geometry(ctx):
    """Recompute an optional scale plan; never trust caller-supplied derived fields."""
    supplied = ctx.get("inference_geometry")
    if supplied is None:
        return None
    if not isinstance(supplied, dict):
        raise ValueError("inference_geometry must be a complete geometry-plan object")
    from inference_geometry import plan_inference_geometry

    derived = plan_inference_geometry(
        supplied.get("observed_native_size"), supplied.get("model_reference_size"),
        same_physical_fov_confirmed=supplied.get("same_physical_fov_confirmed", False),
        model_tile=supplied.get("model_tile", 512),
        model_overlap=supplied.get("model_overlap", 64),
        provenance=supplied.get("provenance"),
    )
    if derived["status"] != "ready" or not derived["adapted_mode_allowed"]:
        raise ValueError("Inference geometry needs calibration: " + ", ".join(derived["reasons"]))
    if supplied != derived:
        raise ValueError("Inference geometry differs from its recomputed plan; prepare it again")
    for key, planned in (("tile", derived["model_tile"]), ("overlap", derived["model_overlap"])):
        if key in ctx and (type(ctx[key]) is not int or ctx[key] != planned):
            raise ValueError(f"Configured {key} conflicts with the inference geometry model window")
    return derived


def stage_segment(ctx, mosaic_path):
    from scipy import ndimage
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(Path(__file__).parent))
    from seg_model import load_segmenter

    geometry = _segment_geometry(ctx)  # Check before loading weights or creating outputs.
    outdir = ctx["work"] / "02_segment"
    outdir.mkdir(parents=True, exist_ok=True)
    mask_p = outdir / f"{ctx['sample']}_mask.npy"
    valid_p = outdir / f"{ctx['sample']}_valid.npy"

    photo = np.array(Image.open(mosaic_path).convert("RGB"))
    H, W, _ = photo.shape
    valid = _valid_region(photo, ctx.get("filler", "white"))
    trim = int(ctx.get("border_trim_px", 100))
    if trim > 0:
        # One distance transform gets this exactly right. The old code was
        # binary_erosion(3x3, iterations=trim//3), but a 3x3 structuring element erodes
        # only 1 px per iteration, so border_trim_px=100 actually trimmed just 33 px --
        # contradicting the number stated in the config, the README and the handbook
        # (fixed 2026-08-29).
        valid = ndimage.distance_transform_edt(valid) >= trim

    torch.set_num_threads(ctx.get("threads", os.cpu_count() or 4))
    device = ctx.get("device", "cpu")
    model, meta = load_segmenter(ctx["weights"], device=device)

    model_tile = geometry["model_tile"] if geometry else int(ctx.get("tile", 512))
    model_overlap = geometry["model_overlap"] if geometry else int(ctx.get("overlap", 64))
    tile = geometry["native_tile"] if geometry else model_tile
    ov = geometry["native_overlap"] if geometry else model_overlap
    resize = bool(geometry and geometry["model_resize_scale"] != 1.0)
    step = max(1, tile - ov)
    bs = int(ctx.get("batch", 4))
    full = np.zeros((H, W), np.uint8)
    seen = np.zeros((H, W), bool)
    ys = list(range(0, max(1, H - ov), step))
    xs = list(range(0, max(1, W - ov), step))
    boxes = [(t, min(t + tile, H), l, min(l + tile, W)) for t in ys for l in xs]
    t0 = time.time()
    n = 0

    def flush(batch):
        """Infer equal-size source tiles and preserve the first-written overlap policy.

        This merge rule is retained for compatibility; it does not establish which
        overlapping tile provides the more accurate boundary prediction.
        """
        if not batch:
            return
        if resize:
            # Pad partial source tiles at their native pixel scale. Resizing every
            # partial extent directly to model_tile would distort the edge scale.
            raw = np.zeros((len(batch), tile, tile, 3), dtype=np.uint8)
            for k, (t, b, l, r) in enumerate(batch):
                raw[k, :b - t, :r - l] = photo[t:b, l:r]
            x = torch.from_numpy(raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous().to(device)
            x = F.interpolate(x, size=(model_tile, model_tile), mode="bilinear", align_corners=False)
            mean = torch.as_tensor(MEAN, device=device).view(1, 3, 1, 1)
            std = torch.as_tensor(STD, device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
        else:
            # Preserve the existing preprocessing and padding convention exactly.
            arr = np.stack([photo[t:b, l:r] for t, b, l, r in batch]).astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        ph, pw = (32 - x.shape[2] % 32) % 32, (32 - x.shape[3] % 32) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        with torch.no_grad():
            logit = model(x)["out"]
        if resize:
            labels = torch.argmax(logit[:, :, :model_tile, :model_tile], 1)
            pred = F.interpolate(labels[:, None].to(torch.float32), size=(tile, tile),
                                 mode="nearest")[:, 0].to(torch.uint8).cpu().numpy()
        else:
            pred = torch.argmax(logit, 1).to(torch.uint8).cpu().numpy()
        for k, (t, b, l, r) in enumerate(batch):
            blk = full[t:b, l:r]
            sn = seen[t:b, l:r]
            p = pred[k][:b - t, :r - l]
            blk[~sn] = p[~sn]
            sn[:] = True

    batch = []
    for box in boxes:
        t, b, l, r = box
        if batch and (b - t, r - l) != (batch[0][1] - batch[0][0], batch[0][3] - batch[0][2]):
            flush(batch); n += len(batch); batch = []
        batch.append(box)
        if len(batch) >= bs:
            flush(batch); n += len(batch); batch = []
        if n and n % 200 == 0:
            print(f"      segment {n}/{len(boxes)} 块  {time.time()-t0:.0f}s", flush=True)
    flush(batch); n += len(batch)
    uncovered = int((~seen).sum())
    if uncovered:
        raise RuntimeError(f"Segmentation left {uncovered} native image pixels uncovered")
    # Force the non-scanned region to zero so it cannot land in any class
    full[~valid] = 0
    np.save(mask_p, full)
    np.save(valid_p, valid)
    c = np.bincount(full[valid].ravel(), minlength=4)
    nonscan = float((~valid).mean() * 100)

    # ---- self-check ----
    # (1) The non-scanned fraction has to be plausible. 0% means the filler colour was
    #     guessed wrong (the mosaic is not on a white ground); >25% means the stitch
    #     itself went wrong. Neither should pass silently.
    # (2) Any of the four classes coming back empty is suspicious.
    # (3) Comparing the class distribution in the tile-seam band against the tile
    #     interiors quantifies how much tiling artefact is left.
    checks, esc, reasons = [], False, []
    if ctx.get("filler") in ("white", "black"):
        if nonscan < 0.5:
            esc = True; reasons.append(f"非扫描区只占 {nonscan:.2f}%，填充色可能判错")
        elif nonscan > 25.0:
            esc = True; reasons.append(f"非扫描区占 {nonscan:.2f}%，拼接可能有问题")
    checks.append(f"非扫描区 {nonscan:.2f}%")
    zero = [CLASSES[i][0] for i in range(4) if c[i] == 0]
    if zero:
        esc = True; reasons.append(f"这些类别一个像素都没有: {', '.join(zero)}")
    fg = int(c[1] + c[2] + c[3])
    if fg > 0 and ov > 0:
        yy = np.arange(H) % max(tile - ov, 1)
        xx = np.arange(W) % max(tile - ov, 1)
        band = (np.minimum(yy, 3)[:, None] < 3) | (np.minimum(xx, 3)[None, :] < 3)
        band &= valid
        inner = valid & ~band
        if band.sum() > 1000 and inner.sum() > 1000:
            cb = np.bincount(full[band].ravel(), minlength=4)
            ci = np.bincount(full[inner].ravel(), minlength=4)
            fb, fi = cb[1:].sum(), ci[1:].sum()
            if fb and fi:
                d2 = abs(cb[1] / fb - ci[1] / fi) * 100
                checks.append(f"切块边界带 vs 内部的双层占比差 {d2:.2f} pp")
                if d2 > 5.0:
                    esc = True; reasons.append(f"切块边界伪影偏大（双层差 {d2:.1f} pp）")
    conf = 0.5 if esc else 0.95
    return Result(ok=not esc, confidence=conf,
                  data={"mask": str(mask_p), "valid_mask": str(valid_p),
                        "image_size": [int(W), int(H)], "tiles": n, "tile": tile,
                        "overlap": ov, "border_trim_px": trim,
                        "requested_model_tile": model_tile, "requested_model_overlap": model_overlap,
                        "source_crop_tile": tile, "source_crop_overlap": ov,
                        "inference_geometry": geometry,
                        "resample_mode": {"input": "bilinear" if resize else "none",
                                          "labels": "nearest" if resize else "none",
                                          "partial_tile_padding": "raw_rgb_zero_before_resize" if resize else "normalized_zero_to_multiple_of_32"},
                        "native_image_pixels": int(H * W), "valid_native_pixels": int(valid.sum()),
                        "uncovered_native_pixels": uncovered, "counts_coordinate_system": "native_image_pixels",
                        "excluded_non_scan_pct": round(nonscan, 2),
                        "model": meta, "seconds": round(time.time() - t0, 1),
                        "raw_counts": {CLASSES[i][0]: int(c[i]) for i in range(4)}},
                  evidence="; ".join(checks),
                  escalate=esc, escalate_reason="; ".join(reasons))


# ============================================================ audit
# ROADMAP 中-4. The segment stage's own self-check only looks at aggregate numbers
# (non-scan fraction, class counts, seam band). Nothing there catches the 2026-08-29
# failure *as a class*: a whole featureless region -- pure white filler, an overexposed
# patch, a bleached area -- confidently labelled crystal. The model was 99.9% sure on
# pure white, so neither entropy nor confidence flags it. What does flag it, for free,
# is the photo itself: a region with no photometric structure at all cannot contain
# crystal edges, so "local variance ~ 0 AND mask says crystal" is a contradiction the
# code can state without a model.
#
# Gated design, paid for in stages:
#   (a) deterministic low-variance detector, always on, zero cost;
#   (b) a VLM pilot on a handful of deterministically sampled windows, only when
#       audit_vlm_windows > 0 and a Kimi key exists; the role only DESCRIBES whether
#       the mask matches the photo, it never edits the mask;
#   (c) the stage escalates; it changes nothing downstream.
# The thresholds are meant to be tuned so a clean run raises nothing -- alarm fatigue
# is what kills a tool like this in a group. And a pilot without planted controls
# bounds the auditor's *capability*; it does not prove the segmentation is right.
#
# ROADMAP 中-4 also sketches a stricter variant (model sees the photo only, agreement
# computed afterwards by code, planted controls with a measured miss rate). That is
# the follow-up once the pilot has shown the model adds anything over the detector.

AUDIT_DEFAULTS = {
    "audit_scale_div": 4,        # the detector runs on photo[::d, ::d]
    "audit_window": 32,          # local-variance window, px at the downsampled scale
    "audit_var_thr": 3.0,        # grayscale variance (0-255 scale) below which a window
                                 # is "featureless"; std < ~1.7 gray levels. Pure filler
                                 # is exactly 0; a colour gradient across a melt pool is
                                 # tens, so gradients do not alarm.
    "audit_dark_gray": 60,       # grayscale below which a pixel is "near black" and
                                 # cannot carry crystal contrast either. Variance misses
                                 # the noisy near-black band beyond the wafer edge that the
                                 # U-Net labels 1L (audit_pilot: 1.62 Mpx on S05mg, the
                                 # detector saw 14% of it; gray < 40-60 catches 96-100%
                                 # with zero alarms on the two clean runs). A substrate
                                 # whose interference colour is legitimately dark would
                                 # alarm -- that is why it is a named field, in config_hash,
                                 # and every alarm reports its mean_gray.
    "audit_min_px": 20000,       # smallest alarm, full-resolution px (~140 x 140 px)
    "audit_vlm_windows": 0,      # VLM pilot windows; 0 = detector only
    "audit_vlm_window_px": 512,  # pilot window side, full-resolution px
    "audit_vlm_crop_side": 768,  # crop side sent to the model (up-sampled, see ROADMAP)
    "audit_flag_frac": 0.2,      # escalate when more than this fraction of the pilot
                                 # windows comes back flagged
}

AUDIT_VERDICTS = ("consistent", "systematic_error", "boundary_error", "unsure")
AUDIT_FLAGGED = ("systematic_error", "boundary_error")


def audit_params(ctx):
    """The audit's parameters, all from ctx with documented defaults. `scale_div` is
    honoured as a fallback for audit_scale_div so a config that already carries one
    number for all downsampled work keeps working."""
    P = {k: ctx.get(k, v) for k, v in AUDIT_DEFAULTS.items()}
    if "audit_scale_div" not in ctx and "scale_div" in ctx:
        P["audit_scale_div"] = ctx["scale_div"]
    for k in ("audit_scale_div", "audit_window", "audit_min_px", "audit_vlm_windows",
              "audit_vlm_window_px", "audit_vlm_crop_side"):
        P[k] = int(P[k] or 0)
    for k in ("audit_var_thr", "audit_flag_frac", "audit_dark_gray"):
        P[k] = float(P[k])
    return P


def _gray(photo):
    p = photo.astype(np.float32)
    return 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]


def local_variance(gray, window):
    """Sliding-window variance E[x^2] - E[x]^2 in float64 (float32 loses the last
    digits at x ~ 255 and can come out slightly negative; it is clipped anyway)."""
    from scipy import ndimage
    g = np.asarray(gray, np.float64)
    m = ndimage.uniform_filter(g, size=int(window), mode="reflect")
    m2 = ndimage.uniform_filter(g * g, size=int(window), mode="reflect")
    return np.clip(m2 - m * m, 0.0, None)


def detect_low_variance(photo, mask, valid, scale_div=4, window=32, var_thr=3.0,
                        min_px=20000, dark_gray=60.0):
    """(a) The free detector. Connected regions that are featureless in the photo
    (local grayscale variance < var_thr) OR near black (gray < dark_gray) yet labelled
    crystal (1/2/3) in the mask, at least min_px full-resolution pixels large, become
    alarms: hit = ((var < var_thr) | (gray < dark_gray)) & crystal & valid.

    Returns (alarms, summary, lowvar_small, d). alarms are sorted by area, descending;
    bbox is (x0, x1, y0, y1) at full resolution like everywhere else in the pipeline.
    lowvar_small is the variance criterion alone (the pilot prompt quotes it as such).
    """
    from scipy import ndimage
    d = max(1, int(scale_div))
    ph, m = photo[::d, ::d], mask[::d, ::d]
    vm = np.ones(m.shape, bool) if valid is None else valid[::d, ::d]
    gray = _gray(ph)
    var = local_variance(gray, window)
    lowvar = var < float(var_thr)
    dark = gray < float(dark_gray)
    crystal = (m > 0) & vm
    hit = (lowvar | dark) & crystal
    n_crystal = int(crystal.sum())
    by_class, dark_by_class = {}, {}
    for i in (1, 2, 3):
        sel = (m == i) & vm
        by_class[CLASSES[i][0]] = round(float((sel & lowvar).sum() / max(sel.sum(), 1)), 4)
        dark_by_class[CLASSES[i][0]] = round(float((sel & dark).sum() / max(sel.sum(), 1)), 4)
    summary = {"crystal_px": n_crystal * d * d,
               "low_var_crystal_frac": round(float((lowvar & crystal).sum() / max(n_crystal, 1)), 4),
               "dark_crystal_frac": round(float((dark & crystal).sum() / max(n_crystal, 1)), 4),
               "flagged_crystal_frac": round(float(hit.sum() / max(n_crystal, 1)), 4),
               "low_var_frac_by_class": by_class,
               "dark_frac_by_class": dark_by_class}
    alarms = []
    if hit.any():
        lab, n = ndimage.label(hit)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        sizes[0] = 0
        min_small = float(min_px) / (d * d)
        objs = ndimage.find_objects(lab)
        for i in np.argsort(-sizes):
            i = int(i)
            if sizes[i] == 0 or sizes[i] < min_small:
                break
            sl = objs[i - 1]
            comp = lab[sl] == i
            cls = np.bincount(m[sl][comp].ravel(), minlength=4)
            dom = int(np.argmax(cls[1:]) + 1)
            tot_dom = int(((m == dom) & vm).sum())
            alarms.append({
                "rank": len(alarms) + 1,
                "bbox": (sl[1].start * d, sl[1].stop * d, sl[0].start * d, sl[0].stop * d),
                "area_px": int(sizes[i]) * d * d,
                "pct_of_class": round(float(cls[dom] / max(tot_dom, 1) * 100), 2),
                "pct_of_crystal": round(float(sizes[i] / max(n_crystal, 1) * 100), 2),
                "dominant_class": CLASSES[dom][0],
                "class_composition": {CLASSES[k][0]: round(float(cls[k] / sizes[i] * 100), 2)
                                      for k in (1, 2, 3)},
                "mean_gray": round(float(gray[sl][comp].mean()), 1),
                "mean_var": round(float(var[sl][comp].mean()), 3),
                # which criterion fired, as fractions of the component's pixels
                "low_var_frac": round(float(lowvar[sl][comp].mean()), 3),
                "dark_frac": round(float(dark[sl][comp].mean()), 3),
            })
    summary["n_alarms"] = len(alarms)
    return alarms, summary, lowvar, d


def sample_audit_windows(mask, valid, n, win_px, seed=0, min_valid_frac=0.9):
    """(b) The deterministic sampler -- ROADMAP 中-4 says the sampler is the actual
    contribution, not the model. Non-overlapping grid cells of win_px, at least
    min_valid_frac inside the scanned area, stratified over an S x S spatial grid
    (S = ceil(sqrt(n))) so the windows do not all land in one corner, and drawn within
    each stratum with weight 0.05 + crystal_fraction so crystal-rich windows are
    favoured without bare windows becoming impossible. Fixed seed -> same windows.

    Returns [{bbox, valid_frac, crystal_frac, stratum}] with bbox = (x0, x1, y0, y1).
    """
    H, W = mask.shape
    n = int(n)
    win = int(win_px)
    ny, nx = H // win, W // win
    if n <= 0 or win <= 0 or ny == 0 or nx == 0:
        return []
    S = int(np.ceil(np.sqrt(n)))

    def stratum(iy, ix):
        return (iy * S // ny) * S + (ix * S // nx)

    strata = {}
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * win, ix * win
            v = valid[y0:y0 + win, x0:x0 + win]
            vf = float(v.mean())
            if vf < min_valid_frac:
                continue
            mm = mask[y0:y0 + win, x0:x0 + win]
            cf = float(((mm > 0) & v).sum() / max(int(v.sum()), 1))
            strata.setdefault(stratum(iy, ix), []).append((iy, ix, vf, cf))
    if not strata:
        return []
    rng = np.random.default_rng(int(seed))
    order = sorted(strata)
    chosen = []
    while len(chosen) < n and any(strata[s] for s in order):
        for s in order:
            if len(chosen) >= n:
                break
            cells = strata[s]
            if not cells:
                continue
            w = np.array([0.05 + c[3] for c in cells], np.float64)
            k = int(rng.choice(len(cells), p=w / w.sum()))
            chosen.append(cells.pop(k))
    return [{"bbox": (ix * win, (ix + 1) * win, iy * win, (iy + 1) * win),
             "valid_frac": round(vf, 3), "crystal_frac": round(cf, 3),
             "stratum": int(stratum(iy, ix))}
            for iy, ix, vf, cf in chosen]


def build_audit_role(KA):
    """The segmentation auditor. It only describes whether the mask matches the photo
    -- it never decides what to count, and nothing it says edits the mask."""
    return KA.Role(
        name="segmentation_auditor",
        system=(
            "你是二维材料 CVD 生长显微图像的分割审计员。你会看到大面积拼接图上"
            "**一个窗口**：第一张是光学显微原图裁剪，第二张是同一位置的分割结果"
            "（黑=裸衬底，绿=单层 1L，深红=双层 2L，橄榄色=厚层 TL）。\n\n"
            "你要回答的唯一问题是：**分割结果和原图对得上吗？**你只做描述，"
            "不决定任何统计怎么算；你的结论只会送人复核，不会改动分割图。\n\n"
            "结论（四选一）：\n"
            "  - consistent：分割里每一类区域在原图上都有对应的光学对比区域，"
            "边界大体沿着可见的对比边走。\n"
            "  - systematic_error：**整块**在原图上没有任何特征的区域"
            "（纯白、纯黑、均匀一色，看不到晶体轮廓）被标成了某个晶体类别；"
            "或者某一类整体外溢到明显不属于它的地方（例如大片裸衬底被标成单层）。"
            "这是最严重的一类，2026-08-29 那次'纯白非扫描区被判成厚层'就是它。\n"
            "  - boundary_error：类别大体对，但边界明显偏离可见的对比边"
            "（外扩或内缩超过晶体自身尺寸的两成左右）。\n"
            "  - unsure：图太小、太模糊或对比度不够，判不了。\n\n"
            "三条容易搞错的判据：\n"
            "  1. **先看原图，再看分割**。问自己：只看原图，我会在这里画出这块吗？"
            "不要被分割的颜色锚定，然后去原图上找理由。\n"
            "  2. **不报锯齿**。这个尺度的分割边界本来就有锯齿，只有面积尺度的"
            "分歧才算 boundary_error，小的毛边一律算 consistent。\n"
            "  3. **大面积均匀 + 晶体类别 = 最要警惕**。一块没有任何纹理的区域"
            "不可能有晶体的边，被标成晶体就是 systematic_error。\n\n"
            "会同时给你这个窗口的数值特征（各类占比、低方差像素占比、"
            "是否与确定性检测的报警重叠），作为辅助，但**以图像为准**。"
        ),
        schema_hint=(
            '{"verdict": "consistent|systematic_error|boundary_error|unsure", '
            '"error_class": "bare|1L|2L|TL|none", '
            '"confidence": 0.0-1.0, '
            '"reason": "一句话，中文，40 字以内，只说决定性证据"}'),
        # Same reasoning-model budget as region_adjudicator: the chain of thought
        # counts against completion tokens, and 8192 avoids the truncate-and-retry.
        max_tokens=8192,
    )


def _audit_crops(photo, mask, bbox, side=768):
    """Photo crop + palette-coloured mask crop for one window (BGR, MosaicAgent
    convention). Up-sampled to `side` because at mosaic scale a 20 um crystal is
    ~27 px, too small for the model to judge a boundary (ROADMAP 中-4)."""
    import cv2
    x0, x1, y0, y1 = bbox
    ph = np.ascontiguousarray(photo[y0:y1, x0:x1])
    mk = np.ascontiguousarray(PALETTE[mask[y0:y1, x0:x1]])
    s = float(side) / max(ph.shape[0], ph.shape[1], 1)
    if abs(s - 1.0) > 1e-6:
        wh = (max(1, int(ph.shape[1] * s)), max(1, int(ph.shape[0] * s)))
        ph = cv2.resize(ph, wh, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        mk = cv2.resize(mk, wh, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(ph, cv2.COLOR_RGB2BGR), cv2.cvtColor(mk, cv2.COLOR_RGB2BGR)


def _bbox_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _audit_pilot(ctx, photo, mask, valid, alarms, lowvar, d, pool, P, votes):
    """(b) Run the VLM pilot on deterministically sampled windows. Needs a pool whose
    client is available; the caller guarantees that."""
    import region_agents as RA
    n = P["audit_vlm_windows"]
    wins = sample_audit_windows(mask, valid, n, P["audit_vlm_window_px"],
                                seed=int(ctx.get("seed", 0)))
    if not wins:
        return {"ran": False, "windows": n, "sampled": 0,
                "reason": "没有满足条件的窗口（有效区太小或窗口太大）"}
    for w in wins:
        x0, x1, y0, y1 = w["bbox"]
        mm = mask[y0:y1, x0:x1]
        vv = valid[y0:y1, x0:x1]
        c = np.bincount(mm[vv].ravel(), minlength=4)
        tot = max(int(c.sum()), 1)
        w["class_pct"] = {CLASSES[k][0]: round(float(c[k] / tot * 100), 1) for k in range(4)}
        lv = lowvar[y0 // d:max(y0 // d + 1, y1 // d), x0 // d:max(x0 // d + 1, x1 // d)]
        w["low_var_frac"] = round(float(lv.mean()), 3) if lv.size else 0.0
        w["overlaps_alarm"] = [a["rank"] for a in alarms if _bbox_overlap(a["bbox"], w["bbox"])]

    KA, _ = RA._import_kimi()
    role = build_audit_role(KA)

    def build(w):
        ph, mk = _audit_crops(photo, mask, w["bbox"], side=P["audit_vlm_crop_side"])
        cp = w["class_pct"]
        prompt = (
            f"这个窗口的数值特征：\n"
            f"  位置：x {w['bbox'][0]}-{w['bbox'][1]}，y {w['bbox'][2]}-{w['bbox'][3]} px\n"
            f"  分割各类占比：裸衬底 {cp['bare']}%，1L {cp['1L']}%，"
            f"2L {cp['2L']}%，TL {cp['TL']}%\n"
            f"  原图中局部方差极低（无纹理）的像素占比：{w['low_var_frac'] * 100:.1f}%\n"
            f"  与确定性检测报警重叠：{'是，报警 #' + ','.join(map(str, w['overlaps_alarm'])) if w['overlaps_alarm'] else '否'}\n\n"
            f"第一张图是光学显微原图裁剪，第二张是同位置的分割结果。"
            f"描述分割结果是否与原图一致。")
        return prompt, [ph, mk]

    res = pool.map(role, wins, build, key="verdict", use_votes=votes > 1)
    results, counts = [], {v: 0 for v in AUDIT_VERDICTS}
    n_split = n_flag = n_unan_sys = n_lost = 0
    confs = []
    for w, r in zip(wins, res):
        entry = {k: w[k] for k in ("bbox", "valid_frac", "crystal_frac", "stratum",
                                   "class_pct", "low_var_frac", "overlaps_alarm")}
        if r is None:
            entry.update(verdict=None, answered=False)
            results.append(entry)
            continue
        verdict = str(r.get("verdict", "unsure"))
        if verdict not in AUDIT_VERDICTS:
            verdict = "unsure"
        tally = r.get("_tally") or {}
        n_votes = int(r.get("_n_votes", 1) or 1)
        # split = more than one verdict among the answers that came back; a vote lost to
        # a timeout or 429 is reported as lost_votes, not as dissent
        split = len(tally) > 1
        unanimous = len(tally) == 1
        entry.update(verdict=verdict, answered=True,
                     error_class=r.get("error_class"),
                     confidence=float(r.get("confidence", 0) or 0),
                     reason=str(r.get("reason", ""))[:120],
                     _tally=tally, _n_votes=n_votes, lost_votes=max(0, votes - n_votes),
                     split=split, unanimous=unanimous)
        counts[verdict] += 1
        n_split += int(split)
        n_flag += int(verdict in AUDIT_FLAGGED)
        n_unan_sys += int(unanimous and verdict == "systematic_error")
        n_lost += entry["lost_votes"]
        confs.append(entry["confidence"])
        results.append(entry)
    answered = len(confs)
    return {"ran": True, "windows": n, "sampled": len(wins), "answered": answered,
            "votes": votes, "flagged": n_flag,
            "flagged_frac": round(n_flag / answered, 3) if answered else None,
            "split": n_split, "unanimous_systematic": n_unan_sys, "lost_votes": n_lost,
            "verdict_counts": counts,
            "mean_confidence": round(sum(confs) / answered, 3) if answered else None,
            "results": results,
            "note": ("试点只是抽样描述，没有植入对照，不能据此估计漏检率；"
                     "它限定的是审计员的能力，不证明分割是对的。")}


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def stage_audit(ctx, mosaic_path, mask_path, valid_path, pool=None):
    """Audit the segmentation for whole-region failures. Escalates only; the mask is
    never modified. `pool` is injectable for tests; normally it is built here from
    the Kimi client when audit_vlm_windows > 0 and a key exists."""
    outdir = ctx["work"] / "02_segment"
    outdir.mkdir(parents=True, exist_ok=True)
    out_p = outdir / f"{ctx['sample']}_audit.json"
    P = audit_params(ctx)
    _, votes = votes_setting(ctx)
    t0 = time.time()

    photo = np.array(Image.open(mosaic_path).convert("RGB"))
    mask = np.load(mask_path)
    valid = np.load(valid_path)
    alarms, summ, lowvar, d = detect_low_variance(
        photo, mask, valid, scale_div=P["audit_scale_div"], window=P["audit_window"],
        var_thr=P["audit_var_thr"], min_px=P["audit_min_px"],
        dark_gray=P["audit_dark_gray"])

    # ---- (b) the gated pilot ----
    pilot = {"ran": False, "windows": P["audit_vlm_windows"]}
    if P["audit_vlm_windows"] <= 0:
        pilot["reason"] = "audit_vlm_windows=0，只跑确定性检测"
    elif ctx.get("no_ai"):
        pilot["reason"] = "--no-ai，试点跳过"
    else:
        if pool is None:
            try:
                import region_agents as RA
                KA, _ = RA._import_kimi()
                client = KA.KimiClient(cache_dir=str(outdir / "_kimi_cache"))
                if client.available:
                    pool = KA.AgentPool(client, workers=int(ctx.get("workers", 6)),
                                        votes=votes)
            except Exception as e:
                pilot["reason"] = f"Kimi 不可用（{e}），试点跳过"
        if pool is not None and getattr(pool.client, "available", False):
            pilot = _audit_pilot(ctx, photo, mask, valid, alarms, lowvar, d, pool, P, votes)
        elif "reason" not in pilot:
            pilot["reason"] = "没有 KIMI_API_KEY，试点跳过"

    # ---- self-check / escalation ----
    # The pilot escalates on evidence, not on disagreement: the flagged fraction over
    # the threshold, or any window where every vote said systematic_error. Split votes
    # are reported as information only -- at temperature 1 with an 11% individual flag
    # rate, 8/45 pilot windows split in the audit_pilot experiment and none of the
    # splits was a real error; escalating on them is an alarm-fatigue generator.
    esc, reasons = False, []
    if alarms:
        esc = True
        a = alarms[0]
        reasons.append(f"确定性检测报警 {len(alarms)} 处：无纹理或近黑区域被判成晶体"
                       f"（最大一处 {a['area_px']:,} px，{a['dominant_class']} "
                       f"占该类 {a['pct_of_class']}%，平均灰度 {a['mean_gray']}，"
                       f"平均方差 {a['mean_var']}）")
    if pilot.get("ran"):
        if not pilot["answered"]:
            esc = True
            reasons.append("VLM 试点无有效返回")
        else:
            if pilot["flagged_frac"] > P["audit_flag_frac"]:
                esc = True
                reasons.append(f"试点 {pilot['flagged']}/{pilot['answered']} 个窗口判分割有误"
                               f"（{pilot['flagged_frac']:.0%} > {P['audit_flag_frac']:.0%}）")
            if pilot["unanimous_systematic"]:
                esc = True
                wins_u = [f"{r['bbox']}" for r in pilot["results"]
                          if r.get("unanimous") and r.get("verdict") == "systematic_error"]
                reasons.append(f"试点 {pilot['unanimous_systematic']} 个窗口全票判 "
                               f"systematic_error（{'; '.join(wins_u[:3])}）")
    if pilot.get("ran") and pilot.get("answered"):
        conf = float(pilot["mean_confidence"])
    else:
        conf = 0.5 if alarms else 0.9

    evidence = (f"低方差/暗度检测：{len(alarms)} 处报警，"
                f"{summ['low_var_crystal_frac'] * 100:.2f}% 的晶体像素落在无纹理区、"
                f"{summ['dark_crystal_frac'] * 100:.2f}% 落在近黑区"
                f"（var<{P['audit_var_thr']} 或 灰度<{P['audit_dark_gray']:g}，"
                f"窗口 {P['audit_window']} px @1/{d}，最小 {P['audit_min_px']} px）")
    if pilot.get("ran"):
        vc = pilot["verdict_counts"]
        evidence += (f"；VLM 试点 {pilot['answered']}/{pilot['sampled']} 窗口："
                     f"一致 {vc['consistent']}，系统性错误 {vc['systematic_error']}"
                     f"（全票 {pilot['unanimous_systematic']}），"
                     f"边界错误 {vc['boundary_error']}，unsure {vc['unsure']}，"
                     f"分歧 {pilot['split']}（仅作信息）")
    else:
        evidence += f"；VLM 试点未运行（{pilot.get('reason', '')}）"

    doc = {"sample": ctx["sample"], "parameters": {**P, "votes": votes,
                                                    "seed": int(ctx.get("seed", 0)),
                                                    "no_ai": bool(ctx.get("no_ai"))},
           "detector": {**summ, "alarms": alarms}, "pilot": pilot,
           "escalate": esc, "escalate_reason": "; ".join(reasons),
           "seconds": round(time.time() - t0, 1)}
    out_p.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return Result(ok=not esc, confidence=conf,
                  data={"audit": str(out_p), "n_alarms": len(alarms), "alarms": alarms,
                        "low_var_crystal_frac": summ["low_var_crystal_frac"],
                        "dark_crystal_frac": summ["dark_crystal_frac"],
                        "low_var_frac_by_class": summ["low_var_frac_by_class"],
                        "dark_frac_by_class": summ["dark_frac_by_class"],
                        "pilot": pilot, "parameters": doc["parameters"],
                        "seconds": doc["seconds"]},
                  evidence=evidence, escalate=esc, escalate_reason="; ".join(reasons))


# ============================================================ regions
def _exclude_split(v) -> bool:
    """Was the keep/exclude majority of this verdict not unanimous? Reads the pool's
    `_split`, falls back to `_decision_tally`; a label-only tally (old records, replayed
    verdicts) is never a split -- label wobble inside the exclude classes is not dissent."""
    if v.get("_split") is not None:
        return bool(v["_split"])
    dt = v.get("_decision_tally")
    if dt:
        return min(dt.values()) > 0
    return False


def stage_regions(ctx, mosaic_path, mask_path, valid_path):
    sys.path.insert(0, str(Path(__file__).parent))
    import region_agents as RA

    outdir = ctx["work"] / "03_regions"
    outdir.mkdir(parents=True, exist_ok=True)
    photo = np.array(Image.open(mosaic_path).convert("RGB"))
    mask = np.load(mask_path)
    valid = np.load(valid_path)

    proposed = RA.propose_regions(
        mask, photo, target_class=3, valid_mask=valid,
        min_pct_of_class=float(ctx.get("min_region_pct", 1.0)),
        max_candidates=int(ctx.get("max_regions", 8)),
        scale_div=int(ctx.get("region_scale_div", 4)))
    if not proposed:
        (outdir / f"{ctx['sample']}_exclusions.json").write_text(
            json.dumps({"sample": ctx["sample"], "verdicts": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        return Result(ok=True, confidence=1.0,
                      data={"candidates": 0, "excluded": 0, "verdicts": []},
                      evidence="没有够大的厚层连通域，无需裁决")
    cands, lab_pack = proposed
    import review as RV

    pool = None
    client = None
    replayed = False
    adaptive, votes_n = votes_setting(ctx)
    votes_cfg = "adaptive" if adaptive else votes_n
    rec_path = outdir / f"{ctx['sample']}_exclusions.json"
    if ctx.get("replay") and rec_path.exists():
        # --replay: rebuild the verdicts from the recorded exclusions.json, no model
        # call. This is what makes "these files are enough to reproduce" true.
        recorded = json.loads(rec_path.read_text(encoding="utf-8")).get("verdicts", [])
        verdicts = RV.replay_verdicts(ctx["sample"], cands, recorded)
        replayed = True
    else:
        if not ctx.get("no_ai"):
            try:
                KA, repo = RA._import_kimi()
                client = KA.KimiClient(cache_dir=str(outdir / "_kimi_cache"))
                if client.available:
                    pool = KA.AgentPool(client, workers=int(ctx.get("workers", 6)),
                                        votes=votes_cfg)
            except Exception as e:
                print(f"  [warn] Kimi 不可用（{e}），本阶段全部按保留处理")
        verdicts = RA.adjudicate_regions(cands, photo, mask, pool=pool, votes=votes_cfg,
                                         px_per_mm=ctx.get("px_per_mm"))
    # A keep/exclude majority that is still not unanimous after the maximum number of
    # votes (the fixed N, or the adaptive cap) is not acted on: the region stays in the
    # statistics and a human decides. This makes the prompt's own rule -- "宁可保留也
    # 不要误删真实数据" -- deterministic instead of depending on which votes came back
    # first (adaptive_votes: S15mg_r05 flips between keep 2-1 and exclude 2-1 across
    # fresh triples). The model's majority is kept next to it as majority_exclude.
    for v in verdicts:
        if v.get("decided_by") or not _exclude_split(v):
            continue
        v["majority_exclude"] = bool(v.get("exclude"))
        v["exclude"] = False
        v["needs_human"] = True
    # A recorded human verdict overrides the agent's; both stay in the record.
    human_log = []
    if ctx.get("human_verdicts"):
        verdicts, human_log = RV.apply_overrides(ctx["sample"], verdicts,
                                                 ctx["human_verdicts"], votes_n)
    excl = RA.build_exclusion_mask(mask.shape, verdicts, lab_pack)

    # The connected component only covers pixels the segmenter labelled
    # thick-layer, and the segmenter is not reliable *inside* a precursor pool:
    # on the 5 mg map it labelled the whole interior of a solid puddle as bare
    # substrate, leaving only an annulus as thick-layer. Filling holes cannot
    # recover that -- the annulus is not even closed -- because the boundary
    # should never have come from the segmentation in the first place. So for
    # every sufficiently large excluded region we re-derive the boundary from
    # the PHOTO by deterministic region growing, with the vision model only
    # choosing how far to grow (too_tight / about_right / too_loose).
    refine_pct = float(ctx.get("refine_min_pct", 5.0))
    refine_log = []
    big = [v for v in verdicts
           if v.get("exclude") and v.get("pct_of_class", 0) >= refine_pct]
    if big and not ctx.get("no_refine"):
        import refine_region as RR
        from scipy import ndimage as _nd
        for v in big:
            x0, x1, y0, y1 = v["bbox"]
            # 0.35 was not enough on the 5 mg pool: the grown region hit the
            # window edge and got clipped to a straight line. 0.6 leaves room.
            pad = int(0.6 * max(x1 - x0, y1 - y0))
            X0, X1 = max(0, x0 - pad), min(mask.shape[1], x1 + pad)
            Y0, Y1 = max(0, y0 - pad), min(mask.shape[0], y1 + pad)
            d = 2
            sp = photo[Y0:Y1, X0:X1][::d, ::d]
            sv = valid[Y0:Y1, X0:X1][::d, ::d]

            # --- COARSE ---------------------------------------------------
            # Take the union of every excluded component inside this window,
            # not just the one region. A precursor pool is usually segmented as
            # a rim plus several separate blobs inside it (on the 5 mg map the
            # rim was region #1 and five more regions sat inside its bounding
            # box), so growing each one independently can never merge them back
            # into the single physical object they are. Union them first, then
            # close at a radius tied to the object's own size and fill: that
            # bridges the gaps and yields one solid blob covering the pool.
            seed = excl[Y0:Y1, X0:X1][::d, ::d].copy()
            if seed.sum() < 500:
                continue
            frac = float(ctx.get("coarse_close_frac", 0.05))
            r = max(1, int(np.hypot(*seed.shape) * frac))
            coarse = _nd.binary_fill_holes(
                _nd.binary_closing(seed, np.ones((3, 3), bool), iterations=r)) & sv
            coarse_gain = float(coarse.sum() / max(seed.sum(), 1))

            # --- FINE -----------------------------------------------------
            # Now let the photo decide where the real edge is, with the model
            # only judging too_tight / about_right / too_loose.
            grown, rounds = RR.refine(sp, coarse, pool=pool, votes=votes_n,
                                      max_rounds=int(ctx.get("refine_rounds", 5)),
                                      log=lambda m: print(m, flush=True))
            up = np.repeat(np.repeat(grown, d, axis=0), d, axis=1)
            up = up[:Y1 - Y0, :X1 - X0]
            excl[Y0:Y1, X0:X1] |= up
            refine_log.append({"rank": v["rank"], "pct_of_class": v["pct_of_class"],
                               "coarse_close_px": r * d,
                               "coarse_area_gain": round(coarse_gain, 3),
                               "rounds": rounds})
            print(f"        #{v['rank']} 粗步：并集+闭合{r*d}px+填充 -> 面积×{coarse_gain:.2f}",
                  flush=True)
    excl &= valid
    np.save(outdir / f"{ctx['sample']}_exclusion.npy", excl)
    # Cost of this stage in model calls (SYNTHESIS §3: calls per region is the unit of
    # cost -- completion tokens are hidden reasoning and cannot be reduced)
    kimi_calls = int(sum(int(v.get("_n_asked", 0) or 0) for v in verdicts))
    kimi_usage = ({k: int(getattr(client.usage, k)) for k in
                   ("calls", "cached", "failed", "truncated", "prompt_tokens",
                    "completion_tokens")} if client is not None else None)
    # Log it: this record is precisely what was missing from the published version
    (outdir / f"{ctx['sample']}_exclusions.json").write_text(json.dumps(
        {"sample": ctx["sample"], "rule": ctx.get("exclusion_rule", "kimi_region_adjudicator"),
         "votes": votes_cfg, "verdicts": verdicts,
         "kimi_calls": kimi_calls, "kimi_usage": kimi_usage,
         "boundary_refinement": refine_log,
         "human_overrides": human_log, "replayed": replayed},
        ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- self-check ----
    # A keep/exclude split (forced to keep above), a verdict of uncertain, or the model
    # never being reached at all -- each of these needs a human to take a look. Label
    # wobble inside the exclude classes (melt / film / contamination) is not raised: it
    # is invisible in the statistics. A region a human has already decided (decided_by)
    # is settled and is not raised again; that is what the review queue is for.
    open_v = [v for v in verdicts if not v.get("decided_by")]
    split = [v for v in open_v if _exclude_split(v)]
    unc = [v for v in open_v if v.get("category") == "uncertain"]
    no_ai = [v for v in open_v if not v.get("_ai") and not v.get("_replayed")]
    unmatched = [v for v in verdicts if v.get("_replay_unmatched")]
    esc, reasons = False, []
    if no_ai:
        esc = True; reasons.append(f"{len(no_ai)} 块没有经过模型判定（没有 API key？）")
    if unmatched:
        esc = True; reasons.append(f"回放时 {len(unmatched)} 块找不到历史判决，已保留")
    if unc:
        esc = True; reasons.append(f"{len(unc)} 块判成 uncertain，已保留但需人工复核")
    if split:
        esc = True
        reasons.append("去留分歧（已强制保留，待人工判定）: " + ", ".join(
            f"#{v['rank']}({v.get('category')} 去留 {v.get('_decision_tally')}，"
            f"{v.get('_n_votes')} 票)" for v in split))
    confs = [1.0 if v.get("decided_by") else float(v.get("confidence", 0) or 0)
             for v in verdicts if v.get("_ai") or v.get("decided_by")]
    conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    return Result(ok=not esc, confidence=conf,
                  data={"candidates": len(cands),
                        "excluded": int(sum(1 for v in verdicts if v.get("exclude"))),
                        "excluded_area_pct_of_valid":
                            round(float(excl.sum() / max(valid.sum(), 1) * 100), 2),
                        "exclusion_mask": str(outdir / f"{ctx['sample']}_exclusion.npy"),
                        "excluded_area_pct_after_refine":
                            round(float(excl.sum() / max(valid.sum(), 1) * 100), 2),
                        "boundary_refinement": refine_log,
                        "human_overrides": human_log, "replayed": replayed,
                        "votes": votes_cfg, "kimi_calls": kimi_calls,
                        "kimi_usage": kimi_usage,
                        "needs_human": int(sum(1 for v in verdicts if v.get("needs_human"))),
                        "verdicts": verdicts},
                  evidence=(f"{len(cands)} 块候选，剔除 "
                            f"{sum(1 for v in verdicts if v.get('exclude'))} 块；"
                            f"去留全票一致 {len(open_v) - len(split)}/{len(open_v)}"
                            + (f"；模型调用 {kimi_calls} 次" if kimi_calls else "")
                            + (f"；人工判决 {len(human_log)} 条" if human_log else "")
                            + ("；回放历史判决，未调用模型" if replayed else "")),
                  escalate=esc, escalate_reason="; ".join(reasons))


# ============================================================ stats
def stage_stats(ctx, mask_path, valid_path, exclusion_path=None):
    outdir = ctx["work"] / "04_stats"
    outdir.mkdir(parents=True, exist_ok=True)
    mask = np.load(mask_path)
    valid = np.load(valid_path)
    excl = np.load(exclusion_path) if exclusion_path and Path(exclusion_path).exists() \
        else np.zeros_like(valid)
    sel = valid & ~excl

    def ratios(s):
        c = np.bincount(mask[s].ravel(), minlength=4)
        fg = int(c[1] + c[2] + c[3])
        return c, fg, ({CLASSES[i][0]: round(float(c[i] / fg * 100), 2) for i in (1, 2, 3)}
                       if fg else {})

    c_before, fg_b, r_before = ratios(valid)
    c_after, fg_a, r_after = ratios(sel)
    # The excluded (precursor / artefact) regions are reported as their own
    # quantity. They are NOT folded into 1L / 2L / TL: a precursor pool is not a
    # layer, and whatever the segmenter called the pixels inside it is not
    # meaningful. Reporting the area separately also lets a reader see exactly
    # how much was taken out, and what it had been labelled before.
    c_excl = np.bincount(mask[valid & excl].ravel(), minlength=4)
    excluded_block = {
        "area_px": int(excl.sum()),
        "pct_of_scanned_area": round(float(excl.sum() / max(valid.sum(), 1) * 100), 2),
        "composition_inside_before_exclusion": {
            CLASSES[i][0]: int(c_excl[i]) for i in range(4)},
        "note": ("Pixels inside these regions are excluded from 1L/2L/TL entirely, "
                 "from both numerator and denominator. The composition below only "
                 "records what the segmenter had labelled them before removal."),
    }
    if ctx.get("px_per_mm"):
        excluded_block["area_mm2"] = round(
            float(excl.sum() / ctx["px_per_mm"] ** 2), 4)

    res = {
        "sample": ctx["sample"],
        "denominator": "1L + 2L + TL; bare substrate is not counted, matching the published convention",
        "counted_px": fg_a,
        "bare_substrate_pct_of_scanned": round(float(c_after[0] / max(c_after.sum(), 1) * 100), 2),
        "ratios_pct": r_after,
        "ratios_pct_without_region_exclusion": r_before,
        "excluded_regions": excluded_block,
        "non_scan_px": int((~valid).sum()),
    }
    if ctx.get("px_per_mm"):
        res["scanned_area_mm2"] = round(float(sel.sum() / ctx["px_per_mm"] ** 2), 3)
    # ---- self-check ----
    # A large before/after gap means some conclusion hinges on whether one single region
    # was dropped. That has to be reported, not buried.
    alarm = float(ctx.get("sensitivity_alarm_pct", 2.0))
    deltas = {k: round(r_after.get(k, 0) - r_before.get(k, 0), 2) for k in ("1L", "2L", "TL")}
    big = {k: v for k, v in deltas.items() if abs(v) > alarm}
    res["delta_from_region_exclusion_pp"] = deltas
    res["sensitivity_alarm_pct"] = alarm
    (outdir / f"{ctx['sample']}_stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render the layer map with the excluded regions blanked out, so a reader can
    # see at a glance which area was not counted. Showing a segmentation map next
    # to a pie chart computed on a *different* area is exactly the inconsistency
    # the 2026-08-29 review found in the published figure.
    try:
        from PIL import Image as _I
        d = max(1, int(ctx.get("map_preview_div", 8)))
        sm, se = mask[::d, ::d], excl[::d, ::d]
        sv = valid[::d, ::d]
        rgb = PALETTE[sm]
        rgb[~sv] = (255, 255, 255)          # outside the scan: white
        # excluded region: flat mid grey with a diagonal hatch so it reads as
        # "removed", not as another class
        yy, xx = np.mgrid[0:sm.shape[0], 0:sm.shape[1]]
        hatch = ((yy + xx) % 10) < 3
        rgb[se] = (150, 150, 150)
        rgb[se & hatch] = (90, 90, 90)
        _I.fromarray(rgb).save(outdir / f"{ctx['sample']}_layermap_counted.png")
        res["layer_map_preview"] = str(outdir / f"{ctx['sample']}_layermap_counted.png")
    except Exception as e:
        res["layer_map_preview_error"] = str(e)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lbl = ["1L", "2L", "TL"]
        val = [r_after.get(k, 0) for k in lbl]
        col = [np.array(CLASSES[i][1]) / 255 for i in (2, 1, 3)]
        plt.figure(figsize=(5, 5))
        plt.pie(val, labels=[f"{l}\n{v:.1f}%" for l, v in zip(lbl, val)],
                colors=col, startangle=90, counterclock=False)
        plt.title(f"{ctx['sample']}   area fraction of grown crystals\n"
                  f"(bare substrate and {excluded_block['pct_of_scanned_area']:.1f}% "
                  f"excluded precursor/artefact area not counted)", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / f"{ctx['sample']}_pie.png", dpi=300)
        plt.savefig(outdir / f"{ctx['sample']}_pie.svg")
        plt.close()
        res["pie"] = str(outdir / f"{ctx['sample']}_pie.png")
    except Exception as e:
        res["pie_error"] = str(e)
        (outdir / f"{ctx['sample']}_stats.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    esc = bool(big)
    return Result(ok=True, confidence=1.0 if not big else 0.6, data=res,
                  evidence=("剔除区域使占比变化 " +
                            ", ".join(f"{k} {v:+.2f} pp" for k, v in deltas.items())),
                  escalate=esc,
                  escalate_reason=("以下类别对区域剔除高度敏感（阈值 "
                                   f"{alarm} pp）: " +
                                   ", ".join(f"{k} {v:+.2f} pp" for k, v in big.items())
                                   if big else ""))
