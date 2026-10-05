#!/usr/bin/env python3
"""Pin down the irregular boundary of a precursor melt pool (NaCl/MoO2 residue) through
several rounds of visual feedback.

## Why this step is needed

The exclusion mask produced by `region_agents` is a **connected component of the
thick-layer class**. Its shape is irregular rather than a box, but it contains only the
pixels the segmenter labelled thick layer. The melt pool is a physical object, and its
real boundary does not coincide with that component:

  - inside the melt pool there are scattered patches labelled monolayer or bilayer --
    they float on the pool rather than having grown there, yet they currently stay in
    the numerator of the statistics;
  - around the pool there is a ring of cyan-blue interference haze, labelled monolayer
    or bare substrate, which also falls outside the component.

Using the component directly therefore **underestimates** what should be excluded, and
it does so asymmetrically across the classes.

## Approach: generate candidates deterministically, let the model only judge

This follows the principle used throughout the chain -- measurement is deterministic,
judgement goes to the model:

  1. take the thick-layer component as the seed;
  2. grow a region in the photo using a single scalar parameter t (colour tolerance),
     then fill holes and close, giving a family of **irregular** candidate boundaries.
     A larger t means a looser boundary.
  3. each round, draw the candidate boundary on the photo and ask Kimi one question
     only: **too_tight / about_right / too_loose**;
  4. hill-climb on t by bisection according to the answer, for at most N rounds;
  5. two consecutive about_right verdicts converge; otherwise use the last about_right,
     and if there was none, **fall back to the seed** (better to under-delete than to
     over-delete).

Every round's parameter, candidate area, model verdict and reason is written to
`refine_log`.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def grow_candidate(photo, seed, tol, max_dilate=400):
    """Grow one candidate boundary deterministically.

    Apply a tolerance criterion around the seed's mean colour, confine it to a dilated
    neighbourhood of the seed (so it cannot leak across the whole image), then fill holes
    and close. A larger tol is looser. Returns a boolean mask.
    """
    if not seed.any():
        return seed.copy()
    ref = photo[seed].reshape(-1, 3).mean(0)
    d = np.abs(photo.astype(np.int16) - ref.astype(np.int16)).sum(axis=2)
    near = d <= tol
    # Grow only near the seed, so distant regions of similar colour are not pulled in.
    # One distance transform does it in a single pass: binary_dilation(iterations=400)
    # is O(400*N) and would stall the process outright on a hundred-megapixel crop
    # (the same class of mistake as the border_trim bug).
    reach = ndimage.distance_transform_edt(~seed) <= max_dilate
    near &= reach
    # Keep only the part connected to the seed
    lab, _ = ndimage.label(near | seed)
    ids = np.unique(lab[seed])
    ids = ids[ids > 0]
    cand = np.isin(lab, ids)
    cand = ndimage.binary_closing(cand, np.ones((5, 5), bool))
    cand = ndimage.binary_fill_holes(cand)
    return cand


def _overlay(photo, cand, seed, max_side=900):
    """Draw the candidate boundary on the photo for the model: red = candidate boundary,
    cyan = seed boundary."""
    import cv2
    img = photo.copy()
    for m, col in ((seed, (0, 255, 255)), (cand, (255, 0, 0))):
        edge = ndimage.binary_dilation(m, np.ones((3, 3), bool), iterations=2) & ~m
        img[edge] = col
    tint = cand & ~seed
    img[tint] = (0.55 * img[tint] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    s = max(img.shape[0], img.shape[1]) / max_side
    if s > 1:
        img = cv2.resize(img, (int(img.shape[1] / s), int(img.shape[0] / s)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def build_role(KA):
    return KA.Role(
        name="melt_boundary_referee",
        system=(
            "你在看一张光学显微图的局部，上面画了两条边界：\n"
            "  青色线 = 分割模型给出的『厚层』连通域（种子）\n"
            "  红色线 + 红色半透明填充 = 本轮候选的剔除范围\n\n"
            "画面中央是一坨**前驱体熔池**（未反应的 MoO2/NaCl 残留），"
            "典型特征是大片连续、边界圆滑呈液滴状、青蓝或彩虹干涉色、"
            "内部压着深灰或黑色团块。它周围才是真正长出来的 MoS2 晶体"
            "（三角形或六边形、有笔直晶棱和 60/120 度夹角）。\n\n"
            "你只回答一件事：**红色候选范围相对这坨熔池，是太小、差不多、还是太大？**\n"
            "  - too_tight：熔池明显有一部分露在红色范围外面（常见的是外围那圈"
            "青蓝干涉晕，或者熔池内部被判成别的类别的斑点没被包进去）。\n"
            "  - about_right：红色范围基本贴着熔池的外缘，既没漏掉明显的熔池部分，"
            "也没有大片明显的三角形晶体被红色盖住。\n"
            "  - too_loose：红色范围已经吃到熔池外面，盖住了成片的、"
            "轮廓清晰的三角形/六边形晶体。\n\n"
            "判断的重点在**边缘**：顺着红线走一圈，看线的内外分别是什么。\n"
            "宁可判 too_tight 也不要判 too_loose 之后还说差不多——"
            "删掉真实晶体比少删一点危害大。"
        ),
        schema_hint=('{"verdict": "too_tight|about_right|too_loose", '
                     '"confidence": 0.0-1.0, '
                     '"reason": "一句话，中文，40 字以内，说边缘上看到了什么"}'),
        max_tokens=8192,
    )


def refine(photo, seed, pool, votes=3, max_rounds=5,
           tol_lo=30, tol_hi=330, log=None):
    """Settle the boundary through rounds of feedback. Returns (final mask, round log).

    With pool=None, or when the model is unavailable, the seed is returned unchanged.
    """
    rounds = []
    if pool is None or not getattr(pool.client, "available", False):
        return seed.copy(), [{"note": "模型不可用，退回种子（宁可少删）"}]
    KA = __import__("kimi_agents")
    role = build_role(KA)
    lo, hi = float(tol_lo), float(tol_hi)
    best, ok_streak = None, 0
    for rnd in range(max_rounds):
        t = round((lo + hi) / 2, 1)
        cand = grow_candidate(photo, seed, t)
        grew = float(cand.sum() / max(seed.sum(), 1))
        img = _overlay(photo, cand, seed)
        prompt = (f"第 {rnd + 1} 轮。当前候选范围是种子面积的 {grew:.2f} 倍"
                  f"（颜色容差 {t:.0f}）。红色范围相对这坨熔池是太小、差不多、还是太大？")
        v, tally, _ = pool.vote(role, prompt, [img], key="verdict")
        rec = {"round": rnd + 1, "tol": t, "area_ratio_to_seed": round(grew, 3),
               "candidate_px": int(cand.sum()), "verdict": v, "tally": tally}
        rounds.append(rec)
        if log:
            log(f"        第 {rnd+1} 轮 tol={t:.0f} 面积×{grew:.2f} -> {v} {tally}")
        if v == "about_right":
            best = (t, cand); ok_streak += 1
            if ok_streak >= 2:
                break
            lo, hi = t * 0.9, t * 1.1        # confirm once more in the neighbourhood
        elif v == "too_tight":
            ok_streak = 0; lo = t
        elif v == "too_loose":
            ok_streak = 0; hi = t
        else:
            break
    if best is None:
        rounds.append({"note": "没有任何一轮判为 about_right，退回种子（宁可少删）"})
        return seed.copy(), rounds
    rounds.append({"chosen_tol": best[0],
                   "final_area_ratio_to_seed": round(float(best[1].sum() /
                                                           max(seed.sum(), 1)), 3)})
    return best[1], rounds


# ---------------------------------------------------------------- satellites
def find_satellite_candidates(mask, main, target_class=3, radius_px=600,
                              min_px=400, max_cand=6, scale_div=4):
    """Deterministically list blobs near a main pool that *might* belong to it.

    A single colour tolerance cannot express "that droplet 200 px away came from
    the same spill" or "this arm is a scratch, not melt". Those are judgements a
    person makes by looking, so the code's job is only to enumerate the
    candidates and let the model decide. Returns [(label_id, bbox, area_px)].
    """
    from scipy import ndimage
    d = max(1, scale_div)
    m, mn = mask[::d, ::d], main[::d, ::d]
    near = (ndimage.distance_transform_edt(~mn) <= radius_px / d) & ~mn
    sel = (m == target_class) & near
    lab, _ = ndimage.label(sel)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    objs = ndimage.find_objects(lab)
    out = []
    for i in np.argsort(-sizes)[:max_cand]:
        i = int(i)
        if sizes[i] * d * d < min_px:
            break
        sl = objs[i - 1]
        out.append({"label_id": i, "area_px": int(sizes[i]) * d * d,
                    "bbox": [sl[1].start * d, sl[1].stop * d,
                             sl[0].start * d, sl[0].stop * d]})
    return out, (lab, d)


def build_satellite_role(KA):
    return KA.Role(
        name="satellite_referee",
        system=(
            "画面中央用青色圈出的是一坨已经确认的**前驱体熔池**"
            "（未反应的 MoO2/NaCl 残留）。另有一小块用黄色圈出。\n\n"
            "你只回答一件事：**黄色这块和青色那坨是不是同一次溢流出来的、"
            "属于同一个熔池系统？**\n\n"
            "  - same_spill：是同一次溢流。判据是二者之间有连续或断续的流痕、"
            "干涉色的色调和层序一致、黄色这块也呈液滴状且边界圆滑、"
            "中心同样压着深色未反应团块。**要一起剔除。**\n"
            "  - separate_melt：也是熔池，但明显是独立的一坨"
            "（中间隔着大片正常生长的晶体，没有流痕相连）。"
            "**也该剔除，但作为独立区域单独计量。**\n"
            "  - not_melt：根本不是熔池。可能是划痕（笔直、细长、"
            "两端不收口）、纤维或灰尘（无规则、有毛边）、"
            "或者是一片真正的多层晶体（有笔直晶棱和 60/120 度夹角）。"
            "**不要剔除。**\n"
            "  - uncertain：看不清。**不剔除**，标记待人工复核。\n\n"
            "特别注意那种从主熔池伸出去的细长\"臂\"：如果它笔直、宽度均匀、"
            "两端不收口，那多半是**划痕**而不是熔体流痕，判 not_melt。"
            "熔体流痕通常宽度不均、末端收口成圆头。"
        ),
        schema_hint=('{"verdict": "same_spill|separate_melt|not_melt|uncertain", '
                     '"exclude": true/false, "confidence": 0.0-1.0, '
                     '"reason": "一句话，中文，40 字以内"}'),
        max_tokens=8192,
    )
