"""Thick-layer region measurement and optional model-assisted eligibility review.

Connected-component geometry proposes regions and records their area fraction,
solidity, bounding box and colour. Optional vision-model review returns recorded
keep/exclude proposals under a common policy. Missing or unresolved model verdicts
are retained for review. An optical category is not independent physical evidence
of chemical composition; connected components are not separate crystal domains.
This source package does not claim a measured model-performance improvement."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# This package is a subpackage of MosaicAgent, so kimi_agents.py sits in the parent
# directory. MOSAIC_AGENT_DIR overrides that, for when the two are kept apart.
def _mosaic_dir() -> Path:
    env = os.environ.get("MOSAIC_AGENT_DIR")
    if env and (Path(env) / "kimi_agents.py").exists():
        return Path(env)
    parent = Path(__file__).resolve().parent.parent
    if (parent / "kimi_agents.py").exists():
        return parent
    raise ImportError(
        "找不到 kimi_agents.py。本包应作为子包放在 MosaicAgent 仓库根目录下，"
        "或用环境变量 MOSAIC_AGENT_DIR 指向 MosaicAgent 目录。")


def _import_kimi():
    p = _mosaic_dir()
    sys.path.insert(0, str(p))
    import kimi_agents as KA  # noqa
    return KA, p


# --------------------------------------------------------------- role definition
def build_role(KA):
    """The thick-layer region adjudicator.

    The class set is deliberately exhaustive. Experience (see the mosaic-agent-repo
    notes): when a classification model describes an image correctly but reaches the
    wrong verdict, the usual cause is an incomplete class set, not a weak model. Back
    when sample_edge was missing from the list, Kimi described the substrate edge
    accurately and then called it underexposed and asked for it to be discarded.
    """
    return KA.Role(
        name="region_adjudicator",
        system=(
            "你是二维材料 CVD 生长的显微图像分析员。你会看到一张大面积拼接图里"
            "**一块被分割模型判成\"厚层\"的连通区域**：左边是光学显微原图裁剪，"
            "右边是同一位置的分割结果（橄榄色=厚层，绿色=单层，深红=双层，黑=裸衬底）。\n\n"
            "你要回答的唯一问题是：**这块区域应不应该计入\"晶体层数分布\"统计？**\n"
            "统计的目的是描述衬底上长出来的 MoS2 晶体在单层/双层/多层之间怎么分配。"
            "所以凡是\"不是长出来的晶体\"的东西都要剔除。\n\n"
            "类别（先判类别，再由类别决定去留）：\n"
            "  - precursor_melt：前驱体熔池／未反应残留。特征是大片**连续**的厚区，"
            "干涉色偏青蓝或彩虹渐变，边界圆滑呈液滴状而**没有晶面棱角**，"
            "内部常压着深灰或黑色的团块（未反应的 MoO2/NaCl）。**剔除。**\n"
            "  - continuous_film：连续的多层薄膜，大面积连成一片、看不出单个晶体轮廓，"
            "不是离散生长的晶体。**剔除。**\n"
            "  - multilayer_crystals：真正的多层晶体聚集区。虽然连成一片，但能看出"
            "一个个**三角形或六边形**的轮廓、有笔直的晶棱、层与层之间有清晰的台阶边界。"
            "**保留**，这是真实的生长结果。\n"
            "  - thick_flake：单个厚片晶体，轮廓规则有棱角。**保留。**\n"
            "  - contamination：污染物——纤维、灰尘、胶残留、油渍、气泡。"
            "形状无规则、不服从晶体对称性、常有毛边或高光。**剔除。**\n"
            "  - scan_artifact：拼接或成像造成的伪影——接缝亮度台阶、虚焦区、"
            "过曝死白、扫描走出衬底外的区域。**剔除。**\n"
            "  - substrate_damage：衬底划痕、破损、腐蚀坑。**剔除。**\n"
            "  - uncertain：证据不足判不了。**保留**并标记待人工复核"
            "（宁可保留也不要误删真实数据）。\n\n"
            "三条容易搞错的判据：\n"
            "  1. **先找棱角**。判 precursor_melt 之前先问：这块区域的外边界上"
            "有没有任何一段是笔直的、成 60/120 度夹角的？只要有，就更可能是"
            "multilayer_crystals 而不是熔池。\n"
            "  2. **面积大不等于要剔除**。一大片真实的多层晶体也可以很大。"
            "决定去留的是\"是不是晶体\"，不是它占多少面积。\n"
            "  3. **青蓝色不等于熔池**。厚的 MoS2 在特定 SiO2 厚度下也会显青蓝。"
            "要结合边界形态一起判，不要只看颜色。\n\n"
            "会同时给你这块区域的数值特征，作为辅助，但**以图像为准**。"
        ),
        schema_hint=(
            '{"category": "precursor_melt|continuous_film|multilayer_crystals|thick_flake|'
            'contamination|scan_artifact|substrate_damage|uncertain", '
            '"exclude": true/false, "confidence": 0.0-1.0, '
            '"has_faceted_edges": true/false, '
            '"reason": "一句话，中文，40 字以内，只说决定性证据"}'),
        # kimi-k3 is a reasoning model, so its chain of thought counts against the
        # completion tokens too. The old limit of 1536 frequently hit
        # finish_reason='length' and returned an empty string, which the client rescued
        # by doubling and retrying -- workable, but it cost two or three extra
        # round trips on every region. 8192 gets it right the first time (grid_planner
        # in the same file also uses 8192).
        max_tokens=8192,
    )


# --------------------------------------------------------------- step 1: measurement
@dataclass
class RegionCandidate:
    rank: int
    label_id: int
    area_px: int
    pct_of_class: float          # percentage of all pixels in this class
    pct_of_all_crystal: float    # percentage of all crystal pixels (bare substrate excluded)
    bbox: tuple                  # (x0, x1, y0, y1) at full resolution
    solidity: float              # area / hole-filled area; closer to 1 means more solid
    bbox_fill: float             # area / bounding-box area
    mean_rgb: tuple
    def to_dict(self):
        return asdict(self)


def propose_regions(mask, photo, target_class=3, valid_mask=None,
                    min_pct_of_class=1.0, max_candidates=8, scale_div=4):
    """Measurement: find the large connected components of target_class and compute
    their features. Makes no keep/drop decision whatsoever.

    mask   : HxW uint8, 0=bare substrate 1=bilayer 2=monolayer 3=thick (VOC palette order)
    photo  : HxWx3 uint8 optical image
    valid_mask : HxW bool, True = the region counted in the statistics (the non-scanned
                 filler has already been excluded)
    Returns [RegionCandidate], sorted by descending area.
    """
    from scipy import ndimage
    d = max(1, int(scale_div))
    m = mask[::d, ::d]
    ph = photo[::d, ::d]
    vm = np.ones(m.shape, bool) if valid_mask is None else valid_mask[::d, ::d]

    sel = (m == target_class) & vm
    if not sel.any():
        return []
    # Closing bridges the small holes inside one region without being aggressive enough
    # to fuse discrete crystals into a single blob
    lab, n = ndimage.label(ndimage.binary_closing(sel, np.ones((5, 5))))
    # Closing grows the component beyond sel (it fills holes). The area must therefore
    # count only pixels genuinely belonging to target_class, or the fractions come out
    # above 100%.
    lab_in = np.where(sel, lab, 0)
    sizes = np.bincount(lab_in.ravel(), minlength=n + 1)
    sizes[0] = 0
    total_class = int(sel.sum())
    total_crystal = int(((m > 0) & vm).sum())
    objs = ndimage.find_objects(lab)

    out = []
    for i in np.argsort(-sizes)[:max_candidates]:
        i = int(i)
        if sizes[i] == 0 or sizes[i] < total_class * min_pct_of_class / 100.0:
            break
        sl = objs[i - 1]
        comp = (lab_in[sl] == i)
        filled = ndimage.binary_fill_holes(comp)
        h, w = comp.shape
        rgb = ph[sl][filled].reshape(-1, 3).mean(0) if filled.any() else np.zeros(3)
        out.append(RegionCandidate(
            rank=len(out) + 1, label_id=i,
            area_px=int(sizes[i]) * d * d,
            pct_of_class=round(sizes[i] / max(total_class, 1) * 100, 2),
            pct_of_all_crystal=round(sizes[i] / max(total_crystal, 1) * 100, 2),
            bbox=(sl[1].start * d, sl[1].stop * d, sl[0].start * d, sl[0].stop * d),
            solidity=round(float(comp.sum() / max(filled.sum(), 1)), 3),
            bbox_fill=round(float(comp.sum() / max(h * w, 1)), 3),
            mean_rgb=tuple(round(float(x), 1) for x in rgb)))
    return out, (lab_in, d)


# --------------------------------------------------------------- step 2: adjudication
def _crops(photo, mask, bbox, margin_frac=0.12, max_side=900):
    """The two images shown to the agent: a photo crop and the matching segmentation
    crop (BGR, following the MosaicAgent convention)."""
    import cv2
    PAL = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0]], np.uint8)
    x0, x1, y0, y1 = bbox
    mx = int((x1 - x0) * margin_frac)
    my = int((y1 - y0) * margin_frac)
    H, W = mask.shape
    X0, X1 = max(0, x0 - mx), min(W, x1 + mx)
    Y0, Y1 = max(0, y0 - my), min(H, y1 + my)
    ph = photo[Y0:Y1, X0:X1]
    mk = PAL[mask[Y0:Y1, X0:X1]]
    s = max(ph.shape[0], ph.shape[1]) / max_side
    if s > 1:
        wh = (int(ph.shape[1] / s), int(ph.shape[0] / s))
        ph = cv2.resize(ph, wh, interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, wh, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(ph, cv2.COLOR_RGB2BGR), cv2.cvtColor(mk, cv2.COLOR_RGB2BGR)


def adjudicate_regions(candidates, photo, mask, pool=None, votes=3, px_per_mm=None):
    """Review each proposed region; without an available model, keep it as uncertain.

    Votes may be a fixed count or adaptive, capped at five. The majority is evaluated
    on the keep/exclude decision, not on category names. Missing, invalid or tied
    verdicts become uncertain/keep for downstream review. Returned records include
    category, exclude, confidence, reason and vote tallies. These records preserve
    model proposals rather than certifying the region's physical identity."""
    if not candidates:
        return []
    if pool is None or not getattr(pool.client, "available", False):
        return [{**c.to_dict(), "category": "uncertain", "exclude": False,
                 "confidence": 0.0, "reason": "没有 Kimi API key，未做判定，按保留处理",
                 "_tally": {}, "_n_votes": 0, "_ai": False} for c in candidates]
    adaptive = isinstance(votes, str) and votes.strip().lower() == "adaptive"
    n_votes = 3 if adaptive else max(1, int(votes))

    KA, _ = _import_kimi()
    role = build_role(KA)

    def build(c):
        ph, mk = _crops(photo, mask, c.bbox)
        area_mm = (f"{c.area_px / (px_per_mm ** 2):.4f} mm^2"
                   if px_per_mm else f"{c.area_px:,} px")
        prompt = (
            f"这块厚层连通域的数值特征：\n"
            f"  面积：{area_mm}\n"
            f"  外接框：{c.bbox[1]-c.bbox[0]} x {c.bbox[3]-c.bbox[2]} px\n"
            f"  占本图全部厚层像素：{c.pct_of_class}%\n"
            f"  占本图全部晶体像素（不含裸衬底）：{c.pct_of_all_crystal}%\n"
            f"  实心度（面积/填洞后面积）：{c.solidity}\n"
            f"  外接框填充率：{c.bbox_fill}\n"
            f"  区域内原图平均 RGB：{c.mean_rgb}\n\n"
            f"第一张图是光学显微原图裁剪，第二张是同位置的分割结果。"
            f"判断这块区域应不应该计入晶体层数分布统计。")
        return prompt, [ph, mk]

    res = pool.map(role, candidates, build, key="category", decision_key="exclude",
                   use_votes=adaptive or n_votes > 1, adaptive=adaptive)
    out = []
    for c, r in zip(candidates, res):
        if r is None:
            out.append({**c.to_dict(), "category": "uncertain", "exclude": False,
                        "confidence": 0.0,
                        "reason": "智能体无有效返回或去留恰好平票，按保留处理",
                        "_tally": {}, "_n_votes": 0, "_ai": True})
        else:
            # Order matters: the deterministic measurements come second so they
            # override any field of the same name returned by the model. The other way
            # round, a single hallucinated label_id would be enough for
            # build_exclusion_mask to delete the wrong component -- and to attach a
            # plausible-sounding justification to it (fixed 2026-08-29).
            out.append({**r, **c.to_dict(), "verdict": {k: v for k, v in r.items()
                                                        if k not in c.to_dict()},
                        "_ai": True})
    return out


def build_exclusion_mask(shape, verdicts, lab_pack, fill_interior=True):
    """Paint the excluded components into a full-resolution boolean mask.

    `fill_interior=True` is the important part. The raw connected component only
    contains pixels the segmenter labelled thick-layer, so the interior of a
    precursor pool is riddled with holes: the dark unreacted clumps and the
    speckle inside the pool get labelled 1L / 2L / bare and therefore survive
    into the statistics as if they were grown crystals. Filling the holes makes
    the mask a closed region, so *everything* inside the pool boundary leaves the
    layer statistics -- which is what the pool physically is: not substrate on
    which crystals grew, but residue lying on top of it.

    Anything inside this mask is dropped from the numerator and the denominator
    alike; its area is reported separately by stage_stats.
    """
    from scipy import ndimage
    lab_in, d = lab_pack
    small = np.zeros(lab_in.shape, bool)
    for v in verdicts:
        if not v.get("exclude"):
            continue
        comp = (lab_in == v["label_id"])
        if fill_interior:
            # close first so a ragged rim does not leave the region open, then
            # fill: an open region would let fill_holes do nothing at all.
            comp = ndimage.binary_fill_holes(
                ndimage.binary_closing(comp, np.ones((9, 9), bool)))
        small |= comp
    if d == 1:
        return small
    return np.repeat(np.repeat(small, d, axis=0), d, axis=1)[:shape[0], :shape[1]]
