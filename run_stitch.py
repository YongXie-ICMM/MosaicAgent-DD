#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_stitch.py — 大面积显微拼接的总控。

一句话：把一堆按蛇形扫描拍出来的瓦片，自动挑片、配准、平场、融合成一张大图，
拿不准的地方交给 Kimi 看图投票决定。

流水线
------
  1 scan       清点数据，解析列/行、扫描方向、采集顺序
  2 cache      解码并缓存 1/N 缩略图（幂等，重跑不重算），同一趟算好全帧对焦分数
  3 qc         三路门：全帧对焦分数 < 0.4 x 参考 -> 闭式丢弃（不问模型）；
               0.4–0.6 x 参考或任何统计标记 -> Kimi 逐张看图投票定去留；其余保留
  4 conflict   同一格位有多张候选时 -> Kimi 对比上下邻居投票二选一
  5 register   相位相关求相邻位移 -> 全局加权最小二乘解出每张的位置
  6 flatfield  估计渐晕场
  7 render     羽化融合出图
  8 inspect    Kimi 抽查拼接结果的裁剪，发现接缝/错位就调参重来
  9 report     写运行报告

每一步的结果都落盘到 state.json，可以中断后接着跑。

给学生用
--------
  换一个数据集，直接跑。第 1 步的硬编码命名规则认不出瓦片时，会自动让 Kimi
  从文件清单反推网格布局，再用闭式规则对着清单核一遍（正则匹配率 >= 95%、
  序号能解析、各列张数），通过了才存到 work/layout.json 并继续拼接；
  校验不过就把校验报告（没对上的文件名、撞车的序号）喂回去让 Kimi 修一次
  （--plan-attempts，默认 2 次；第三次从来没救回过任何清单，别调高），
  每次尝试都记进 layout.json 的 attempts[]。下次重跑直接复用这份布局，不再问 Kimi：
      python3 run_stitch.py --data /path/to/new_dataset --out mosaic.png
  只看布局不拼接（同样会校验并存 layout.json）：
      python3 run_stitch.py --data /path/to/new_dataset --plan
  没有 Kimi key、也没有缓存的布局时，会停下来要一份手写布局（照样校验）：
      python3 run_stitch.py --data ... --layout my_layout.json --no-ai
  硬编码规则认得出瓦片、但想强制走布局推断：加 --auto-layout。
  校验不通过就停，把没对上的文件名列出来（work/layout_rejected.json），
  绝不悄悄退回别的规则。
  没有 Kimi key 也能跑，AI 环节会自动跳过、退回纯统计判据：
      python3 run_stitch.py --data ... --no-ai
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tiles as T                      # noqa: E402
import kimi_agents as KA               # noqa: E402
import stitch_profile as SP             # noqa: E402
from tools.stitch_setup.acquisition_metadata import inspect_acquisition  # noqa: E402


def log(msg=""):
    print(msg, flush=True)


def banner(step, title):
    log(f"\n{'='*72}\n[{step}] {title}\n{'='*72}")


# ------------------------------------------------------------------ 状态
class State:
    """把每一步的产物落盘。位置矩阵单独存 .npy，其余存 JSON。"""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save(self):
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=1,
                                        default=float), encoding="utf-8")

    def done(self, step) -> bool:
        return step in self.d.get("_done", [])

    def mark(self, step):
        self.d.setdefault("_done", [])
        if step not in self.d["_done"]:
            self.d["_done"].append(step)
        self.save()


# ------------------------------------------------------------------ 3 质检
CLOSED_FORM_CONFIDENCE = 0.8     # 闭式 defocus 判决的置信度（实验 E：模型对这 25 张全票 drop）


def closed_form_defocus_verdict(t, ref: float) -> dict:
    """defocus 标记的瓦片不问模型：判决由 tiles.flag_tiles 的比值规则给出。"""
    ratio = float(t.focus_score) / ref if ref else float("nan")
    return {"category": "defocus", "keep": False, "confidence": CLOSED_FORM_CONFIDENCE,
            "reason": (f"闭式规则：全帧对焦分数 {t.focus_score:.2f} < {T.FOCUS_RATIO_DEFOCUS:g} x "
                       f"参考 {ref:.2f}（比值 {ratio:.2f}），未调模型"),
            "_ai": False}


def step_qc(ts, cache_dir, pool: KA.AgentPool | None, st: State, votes: int):
    """三路门（实验 E 的 g4）：
         defocus 标记            -> 闭式丢弃，不调模型（Figure 3a：25 张，模型 25/25 全票同意）
         focus_band 或其他任何标记 -> Kimi 看图投票（去留按 keep 字段计多数）
         没有标记                -> 保留，不看
       无 AI 时只丢闭式 defocus 的那些；送模型的瓦片没有判决（无 AI、平票、失败）
       一律保留并记进 no_verdict 给人看。哪些判决是闭式的记在 state.json 的
       qc.closed_form 里，判决本身带 _ai=false。"""
    banner(3, "质检：闭式对焦门 + Kimi 看图定去留")
    qc = T.flag_tiles(ts)
    for w in qc["warnings"]:
        log(f"  [提醒] {w}")
    ref = qc["focus_ref"]
    flagged = [t for t in qc["flagged"] if "conflict" not in t.flags]
    defocus = [t for t in flagged if "defocus" in t.flags]
    to_model = [t for t in flagged if "defocus" not in t.flags]
    n_votes = int(getattr(pool, "votes", votes) or votes) if pool is not None else 0
    log(f"  对焦参考 ref = {ref if ref is None else f'{ref:.2f}'}"
        f"（p{T.FOCUS_REF_PERCENTILE:g}，{qc['focus_ref_n']} 张有纹理瓦片）；"
        f"闭式 defocus {len(defocus)} 张，focus_band {len(qc['focus_band'])} 张，"
        f"low_texture {len(qc['low_texture'])} 张；"
        f"提名 {len(flagged)} 张（共 {len(ts)} 张），其中 {len(to_model)} 张送模型")

    focus_med = float(np.median([t.focus for t in ts]))
    verdicts = {t.tid: closed_form_defocus_verdict(t, ref) for t in defocus}
    closed_form = sorted(verdicts)
    no_verdict, split = [], []
    n_calls = 0

    if pool is None or not pool.client.available:
        log(f"  [跳过 AI] 无 AI 回退：只丢弃闭式 defocus 标记的 {len(defocus)} 张；"
            f"其余 {len(to_model)} 张提名瓦片保留，记入 no_verdict")
        no_verdict = [t.tid for t in to_model]
    else:
        n_cols = len({x.col for x in ts})

        def build(t):
            im = T.load_small(cache_dir, t)
            if not np.isfinite(t.focus_score):
                focus_line = "全帧对焦分数不可用；"
            elif ref:
                focus_line = (f"全帧对焦分数 score = {t.focus_score:.2f}，"
                              f"整个数据集的参考值 ref = {ref:.2f}（有纹理瓦片的 p75），"
                              f"score/ref = {t.focus_score / ref:.2f}"
                              f"（< {T.FOCUS_RATIO_DEFOCUS:g} 判虚焦，"
                              f"{T.FOCUS_RATIO_DEFOCUS:g}–{T.FOCUS_RATIO_BAND:g} 是需要看图的过渡带）；")
            else:
                focus_line = (f"全帧对焦分数 score = {t.focus_score:.2f}，"
                              f"没有参考值（有纹理的瓦片不足 {T.FOCUS_MIN_REF_TILES} 张）；")
            prompt = (f"这张瓦片来自 {t.col} 列（方向 {t.direction}）的第 {t.name}，"
                      f"属于一张 {n_cols} 列的大面积拼接图。\n"
                      f"统计指标：{focus_line}"
                      f"缩略图 Laplacian 方差(focus) = {t.focus:.1f}，"
                      f"整个数据集的 focus 中位数是 {focus_med:.1f}；"
                      f"灰度均值 {t.mean:.0f}、标准差 {t.std:.1f}、"
                      f"过曝欠曝像素占比 {t.clip_frac:.3f}；"
                      f"文件大小 {t.nbytes/1e6:.2f} MB。\n"
                      f"自动筛查给它打的标签：{','.join(t.flags)}。\n"
                      f"请判断这张瓦片该保留还是丢弃。")
            return prompt, [im]

        res = pool.map(KA.ROLE_TILE_INSPECTOR, to_model, build, key="category",
                       decision_key="keep")
        n_calls = sum(int(r.get("_n_asked", n_votes)) for r in res if r) + sum(1 for r in res if not r) * n_votes
        for t, r in zip(to_model, res):
            if r:
                verdicts[t.tid] = {**r, "_ai": True}
                if r.get("_split"):
                    split.append(t.tid)
            else:
                no_verdict.append(t.tid)      # 平票或没拿到回答：保留，交给人看
        if no_verdict:
            log(f"  {len(no_verdict)} 张没有多数判决（平票/无回答），按保留处理并记入 no_verdict")

    drop = {tid for tid, v in verdicts.items() if v.get("keep") is False}
    by_cat: dict[str, int] = {}
    for v in verdicts.values():
        by_cat[v.get("category", "?")] = by_cat.get(v.get("category", "?"), 0) + 1
    log(f"  判定分布：{by_cat}")
    log(f"  判定丢弃 {len(drop)} 张（闭式 {len([x for x in drop if x in closed_form])}，"
        f"模型 {len([x for x in drop if x not in closed_form])}）")
    for tid in sorted(drop):
        v = verdicts[tid]
        tag = "闭式" if v.get("_ai") is False else "模型"
        log(f"    - {tid:24s} {tag} {v.get('category','')}  {str(v.get('reason',''))[:52]}")

    st.d["qc"] = {
        "verdicts": verdicts, "drop": sorted(drop),
        "closed_form": closed_form, "no_verdict": sorted(no_verdict), "split": sorted(split),
        "focus_median": focus_med,
        "focus": {"ref": ref, "ref_n": qc["focus_ref_n"],
                  "n_defocus": len(defocus), "n_focus_band": len(qc["focus_band"]),
                  "n_low_texture": len(qc["low_texture"]),
                  "n_nominated": len(flagged), "n_to_model": len(to_model),
                  "n_calls": n_calls, "votes": n_votes,
                  "thresholds": {"ratio_defocus": T.FOCUS_RATIO_DEFOCUS,
                                 "ratio_band": T.FOCUS_RATIO_BAND,
                                 "min_contrast": T.FOCUS_MIN_CONTRAST,
                                 "ref_percentile": T.FOCUS_REF_PERCENTILE},
                  "warnings": list(qc["warnings"])},
    }
    st.mark("qc")
    return drop


# ------------------------------------------------------------------ 4 冲突
def step_conflict(ts, cache_dir, conflicts, pool: KA.AgentPool | None, st: State):
    banner(4, "位置冲突：同一格位多张候选，二选一")
    if not conflicts:
        log("  没有冲突组")
        st.d["conflict"] = {}
        st.mark("conflict")
        return set()

    by_col: dict[int, list] = {}
    for t in ts:
        by_col.setdefault(t.col, []).append(t)
    for v in by_col.values():
        v.sort(key=lambda t: t.key)

    losers, decisions = set(), {}
    for group in conflicts:
        col = group[0].col
        seq = by_col[col]
        lo = min(seq.index(g) for g in group)
        hi = max(seq.index(g) for g in group)
        before = seq[lo - 1] if lo - 1 >= 0 else None
        after = seq[hi + 1] if hi + 1 < len(seq) else None
        names = [g.name for g in group]
        log(f"  L{col}: 候选 {names}"
            f"  上邻={before.name if before else '-'} 下邻={after.name if after else '-'}")

        if pool is None or not pool.client.available:
            # 无 AI 时的兜底：留 focus 最高的
            win = max(range(len(group)), key=lambda i: group[i].focus)
            reason = "无 AI，按 focus 最高保留"
        else:
            imgs = [T.load_small(cache_dir, g) for g in group]
            ctx = []
            if before is not None:
                ctx.append(T.load_small(cache_dir, before))
            if after is not None:
                ctx.append(T.load_small(cache_dir, after))
            prompt = (
                f"第 {col} 列的同一个网格位置上有 {len(group)} 张候选照片，只能留一张。\n"
                + "".join(f"  候选 {i}：文件 {g.name}，focus={g.focus:.1f}，"
                          f"灰度均值 {g.mean:.0f}\n" for i, g in enumerate(group))
                + f"前 {len(group)} 张图是候选，按上面的顺序排列。"
                + (f"接下来 1 张是它上方相邻位置的照片（{before.name}）。" if before is not None else "")
                + (f"最后 1 张是它下方相邻位置的照片（{after.name}）。" if after is not None else "")
                + "\n请选出与上下邻居视野最连贯、对焦和曝光最匹配的那一张。")
            r = pool.vote(KA.ROLE_CONFLICT_ADJUDICATOR, prompt, imgs + ctx,
                          key="winner_index")
            w, tally, answers = r
            if w is None:
                win, reason = int(np.argmax([g.focus for g in group])), "AI 无结论，按 focus 兜底"
            else:
                win = int(w)
                best = max([a for a in answers if str(a.get("winner_index")) == w],
                           key=lambda a: float(a.get("confidence", 0) or 0))
                reason = f"{best.get('reason','')} 票数={tally}"
        win = max(0, min(win, len(group) - 1))
        for i, g in enumerate(group):
            if i != win:
                losers.add(g.tid)
        decisions[f"L{col}"] = {"candidates": names, "winner": group[win].name,
                                "reason": reason}
        log(f"    -> 保留 {group[win].name}  ({reason[:70]})")

    st.d["conflict"] = decisions
    st.mark("conflict")
    return losers



# ------------------------------------------------------------------ 5b 救回
def step_rescue(ts, drop: set, positions, st: State, min_cover=0.55):
    """把"丢了就会留洞"的瓦片救回来。

    质检判虚焦是对的，但**丢弃的前提是别人能补上这块地方**。实测 Figure 3a
    时，25 张虚焦瓦片全部聚在中央一个大液滴周围——那里有凸起，焦面对不上，
    所以整片都虚，也就没有清晰的邻居能覆盖。丢掉的结果是拼接图正中间开了
    几个白窟窿，抽查环节自己报了 major/seam_line。

    论文图里糊一点远好过一个白洞，所以规则是：一张被判丢弃的瓦片，
    若它的画幅有超过 (1-min_cover) 的面积没有任何保留瓦片覆盖，就救回来。
    """
    if not drop:
        return set()
    keep_idx = [i for i, t in enumerate(ts) if t.tid not in drop]
    drop_idx = [i for i, t in enumerate(ts) if t.tid in drop]
    W, H = T.TILE_W, T.TILE_H
    # 用一张粗网格算覆盖率就够了，精度到 1/32 张瓦片
    cell = W / 32.0
    x0 = positions[:, 0].min(); y0 = positions[:, 1].min()
    gw = int((positions[:, 0].max() + W - x0) / cell) + 2
    gh = int((positions[:, 1].max() + H - y0) / cell) + 2
    cov = np.zeros((gh, gw), bool)

    def box(i):
        cx = int((positions[i, 0] - x0) / cell); cy = int((positions[i, 1] - y0) / cell)
        return cy, cy + max(1, int(H / cell)), cx, cx + max(1, int(W / cell))

    for i in keep_idx:
        a, b, c, d = box(i)
        cov[a:b, c:d] = True

    rescued = set()
    for i in drop_idx:
        a, b, c, d = box(i)
        frac = float(cov[a:b, c:d].mean()) if b > a and d > c else 1.0
        if frac < min_cover:
            rescued.add(ts[i].tid)
            log(f"    救回 {ts[i].tid:24s} 该处仅 {frac*100:.0f}% 被其他瓦片覆盖")
    if rescued:
        log(f"  救回 {len(rescued)} 张（丢了会留洞）；最终丢弃 {len(drop)-len(rescued)} 张")
    else:
        log("  无需救回：被丢弃的瓦片都有邻居覆盖")
    st.d["rescue"] = sorted(rescued)
    st.mark("rescue")
    return rescued


# ------------------------------------------------------------------ 8 抽查
def step_inspect(mosaic_path, pool: KA.AgentPool | None, st: State,
                 n_crops=6, crop=700, seed=0):
    banner(8, "抽查拼接结果")
    if pool is None or not pool.client.available:
        log("  [跳过 AI]")
        return []
    img = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
    if img is None:
        log(f"  读不到 {mosaic_path}")
        return []
    h, w = img.shape[:2]
    rng = random.Random(seed)
    crops, meta = [], []
    tries = 0
    while len(crops) < n_crops and tries < n_crops * 12:
        tries += 1
        x = rng.randint(0, max(0, w - crop))
        y = rng.randint(0, max(0, h - crop))
        c = img[y:y + crop, x:x + crop]
        # 全白的地方是画布空隙，看了也没意义
        if c.size == 0 or c.mean() > 250:
            continue
        crops.append(c)
        meta.append((x, y))

    def build(item):
        (c, (x, y)) = item
        return (f"这是拼接结果里 ({x}, {y}) 处 {c.shape[1]}x{c.shape[0]} 的裁剪，"
                f"整幅图 {w}x{h}。请找拼接缺陷。"), [c]

    res = pool.map(KA.ROLE_SEAM_INSPECTOR, list(zip(crops, meta)), build,
                   key="severity")
    findings = []
    for (x, y), r in zip(meta, res):
        if not r:
            continue
        sev = r.get("severity", "none")
        log(f"  ({x:>6},{y:>6})  {sev:<6} {r.get('defects')}  {str(r.get('where',''))[:40]}")
        if sev in ("minor", "major"):
            findings.append({"xy": [x, y], **{k: v for k, v in r.items()
                                              if not k.startswith("_")}})
    st.d["inspect"] = findings
    st.mark("inspect")
    return findings


# ------------------------------------------------------------------ plan
PLAN_ATTEMPTS = 2            # 校验反馈回路的上限：实验 D 里第三次 0/24 从没救回过什么
LISTING_MAX_ODD_SHAPES = 12  # 每个容器最多列这么多种不合群式样
LISTING_MAX_NAMES = 4        # 一种式样不超过这么多张就把名字全列出来
LISTING_MAX_LINES = 120      # 清单折叠后的行数上限（每行是一个容器或一串相同容器）


def _shape(name: str) -> str:
    return re.sub(r"\d", "#", name)


def _digest(names: list[str]) -> tuple[str, list[str]]:
    """一个容器里的文件名 -> (摘要文本, 内层目录名列表)。

    只给首尾几个文件是不够的：真正决定正则怎么写的恰恰是那些**不合群**的文件名
    （补拍留下的 0035.5.png 就是这么被漏掉的）。所以先归纳出主流命名式样，再把
    偏离主流的文件名**按式样分组计数**列出来——以前是 sorted(odd)[:12]，30 个
    .png.xml 边车文件就能把唯一一张补拍挤出窗口（实验 D 的 ds_G，第一次 0/3）。
    "如 A .. B" 先排序再取，给的是 min..max 而不是文件系统序。内层目录单独返回，
    摘要文本里不含它，这样相同的列才能在清单里折叠成一行。"""
    bases = sorted(os.path.basename(n) for n in names)
    shapes: dict[str, list[str]] = {}
    for b in bases:
        shapes.setdefault(_shape(b), []).append(b)
    # 主流式样 = 张数最多的；打平时图片式样优先（一张瓦片一个边车文件时 30 vs 30）
    main_shape, main_names = max(
        shapes.items(),
        key=lambda kv: (len(kv[1]), kv[0].rsplit(".", 1)[-1].lower() in T.IMAGE_EXTS, kv[0]))
    parts = [f"{len(names)} 个文件",
             f"主流式样 {main_shape} x{len(main_names)}（如 {main_names[0]} .. {main_names[-1]}）"]
    odd = sorted(((sh, bs) for sh, bs in shapes.items() if sh != main_shape),
                 key=lambda kv: (-len(kv[1]), kv[0]))
    if odd:
        groups = []
        for sh, bs in odd[:LISTING_MAX_ODD_SHAPES]:
            if len(bs) <= LISTING_MAX_NAMES:
                groups.append(f"{sh} x{len(bs)}: {', '.join(bs)}")
            else:
                groups.append(f"{sh} x{len(bs)}（如 {bs[0]} .. {bs[-1]}）")
        if len(odd) > LISTING_MAX_ODD_SHAPES:
            groups.append(f"...还有 {len(odd) - LISTING_MAX_ODD_SHAPES} 种式样")
        parts.append("不合群的文件名（按式样分组）：" + "; ".join(groups))
    inner = sorted({n.split("/")[0] for n in names if "/" in n})
    return "  ".join(parts), inner


def build_listing(data_dir: str) -> str:
    """数据目录 -> 给规划器看的清单文本。每个容器（zip 或子目录）一行摘要；
    **相邻且摘要相同**的容器折叠成一行（"L01_down.zip .. L42_up.zip（42 个压缩包，
    摘要相同）"），摘要不同的一律单独列出——以前 entries[:40] 会把第 43 列的补拍
    藏在窗口外（实验 D 的 ds_E，第一次 0/3）。普通文件夹不再出现 "内层目录 ."。"""
    rows = []          # (显示名, 类别, 摘要或 None, 内层目录列表)
    for e in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, e)
        if e.lower().endswith(".zip") and os.path.isfile(p):
            try:
                with zipfile.ZipFile(p) as z:
                    names = [n for n in z.namelist() if not n.endswith("/")]
                d, inner = _digest(names) if names else ("(空)", [])
                rows.append((e, "zip", d, inner))
            except Exception as ex:
                rows.append((e, "plain", f"<打不开: {ex}>", []))
        elif os.path.isdir(p):
            sub = []
            for root, dirs, fs in os.walk(p):
                dirs.sort()
                sub += [os.path.relpath(os.path.join(root, f), p).replace(os.sep, "/")
                        for f in sorted(fs)]
            d, inner = _digest(sub) if sub else ("(空)", [])
            rows.append((e + "/", "dir", d, inner))
        else:
            rows.append((e, "plain", None, []))

    lines = []
    i = 0
    while i < len(rows):
        name, kind, d, inner = rows[i]
        if d is None or kind == "plain":
            lines.append(name if d is None else f"{name}  {d}")
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and rows[j + 1][1] == kind and rows[j + 1][2] == d:
            j += 1
        run = rows[i:j + 1]
        if len(run) == 1:
            lines.append(f"{name}  {d}" + (f"  内层目录 {', '.join(inner[:4])}" if inner else ""))
        else:
            inner_shapes = Counter(_shape(x) for r in run for x in r[3])
            shapes_txt = ", ".join(f"{sh} x{n}" for sh, n in sorted(inner_shapes.items(),
                                                                    key=lambda kv: (-kv[1], kv[0])))
            lines.append(f"{run[0][0]} .. {run[-1][0]}  "
                         f"（{len(run)} 个{'压缩包' if kind == 'zip' else '文件夹'}，摘要相同）  {d}"
                         + (f"  内层目录式样 {shapes_txt}" if inner_shapes else ""))
        i = j + 1
    if len(lines) > LISTING_MAX_LINES:
        lines = lines[:LISTING_MAX_LINES] + [f"...（折叠后共 {len(lines)} 行，只列了前 {LISTING_MAX_LINES}）"]
    return "\n".join(lines)


def plan_prompt(data_dir: str, text: str) -> str:
    return (f"下面是一个显微拼接数据集目录 {data_dir} 的清单：\n\n{text}\n\n"
            f"请反推它的网格布局。")


def repair_prompt(base: str, prev_layout: dict | None, report: dict | None,
                  max_unmatched: int = 30, max_dup: int = 10) -> str:
    """校验不过时的第二次提问：原清单 + 上次布局 + 校验报告 + 没匹配上的文件
    （<= max_unmatched）+ 撞车的 idx（<= max_dup）+ "请修正"。校验器说什么就
    喂什么，不另外加提示。"""
    lines = [base, ""]
    if prev_layout is None or report is None:
        lines.append("你上一次的回答不是带 group_regex 的可解析 JSON。请只输出 JSON。")
        return "\n".join(lines)
    lines += ["你上一次给的布局是：",
              json.dumps({k: prev_layout.get(k) for k in T.LAYOUT_FIELDS if k in prev_layout},
                         ensure_ascii=False), "",
              "用它对着真实清单做闭式校验，没有通过。校验报告：",
              T.format_layout_report(report), ""]
    c, t = report.get("containers", {}), report.get("tiles", {})
    unmatched = list(c.get("unmatched", [])) + list(t.get("unmatched_images", []))
    if unmatched:
        lines.append("你上一次给的正则匹配不上这些文件: " + ", ".join(unmatched[:max_unmatched])
                     + ("…" if len(unmatched) > max_unmatched else "") + "; 请修正。")
    if t.get("duplicate_idx"):
        lines.append("这些文件被解析到了同一个 idx: " + "; ".join(t["duplicate_idx"][:max_dup])
                     + ("…" if len(t["duplicate_idx"]) > max_dup else "") + "; 请修正。")
    bad = list(c.get("bad_index", [])) + list(t.get("bad_idx", []))
    if bad:
        lines.append("这些序号解析不成数字: " + "; ".join(bad[:max_dup]) + "; 请修正。")
    lines.append("请给出修正后的完整布局 JSON。")
    return "\n".join(lines)


def step_plan(data_dir, pool: KA.AgentPool | None, prompt: str | None = None) -> dict | None:
    """给学生用：把一个陌生数据集的文件清单丢给 Kimi，让它反推网格布局（一次提问）。

    prompt 为 None 时自己列清单并提问；plan_layout_with_repair 传入现成的（首问或
    修正提示）。返回 Kimi 的原始推断 dict（**尚未校验**，校验和存盘见 commit_layout）；
    无 AI、或答不出带 group_regex 的 JSON 时返回 None。"""
    if prompt is None:
        banner(0, "让 Kimi 从文件清单反推网格布局")
        text = build_listing(data_dir)
        log(text)
        prompt = plan_prompt(data_dir, text)
    if pool is None or not pool.client.available:
        log("\n  [无 AI，跳过反推]")
        return None
    r = pool.ask(KA.ROLE_GRID_PLANNER, prompt)
    log("\nKimi 的布局推断：")
    log(json.dumps(r, ensure_ascii=False, indent=2))
    if not isinstance(r, dict) or "_unparsed" in r or not r.get("group_regex"):
        log("  Kimi 没有给出带 group_regex 的 JSON，这次推断作废")
        return None
    return r


def _usage_snapshot(client) -> dict:
    u = getattr(client, "usage", None)
    return {k: int(getattr(u, k, 0) or 0) for k in ("calls", "prompt_tokens", "completion_tokens")}


def plan_layout_with_repair(data_dir: str, work: Path, pool: KA.AgentPool | None,
                            max_attempts: int = PLAN_ATTEMPTS) -> dict:
    """规划器的校验反馈回路（实验 D）：第 1 次用清单提问；校验不过就把上次布局 +
    校验报告 + 没匹配上的文件 + 撞车的 idx 喂回去再问一次（隐藏在清单窗口外的
    补拍就是这样第二次 6/6 修好的）。停下来的条件：校验通过 / 达到 max_attempts /
    新的 (group_regex, tile_regex) 和之前某次一模一样（schema 表达不了这份清单，
    再问也是循环）。校验器仍是唯一的闸门，通过才存 work/layout.json；每次尝试
    都记进 attempts[]（连同 attempts_used、source、stopped_because），失败时同样
    记进 layout_rejected.json 并抛 LayoutError。"""
    if pool is None or not pool.client.available:
        step_plan(data_dir, pool)          # 把清单打出来给人看，然后照旧停下
        raise T.LayoutError("Kimi 未启用，无法反推布局。用 --layout <json> 手工给一份"
                            f"（字段：{', '.join(T.LAYOUT_FIELDS)}）。")
    max_attempts = max(1, int(max_attempts))
    banner(0, f"让 Kimi 从文件清单反推网格布局（校验不过就把报告喂回去，最多 {max_attempts} 次）")
    text = build_listing(data_dir)
    log(text)
    base = plan_prompt(data_dir, text)
    model = getattr(pool.client, "model", None)

    attempts, seen, prompt = [], [], base
    prev_layout, report, stopped = None, None, None
    for n in range(1, max_attempts + 1):
        log(f"\n  --- 第 {n}/{max_attempts} 次提问 ---")
        u0, t0 = _usage_snapshot(pool.client), time.time()
        r = step_plan(data_dir, pool, prompt=prompt)
        u1 = _usage_snapshot(pool.client)
        rec = {"n": n, "seconds": round(time.time() - t0, 1),
               "prompt_tokens": u1["prompt_tokens"] - u0["prompt_tokens"],
               "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"],
               "feedback": n > 1}
        if r is None:
            rec.update(ok=False, layout=None, errors=["Kimi 没有给出带 group_regex 的 JSON"],
                       unmatched_images=[], duplicate_idx=[])
            attempts.append(rec)
            pass                                  # keep the last parsed layout/report for the repair prompt
        else:
            layout = {k: r.get(k) for k in T.LAYOUT_FIELDS if k in r}
            report = T.validate_layout(data_dir, layout)
            rec.update(ok=bool(report["ok"]), layout=layout, errors=list(report["errors"]),
                       warnings=list(report["warnings"]),
                       unmatched_images=list(report["tiles"]["unmatched_images"][:20]),
                       duplicate_idx=list(report["tiles"]["duplicate_idx"][:10]))
            attempts.append(rec)
            prev_layout = layout
            if report["ok"]:
                stopped = "ok"
                break
            log(T.format_layout_report(report))
            pair = (layout.get("group_regex"), layout.get("tile_regex"))
            if pair in seen:
                stopped = "repeated_regex"
                log("  这次给的正则和之前某次一模一样：清单里有 layout schema 表达不了的东西，"
                    "不再重试")
                break
            seen.append(pair)
        if n == max_attempts:
            stopped = "max_attempts"
            break
        log(f"  第 {n} 次布局没通过校验，把校验报告喂回去再问一次")
        prompt = repair_prompt(base, prev_layout, report)

    ledger = {"attempts": attempts, "attempts_used": len(attempts),
              "max_attempts": max_attempts, "stopped_because": stopped}
    source = "kimi" if len(attempts) == 1 else "kimi+feedback"
    if stopped == "ok":
        log(f"  第 {len(attempts)} 次通过校验（source={source}）")
        return commit_layout(data_dir, prev_layout, work, source=source, model=model, extra=ledger)
    log(f"  {len(attempts)} 次都没通过校验（stopped_because={stopped}）")
    if prev_layout is not None:
        # 校验不过：commit_layout 写 layout_rejected.json（带台账）并抛 LayoutError
        commit_layout(data_dir, prev_layout, work, source=source, model=model, extra=ledger)
    work.mkdir(parents=True, exist_ok=True)
    p = work / "layout_rejected.json"
    p.write_text(json.dumps({"layout": None, "source": source, "model": model,
                             "data_dir": data_dir, "validation": None, **ledger},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    raise T.LayoutError(f"Kimi {len(attempts)} 次都没有给出可用的布局（{stopped}）。"
                        f"用 --layout <json> 手工给一份（字段：{', '.join(T.LAYOUT_FIELDS)}）。"
                        f"\n  台账：{p}")


# ------------------------------------------------------------------ 1 布局
def layout_path(work: Path) -> Path:
    return work / "layout.json"


def stop(msg: str, code: int = 2):
    """打印原因并以非零退出：上游 flakepipeline/stages.py 靠退出码判断拼接失败。"""
    log(msg)
    sys.exit(code)


def _load_layout_file(path: Path) -> dict:
    """接受两种形状：裸的布局 dict，或 work/layout.json 那种带 validation 的外壳。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and isinstance(d.get("layout"), dict):
        return d["layout"]
    return d


def commit_layout(data_dir: str, layout: dict, work: Path, source: str,
                  model: str | None = None, save: bool = True,
                  extra: dict | None = None) -> dict:
    """校验 -> 打印报告 -> 存盘。不通过就把报告写到 work/layout_rejected.json
    并抛 LayoutError。这里没有任何回退：布局对不上清单，就停下来把没对上的
    名字给人看。通过且 save=True 时写 work/layout.json（连同校验报告和
    Kimi 指出的 anomalies），下次运行直接复用、不再问 Kimi。extra 是随记录
    一起落盘的附加字段（规划器的 attempts[] 台账）。"""
    report = T.validate_layout(data_dir, layout)
    log(T.format_layout_report(report))
    rec = {"layout": {k: layout.get(k) for k in T.LAYOUT_FIELDS if k in layout},
           "source": source, "model": model, "data_dir": data_dir,
           "anomalies": list(layout.get("anomalies") or []),
           "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "validation": report, **(extra or {})}
    work.mkdir(parents=True, exist_ok=True)
    if not report["ok"]:
        p = work / "layout_rejected.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        raise T.LayoutError(T.format_layout_report(report) + f"\n  完整报告：{p}")
    if save:
        p = layout_path(work)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  布局已通过校验，存到 {p}")
    return rec["layout"]


def resolve_layout(data_dir: str, work: Path, pool: KA.AgentPool | None,
                   layout_file: str | None = None,
                   plan_attempts: int = PLAN_ATTEMPTS) -> dict:
    """第 1 步的布局来源，按优先级：
         1. --layout 手工给的文件（照样校验，通过后存成 work/layout.json）；
         2. work/layout.json 里已经校验过的布局——重新对着清单核一遍（闭式、
            不联网），**不问 Kimi**；
         3. Kimi 反推（只在有 key 时），校验不过就把报告喂回去再问，最多
            plan_attempts 次（plan_layout_with_repair），校验通过才存盘。
       三样都没有就抛 LayoutError，消息里明确要 --layout。任何一步校验不过
       都直接抛，不会悄悄换下一个来源。"""
    if layout_file:
        log(f"  使用手工布局 {layout_file}")
        return commit_layout(data_dir, _load_layout_file(Path(layout_file)), work,
                             source=f"file:{layout_file}")
    cached = layout_path(work)
    if cached.exists():
        log(f"  复用已校验的布局 {cached}（不问 Kimi，只重新对一遍清单）")
        rec = json.loads(cached.read_text(encoding="utf-8"))
        layout = rec["layout"] if isinstance(rec.get("layout"), dict) else rec
        try:
            return commit_layout(data_dir, layout, work, source=rec.get("source", "cache"),
                                 model=rec.get("model"), save=False)
        except T.LayoutError as e:
            raise T.LayoutError(f"{e}\n  缓存的布局 {cached} 对不上现在的清单。"
                                f"数据目录变了就删掉它重跑，或用 --layout 给一份新的。")
    if pool is not None and pool.client.available:
        return plan_layout_with_repair(data_dir, work, pool, max_attempts=plan_attempts)
    raise T.LayoutError(
        f"硬编码命名规则认不出 {data_dir} 里的瓦片，Kimi 未启用（--no-ai 或缺 KIMI_API_KEY），"
        f"{cached} 也不存在。请用 --layout <json> 手工给一份布局"
        f"（字段：{', '.join(T.LAYOUT_FIELDS)}），或配好 key 后跑 --plan。")


def step_scan(data_dir: str, work: Path, pool: KA.AgentPool | None,
              layout_file: str | None = None, auto_layout: bool = False,
              plan_attempts: int = PLAN_ATTEMPTS):
    """第 1 步：先按硬编码规则清点；一张都认不出（或 --auto-layout / --layout）
    就走布局推断。返回 (tiles, layout)，layout 为 None 表示走的硬编码规则。
    布局不可用时直接停（非零退出），不猜。"""
    layout, ts = None, []
    try:
        if layout_file or auto_layout:
            layout = resolve_layout(data_dir, work, pool, layout_file, plan_attempts)
        else:
            ts = T.scan_dataset(data_dir)
            if not ts:
                log(f"  硬编码命名规则（a_L<列>_<up|down>.zip）在 {data_dir} 里"
                    f"没认出瓦片，改走布局推断")
                layout = resolve_layout(data_dir, work, pool, plan_attempts=plan_attempts)
        if layout is not None:
            ts = T.scan_dataset(data_dir, layout=layout)
    except T.LayoutError as e:
        stop(f"  布局不可用，停止：\n{e}")
    if not ts:
        stop(f"  {data_dir} 里没找到瓦片")
    return ts, layout


# ------------------------------------------------------------------ preflight

def _preflight_scan(data_dir, work, layout_file=None, auto_layout=False):
    """Read-only scan. Unknown layouts get all image candidates checked before AI."""
    layout, source = None, "historical-filenames"
    if layout_file:
        layout, source = _load_layout_file(Path(layout_file)), f"file:{layout_file}"
    elif not auto_layout:
        grid = SP.scan_grid_tiles(data_dir, ignored_paths=(work,))
        if grid is not None:
            return grid, None, "flat-physical-grid"
        ts = T.scan_dataset(data_dir)
        if ts:
            # The historical parser deliberately ignores unknown names. On a new
            # acquisition that must not silently omit another image container.
            selected = {(str(Path(t.zip_path).resolve()), t.inner) for t in ts}
            extras = []
            containers, _ = T._layout_candidates(data_dir)
            for _, container, is_zip in containers:
                if Path(container).resolve() == work.resolve():
                    continue
                files, _ = T._list_container(container, is_zip)
                for name, inner, _ in files:
                    if (Path(name).suffix.lower().lstrip(".") in T.IMAGE_EXTS
                            and (str(Path(container).resolve()), inner) not in selected):
                        extras.append(f"{Path(container).name}/{inner}")
            extras += [p.name for p in Path(data_dir).iterdir() if p.is_file()
                       and not p.name.startswith((".", "_"))
                       and p.suffix.lower().lstrip(".") in T.IMAGE_EXTS]
            if extras:
                raise SP.ProfileError("Historical filenames match only part of the image inventory: "
                                      + ", ".join(extras[:10]) + ". Supply --layout to select an explicit acquisition.")
            return ts, None, source
    if layout is None and layout_path(work).exists():
        layout, source = _load_layout_file(layout_path(work)), "cached-layout"
    if layout is not None:
        if str(layout.get("major_axis") or "column").lower() != "column":
            raise SP.ProfileError("Stitch profiles currently support column-major layouts only; "
                                  "a row-major layout needs an explicit coordinate conversion.")
        try:
            ts = T.scan_dataset(data_dir, layout=layout)
        except T.LayoutError as exc:
            filename = layout_file or str(layout_path(work))
            raise T.LayoutError(f"Layout {filename} does not match the selected dataset: {exc}") from exc
        return ts, layout, source
    # The planner sees names only after every candidate image passes preflight.
    candidates = []
    containers, _ = T._layout_candidates(data_dir)
    for ci, (_, path, is_zip) in enumerate(containers):
        files, _ = T._list_container(path, is_zip)
        for name, inner, nbytes in files:
            if Path(name).suffix.lower().lstrip(".") in T.IMAGE_EXTS:
                order = len(candidates)
                candidates.append(T.Tile(ci, ci, "down", order, float(order), inner,
                                         path, inner, order, nbytes))
    if not candidates:
        raise SP.ProfileError("No raw tiles found. Supply a complete mosaic_r<row>_c<col> grid, "
                              "known a_L<column>_<direction>.zip files, or --layout. "
                              "A prestitched mosaic alone cannot establish stitching parameters.")
    return candidates, None, "needs-layout"


def _preflight_report(resolved, inventory, binding, source, acquisition=None):
    return {"schema_version": 1, "ok": True, "layout_source": source,
            "resolved": resolved, "input_evidence": inventory, "run_binding": binding,
            "acquisition_metadata": acquisition,
            "note": "Header/profile checks do not establish registration quality; verify adjacent pairs."}


def _write_preflight(path, report):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise SP.ProfileError("Preflight report path must end in .json.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _ai_pool(args, work, cache=True):
    client, pool = None, None
    if not args.no_ai:
        client = KA.KimiClient(cache_dir=None)
        if client.available:
            if cache:
                client.cache_dir = work / "kimi_cache"
                client.cache_dir.mkdir(parents=True, exist_ok=True)
            pool = KA.AgentPool(client, workers=args.workers, votes=args.votes)
            log(f"Kimi: {client.model} @ {client.base}，每个判断 {args.votes} 路投票")
    if pool is None:
        log("Kimi: 未启用（--no-ai 或缺 KIMI_API_KEY），AI 环节将跳过")
    return client, pool


def _layout_only(args, data_dir, work):
    # Filename-layout planning is independent of image geometry. It does not
    # claim a stitch profile was verified and never builds an image cache.
    work.mkdir(parents=True, exist_ok=True)
    _, pool = _ai_pool(args, work, cache=False)
    try:
        if args.layout:
            commit_layout(data_dir, _load_layout_file(Path(args.layout)), work,
                          source=f"file:{args.layout}")
        elif pool is None or not pool.client.available:
            step_plan(data_dir, pool)
            log("  没有可校验的布局。有 key 时重跑 --plan，"
                "或用 --layout <json> 给一份手写布局再跑 --plan 校验。")
        else:
            plan_layout_with_repair(data_dir, work, pool, max_attempts=args.plan_attempts)
    except T.LayoutError as exc:
        stop(f"  布局不可用：\n{exc}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="大面积显微拼接（Kimi 辅助）")
    ap.add_argument("--data", required=True, help="放 zip 的数据目录")
    ap.add_argument("--work", default=None, help="缓存/中间结果目录")
    ap.add_argument("--out", default="mosaic.png")
    ap.add_argument("--scale-div", type=int, default=None, help="缓存降采样倍数（profile 或默认 8）")
    ap.add_argument("--out-scale", type=float, default=None, help="出图比例（profile 或默认 1/8）")
    ap.add_argument("--votes", type=int, default=3, help="每个判断独立问几次再投票")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--no-ai", action="store_true", help="不用 Kimi，全走统计判据")
    ap.add_argument("--plan", action="store_true",
                    help="只做布局反推（校验并存 work/layout.json）然后退出")
    ap.add_argument("--layout", default=None,
                    help="手工给一份布局 JSON（group_regex/tile_regex/...），照样校验")
    ap.add_argument("--auto-layout", action="store_true",
                    help="跳过硬编码命名规则，直接走布局推断（缓存 -> Kimi）")
    ap.add_argument("--plan-attempts", type=int, default=PLAN_ATTEMPTS,
                    help="布局推断校验不过时，把报告喂回去再问的次数上限（含第一次；"
                         "默认 2，第三次从没救回过任何清单）")
    ap.add_argument("--full", action="store_true", help="渲染时读原图而不是缓存")
    ap.add_argument("--force", action="store_true", help="忽略已有 state，从头跑")
    ap.add_argument("--stitch-profile", help="图像尺寸与原图像素位移标定 JSON")
    ap.add_argument("--input-preflight", help="Prepared setup preflight.json; recheck its input and metadata binding before execution")
    ap.add_argument("--preflight-only", action="store_true",
                    help="只读检查所有原图尺寸、profile 和工作目录；必须配合 --no-ai")
    ap.add_argument("--preflight-report", help="可选预检 JSON 报告路径（预检默认只打印）")
    args = ap.parse_args()
    if args.preflight_only and not args.no_ai:
        ap.error("--preflight-only requires --no-ai; preflight never initializes or calls Kimi")
    if args.input_preflight and (args.plan or args.auto_layout):
        ap.error("--input-preflight cannot be combined with --plan or --auto-layout; use the prepared layout")
    if args.preflight_only and args.plan:
        ap.error("--preflight-only and --plan are separate modes")


    data_dir = os.path.abspath(args.data)
    work = Path(args.work or os.path.join(data_dir, "_stitch_work")).resolve()
    if args.plan:
        _layout_only(args, data_dir, work)
        return
    # No directory creation, Kimi client, cache writes, or inference above this gate.
    try:
        profile = SP.load_profile(args.stitch_profile) if args.stitch_profile else None
        ts, layout, source = _preflight_scan(data_dir, work, args.layout, args.auto_layout)
        inventory = SP.inspect_tiles(ts)
        resolved = SP.resolve_profile(profile, inventory, args.scale_div, args.out_scale, args.full)
        acquisition, acquisition_warnings = inspect_acquisition(
            data_dir, inventory, "flat-grid" if source == "flat-physical-grid" else source)
        if args.input_preflight:
            if source == "needs-layout":
                raise SP.ProfileError("Bound execution requires the prepared, explicit tile layout.")
            SP.check_input_preflight(args.input_preflight, resolved, inventory, acquisition)
        binding = SP.make_run_binding(resolved, inventory, acquisition)
        SP.check_work_binding(work, binding)
        if args.preflight_only and source == "needs-layout":
            raise SP.ProfileError("Image dimensions passed, but the grid is not yet known. "
                                  "Supply --layout for a completely offline selected-tile preflight.")
    except (SP.ProfileError, T.LayoutError, OSError, ValueError, zipfile.BadZipFile) as exc:
        stop(f"Preflight failed: {exc}")
    args.scale_div, args.out_scale = resolved["scale_div"], resolved["out_scale"]
    report = _preflight_report(resolved, inventory, binding, source, acquisition)
    for warning in resolved["warnings"] + acquisition_warnings:
        log(f"WARNING: {warning}")
    log(f"Preflight: {inventory['tile_count']} tiles, native {inventory['image_size']}, "
        f"profile={resolved['profile']['name']}, cache=1/{args.scale_div}, output={args.out_scale:g}")
    if args.preflight_only:
        if args.preflight_report:
            _write_preflight(args.preflight_report, report)
        log(json.dumps(report, ensure_ascii=False, indent=2))
        return

    work.mkdir(parents=True, exist_ok=True)
    client, pool = _ai_pool(args, work)

    t_start = time.time()
    if source == "needs-layout":
        ts, layout = step_scan(data_dir, work, pool, args.layout, True, args.plan_attempts)
        try:
            if str(layout.get("major_axis") or "column").lower() != "column":
                raise SP.ProfileError("Row-major layouts are not supported by the stitching profile path.")
            inventory = SP.inspect_tiles(ts)
            resolved = SP.resolve_profile(profile, inventory, args.scale_div, args.out_scale, args.full)
            acquisition, acquisition_warnings = inspect_acquisition(data_dir, inventory, "validated-layout")
            binding = SP.make_run_binding(resolved, inventory, acquisition)
            source = "validated-layout"
            report = _preflight_report(resolved, inventory, binding, source, acquisition)
        except (SP.ProfileError, T.LayoutError) as exc:
            stop(f"Preflight failed: {exc}")
    elif layout is not None and args.layout:
        commit_layout(data_dir, layout, work, source=f"file:{args.layout}")

    st = State(work / "state.json")
    if args.force:
        st.d = {}  # The binding check already passed; --force cannot bypass it.
    st.d["run_binding"] = binding
    st.d["preflight"] = report
    st.d["layout"] = {"used": layout is not None, "source": source,
                      "file": str(layout_path(work)) if layout is not None else None}
    st.save()
    _write_preflight(work / "preflight.json", report)
    if args.preflight_report and Path(args.preflight_report).resolve() != work / "preflight.json":
        _write_preflight(args.preflight_report, report)
    cache_dir = str(work / "cache")
    with SP.geometry_context(resolved["profile"]):
        _run_pipeline(args, work, cache_dir, ts, pool, client, st, t_start)


def _run_pipeline(args, work, cache_dir, ts, pool, client, st, t_start):
    banner(2, f"建缓存（1/{args.scale_div}）+ 全帧对焦分数")
    info = T.build_cache(ts, cache_dir, scale_div=args.scale_div)
    log(f"  新解码 {info['cached']} 张，旧缓存补算全帧分数 {info.get('rescored', 0)} 张，"
        f"合计 {info['total']} 张；指标索引 {info.get('index')}")

    qc0 = T.flag_tiles(ts)
    drop = step_qc(ts, cache_dir, pool, st, args.votes)
    losers = step_conflict(ts, cache_dir, qc0["conflicts"], pool, st)

    # 配准对**全部**瓦片做，包括被判丢弃的：只有先知道它们落在哪，
    # 才能判断丢了会不会留洞（见 step_rescue）。虚焦瓦片相关不上时，
    # register 会退回先验位移并把权重压到 1e-3，不会污染全局解。
    cand = [t for t in ts if t.tid not in losers]
    log(f"\n  进入配准的瓦片：{len(cand)} / {len(ts)}（冲突落选 {len(losers)}）")

    banner(5, "配准")
    import register as R
    edges, rstats = R.build_edges(cand, cache_dir, scale_div=args.scale_div,
                                  workers=args.workers)
    log(f"  边 {rstats.get('n_edges')} 条，剔除 {rstats.get('n_rejected')} 条")
    log(f"  实测纵向位移 {rstats.get('measured_dy')}, 横向位移 {rstats.get('measured_dx')}")
    log(f"  实测重叠：纵 {rstats.get('measured_overlap_y_px')} px，"
        f"横 {rstats.get('measured_overlap_x_px')} px")
    pos, diag = R.solve_positions(len(cand), edges)
    log(f"  全局求解残差 RMS {diag.get('residual_rms'):.2f} px，"
        f"连通分量 {diag.get('n_components')}")
    np.save(work / "positions.npy", pos)
    st.d["register"] = {"stats": rstats, "diag": {k: v for k, v in diag.items()
                                                  if k != "per_edge_residual"}}
    st.mark("register")

    banner("5b", "唯一覆盖救回")
    rescued = step_rescue(cand, drop, pos, st)
    final_drop = drop - rescued
    sel = [i for i, t in enumerate(cand) if t.tid not in final_drop]
    keep = [cand[i] for i in sel]
    pos = pos[sel]
    log(f"  最终参与渲染：{len(keep)} / {len(ts)}")

    banner(6, "平场估计")
    import blend as B
    ff = B.estimate_flatfield(keep, cache_dir, scale_div=args.scale_div)
    np.save(work / "flatfield.npy", ff)
    log(f"  渐晕场 {ff.shape}，动态范围 {ff.min():.3f}-{ff.max():.3f}")

    banner(7, "渲染")
    out = os.path.abspath(args.out)
    rinfo = B.render(keep, pos, out, out_scale=args.out_scale,
                     cache_dir=cache_dir, flatfield=ff, use_full=args.full)
    log(f"  出图 {rinfo['out_w']}x{rinfo['out_h']} -> {out}")
    st.d["render"] = rinfo
    st.mark("render")

    findings = step_inspect(out, pool, st)

    banner(9, "报告")
    st.d["summary"] = {
        "tiles_total": len(ts), "tiles_used": len(keep),
        "dropped": sorted(final_drop),
        "rescued": sorted(rescued), "conflict_losers": sorted(losers),
        "overlap_x_px": rstats.get("measured_overlap_x_px"),
        "overlap_y_px": rstats.get("measured_overlap_y_px"),
        "out": out, "seconds": round(time.time() - t_start, 1),
        "kimi": str(client.usage) if client is not None else "disabled",
    }
    st.mark("summary")
    log(json.dumps(st.d["summary"], ensure_ascii=False, indent=1))
    if findings:
        log(f"\n  抽查发现 {len(findings)} 处可疑，详见 {st.path}")
    log(f"\n完成，用时 {time.time()-t_start:.0f}s。状态：{st.path}")


if __name__ == "__main__":
    main()
