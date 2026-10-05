# -*- coding: utf-8 -*-
"""tiles.py — Figure 3a 马赛克数据集的清点、缓存与质检。

数据的实际形态（不是理想网格，别假设）：
  - 17 个 zip，a_L18_down .. a_L34_down，列号 18-34，奇偶交替 down/up。
  - zip 内层文件夹通常叫 L29_up/，但 a_L29_up.zip 叫 a_L29_up/ —— 任何
    按 "L*/" 通配的脚本都会漏掉这一列，所以这里只按 zip 名解析列号方向，
    内层路径一律用 namelist 实测。
  - 文件名一般是 0001.png..0054.png，但 a_L19_up.zip 没有 0035.png，
    却有 0035.5.png 和 0035.6.png（补拍），所以序号必须按 float 解析。
  - a_L26_down / a_L27_up 各多一张 0055.png。

"顺序" 和 "物理行" 不是一回事：down 列自上而下拍，up 列自下而上拍，
所以 up 列的第 1 张在物理最下方。这里只给出 nominal_row 作为配准的初值，
真实位置由 register.py 解出来。

换数据集（第 1 步的布局推断）：命名规则不同的数据集不改代码，由
run_stitch.py 让 Kimi 从清单反推一份 layout（group_regex / tile_regex /
serpentine / reverse_direction_token ...），再由这里的 validate_layout() 用
闭式规则对着真实清单核一遍（正则能编译、容器和文件的匹配率、序号能解析、
各列张数），通过了才由 scan_dataset(layout=...) 按它清点。模型只负责猜，
判定全在代码里；对不上就抛 LayoutError 把没对上的名字列出来，不回退。

对焦门（2026-09-02 实验 E 之后）：build_cache 在解码原图的同一趟里用
flakepipeline/focus_metrics.focus_metrics 给**全帧**打分（Laplacian 方差，最长边
缩到 960），存进 Tile.focus_score / Tile.contrast 并写入缓存目录的 focus_index.json，
下次重跑不再解原图。flag_tiles 用比值规则提名：参考 ref = 有纹理瓦片分数的 p75，
score < 0.4 ref 判 defocus（闭式丢弃，不问模型），0.4–0.6 ref 判 focus_band
（送模型看图）。1/8 缓存图上的 Laplacian 方差（Tile.focus）只作提示信息，
**不能**拿它套比值规则——在 Figure 3a 上会标出 129 张，而全帧分数恰好标出
那 25 张真糊的。
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

TILE_W, TILE_H = 3840, 2160

# 实测标定（2026-08-27，对 ~60 张瓦片做高通 + 全域零均值 NCC 得到）。
# 采集脚本 02Auto_Snake_Scan.py 里写的 PX_PER_STEP = 25.6 是错的，
# 高了 2.06%；由它推出的 FOV_STEPS_X/Y 因此也都是错的。
# 这些值只用来给配准搜索一个先验中心，真实位置一律由配准解出。
PX_PER_STEP_MEASURED = 25.08      # 脚本写 25.6
NOMINAL_PITCH_Y = 1908.0          # 列内步距，实测（脚本推算值是 1945.6）
NOMINAL_PITCH_X = 3581.0          # 列间步距，实测（脚本推算值是 3660.8）
# 位移台 Y 轴与相机轴之间有 2.238 度转角：每走一个纵向步，图像还横向平移
# 74.6 px，符号随扫描方向翻转。一列 53 步累积 3952 px，比一张瓦片还宽，
# 所以整幅图是平行四边形而不是矩形。这是本数据集最容易让人栽跟头的地方。
SHEAR_PER_ROW = 74.6
STAGE_ROT_DEG = 2.238
ZIP_RE = re.compile(r"^a_L(?P<col>\d+)_(?P<dir>up|down)\.zip$")
NAME_RE = re.compile(r"^(?P<idx>\d+(?:\.\d+)?)\.png$", re.I)

# 布局校验的阈值（validate_layout）。容器名 / 瓦片名至少要有这么高的匹配率；
# 各列张数偏离中位数超过 count_tol（不足 2 张按 2 张算）只作提醒，
# 因为"各列张数不等"是真实数据的性质，不是正则写错。
LAYOUT_MIN_MATCH = 0.95
LAYOUT_COUNT_TOL = 0.10
LAYOUT_FIELDS = ("group_regex", "tile_regex", "serpentine", "major_axis",
                 "reverse_direction_token", "anomalies", "confidence", "notes")


IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}


class LayoutError(ValueError):
    """推断出的布局对不上实际清单。message 就是 format_layout_report() 的文本，
    里面列着没匹配上的名字。"""


@dataclass
class Tile:
    col: int
    col_idx: int
    direction: str      # 归一成 down（正向，第 1 张在最上）/ up（反向）
    order: int          # 采集顺序，0 起
    key: float          # 文件名里的数字，允许 35.5 这种
    name: str
    zip_path: str       # 容器路径：zip 文件；按 layout 清点的文件夹数据集则是目录
    inner: str          # 容器内完整路径
    nominal_row: int    # 物理行初值，0 = 最上
    nbytes: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    focus: float = float("nan")     # 1/scale_div 缓存图上的 Laplacian 方差，只作提示，不做判定
    clip_frac: float = float("nan")
    focus_score: float = float("nan")   # 全帧对焦分数（focus_metrics.focus_metrics 的 score）
    contrast: float = float("nan")      # 全帧灰度标准差（同上的 contrast）；< 4 视为无纹理
    flags: list = field(default_factory=list)

    @property
    def tid(self) -> str:
        return f"L{self.col}_{self.direction}/{self.name}"


def scan_dataset(data_dir: str, layout: dict | None = None) -> list[Tile]:
    """清点 zip，返回按 (列, 采集顺序) 排好的 Tile 列表。

    layout 为 None 时走 Figure 3a 的硬编码命名规则（下面这段是回归基线，
    一个字都别动）；给了 layout 就按推断出的正则清点，见 scan_with_layout。
    """
    if layout is not None:
        return scan_with_layout(data_dir, layout)
    zips = []
    for fn in sorted(os.listdir(data_dir)):
        m = ZIP_RE.match(fn)
        if m:
            zips.append((int(m.group("col")), m.group("dir"), os.path.join(data_dir, fn)))
    zips.sort()
    cols = [z[0] for z in zips]

    tiles: list[Tile] = []
    for col_idx, (col, direction, zp) in enumerate(zips):
        with zipfile.ZipFile(zp) as z:
            entries = []
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = os.path.basename(info.filename)
                m = NAME_RE.match(base)
                if not m:
                    continue
                entries.append((float(m.group("idx")), base, info.filename, info.file_size))
        entries.sort()
        n = len(entries)
        for order, (key, base, inner, size) in enumerate(entries):
            row = order if direction == "down" else (n - 1 - order)
            tiles.append(Tile(col=col, col_idx=col_idx, direction=direction,
                              order=order, key=key, name=base, zip_path=zp,
                              inner=inner, nominal_row=row, nbytes=size))
    return tiles


# ---------------------------------------------------------------- 布局推断
def _layout_candidates(data_dir: str):
    """可能是列容器的目录项：zip 文件和子目录，按名字排序。

    以 . 或 _ 开头的一律跳过（.DS_Store、默认的 _stitch_work 工作目录都住在
    数据目录里，它们不是数据，不该进匹配率的分母），其余非 zip 的普通文件
    （README、标定表）也跳过。跳过的名字原样记进报告，不悄悄吞掉。
    """
    cands, ignored = [], []
    for fn in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, fn)
        if fn.startswith((".", "_")):
            ignored.append(fn)
            continue
        is_zip = fn.lower().endswith(".zip") and os.path.isfile(p)
        if is_zip or os.path.isdir(p):
            cands.append((fn, p, is_zip))
        else:
            ignored.append(fn)
    return cands, ignored


def _list_container(path: str, is_zip: bool):
    """容器里的文件 -> [(basename, 容器内相对路径, 字节数)]，按路径排序。
    __MACOSX/ 和点文件（macOS 打包留下的 ._xxx）不是瓦片，单独记成 skipped。"""
    raw = []
    if is_zip:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir():
                    raw.append((info.filename, info.file_size))
    else:
        for root, dirs, fs in os.walk(path):
            dirs.sort()
            for f in sorted(fs):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, path).replace(os.sep, "/")
                raw.append((rel, os.path.getsize(full)))
    raw.sort()
    files, skipped = [], []
    for rel, size in raw:
        parts = rel.split("/")
        if any(p.startswith(".") or p == "__MACOSX" for p in parts):
            skipped.append(rel)
            continue
        files.append((os.path.basename(rel), rel, size))
    return files, skipped


def _compile_field(layout: dict, field_name: str, group: str, errors: list):
    pat = layout.get(field_name)
    if not isinstance(pat, str) or not pat:
        errors.append(f"{field_name} 缺失或不是字符串")
        return None
    try:
        rx = re.compile(pat)
    except re.error as e:
        errors.append(f"{field_name} 不是合法正则：{pat!r}（{e}）")
        return None
    if group not in rx.groupindex:
        errors.append(f"{field_name} 缺少命名组 (?P<{group}>...)：{pat!r}")
        return None
    return rx


def _match_container(rx, name: str):
    """容器名用 search（允许正则只写核心部分）；zip 再试一次去掉后缀的名字，
    这样 Kimi 写不写 \\.zip 都行。命名组取值不受部分匹配影响。"""
    m = rx.search(name)
    if m is None and name.lower().endswith(".zip"):
        m = rx.search(name[:-4])
    return m


def _match_tile(rx, base: str, rel: str):
    """瓦片名必须整体匹配：部分匹配会把 0035.5.png 悄悄读成 idx=5，
    这正是要靠校验挡住的那类错。先试文件名，再试容器内相对路径。"""
    return rx.fullmatch(base) or rx.fullmatch(rel)


def _parse_layout(data_dir: str, layout: dict,
                  min_match: float = LAYOUT_MIN_MATCH,
                  count_tol: float = LAYOUT_COUNT_TOL):
    """把 layout 对着真实清单核一遍。返回 (report, groups)；groups 只在
    report['ok'] 时可信。全部是闭式规则，没有任何随机或模型调用。"""
    errors: list[str] = []
    warnings: list[str] = []
    report = {"ok": False, "errors": errors, "warnings": warnings,
              "thresholds": {"min_match": min_match, "count_tol": count_tol},
              "containers": {"total": 0, "matched": 0, "unmatched": [], "ignored": [],
                             "bad_index": [], "duplicate_index": []},
              "tiles": {"total": 0, "matched": 0, "unmatched": [], "unmatched_images": [],
                        "duplicate_idx": [], "bad_idx": [], "skipped": []},
              "columns": [], "count_outliers": []}
    if not isinstance(layout, dict):
        errors.append(f"layout 不是 dict，而是 {type(layout).__name__}")
        return report, []
    grx = _compile_field(layout, "group_regex", "index", errors)
    trx = _compile_field(layout, "tile_regex", "idx", errors)
    major = str(layout.get("major_axis") or "column").lower()
    if major not in ("column", "row"):
        errors.append(f"major_axis 应为 column 或 row，得到 {layout.get('major_axis')!r}")
    if not os.path.isdir(data_dir):
        errors.append(f"数据目录不存在：{data_dir}")
    if errors:
        return report, []

    has_dir = "direction" in grx.groupindex
    serp = bool(layout.get("serpentine", False))
    rev = str(layout.get("reverse_direction_token") or "").strip().lower()

    # ---- 容器
    cands, ignored = _layout_candidates(data_dir)
    report["containers"]["ignored"] = ignored
    report["containers"]["total"] = len(cands)
    if not cands:
        errors.append(f"{data_dir} 里没有任何 zip 或子目录（跳过了 {ignored}）")
    unmatched_c, bad_index, groups = [], [], []
    for name, path, is_zip in cands:
        m = _match_container(grx, name)
        if m is None:
            unmatched_c.append(name)
            continue
        idx_s = m.group("index")
        try:
            f = float(idx_s)
            if not f.is_integer():
                raise ValueError("not an integer")
            index = int(f)
        except (TypeError, ValueError):
            bad_index.append(f"{name}: index={idx_s!r}")
            continue
        token = m.group("direction") if has_dir else None
        groups.append({"name": name, "path": path, "is_zip": is_zip,
                       "index": index, "token": token, "entries": []})
    n_c = len(cands)
    matched_c = n_c - len(unmatched_c)
    report["containers"].update(matched=matched_c, unmatched=unmatched_c,
                                bad_index=bad_index)
    if n_c and matched_c / n_c < min_match:
        errors.append(f"group_regex 只匹配了 {matched_c}/{n_c} 个容器"
                      f"（{100 * matched_c / n_c:.1f}% < {100 * min_match:.0f}%）")
    if bad_index:
        errors.append(f"{len(bad_index)} 个容器的 index 解析不成整数")
    seen: dict[int, list[str]] = {}
    for g in groups:
        seen.setdefault(g["index"], []).append(g["name"])
    dups = [f"{k}: {v}" for k, v in sorted(seen.items()) if len(v) > 1]
    report["containers"]["duplicate_index"] = dups
    if dups:
        errors.append("多个容器解析到同一个序号（分不清哪个是哪一列）：" + "; ".join(dups))

    # ---- 瓦片
    unmatched_t, bad_idx, skipped_all, empty, dup_idx = [], [], [], [], []
    n_files = 0
    for g in groups:
        files, skipped = _list_container(g["path"], g["is_zip"])
        skipped_all += [f"{g['name']}/{s}" for s in skipped]
        for base, rel, size in files:
            n_files += 1
            m = _match_tile(trx, base, rel)
            if m is None:
                unmatched_t.append(f"{g['name']}/{rel}")
                continue
            try:
                key = float(m.group("idx"))
            except (TypeError, ValueError):
                bad_idx.append(f"{g['name']}/{rel}: idx={m.group('idx')!r}")
                continue
            g["entries"].append((key, base, rel, size))
        g["entries"].sort()
        if not g["entries"]:
            empty.append(g["name"])
        # 同一容器里两个文件解析到同一个 idx（例如补拍的 0035.5 被 (?:\.\d+)? 吞掉了）
        # 会在缓存路径上撞车、静默丢掉一张：必须报错，不能只是警告
        by_key: dict[float, list[str]] = {}
        for key, base, rel, size in g["entries"]:
            by_key.setdefault(key, []).append(rel)
        for key, rels in sorted(by_key.items()):
            if len(rels) > 1:
                dup_idx.append(f"{g['name']}: idx {key:g} <- {rels}")
    matched_t = n_files - len(unmatched_t)
    # 95% 的容忍只给 Thumbs.db 这类杂物；图片文件一张都不能漏——漏掉的那 5% 就是
    # 静默消失的瓦片（在 Figure 3a 的真实清单上，最自然的正则会漏掉两张补拍）
    unmatched_img = [u for u in unmatched_t
                     if u.lower().rsplit(".", 1)[-1] in IMAGE_EXTS]
    report["tiles"].update(total=n_files, matched=matched_t, unmatched=unmatched_t,
                           unmatched_images=unmatched_img, duplicate_idx=dup_idx,
                           bad_idx=bad_idx, skipped=skipped_all)
    if groups and n_files == 0:
        errors.append("匹配到的容器里没有任何文件")
    if unmatched_img:
        errors.append(f"tile_regex 漏掉了 {len(unmatched_img)} 张图片瓦片（图片必须全部匹配）："
                      + ", ".join(unmatched_img[:8]) + ("…" if len(unmatched_img) > 8 else ""))
    junk = [u for u in unmatched_t if u not in unmatched_img]
    if junk:
        # non-image files (Thumbs.db, notes, logs) are not tiles: report, do not fail
        warnings.append(f"{len(junk)} 个非图片文件没匹配上（已忽略）：" + ", ".join(junk[:6])
                        + ("…" if len(junk) > 6 else ""))
    if dup_idx:
        errors.append("同一容器里多个文件解析到同一个 idx（会在缓存路径上撞车）：" + "; ".join(dup_idx))
    if bad_idx:
        errors.append(f"{len(bad_idx)} 个文件的 idx 解析不成数字")
    if empty:
        errors.append("这些容器里一张瓦片都没匹配到：" + ", ".join(empty))

    # ---- 方向：容器名里的标记比 serpentine 这个布尔值更具体，优先信它
    groups.sort(key=lambda g: g["index"])
    if has_dir:
        tokens = sorted({str(g["token"]).lower() for g in groups})
        if serp and not rev:
            warnings.append("serpentine=true 但没给 reverse_direction_token，"
                            f"容器名里的方向标记有 {tokens}，全部按正向处理")
        elif rev and rev not in tokens:
            warnings.append(f"reverse_direction_token={rev!r} 没在任何容器名里出现"
                            f"（出现的是 {tokens}），全部按正向处理")
        if len(tokens) > 2:
            warnings.append(f"方向标记超过两种：{tokens}")
        for g in groups:
            g["direction"] = "up" if (rev and str(g["token"]).lower() == rev) else "down"
    else:
        if serp:
            warnings.append("group_regex 没有 direction 命名组，按容器序号奇偶交替方向"
                            "（第一列正向）；拼出来上下颠倒就是这里猜反了")
        for i, g in enumerate(groups):
            g["direction"] = "up" if (serp and i % 2 == 1) else "down"
    # 文件名决定不了哪个标记才是反向（实验 D 的 ds_D：校验通过 3/3，但 2/3 把
    # reverse_direction_token 填反了，校验器对此沉默）。第一个容器被判反向是最
    # 常见的翻转征兆，只提醒、不报错：真的以反向列开头的数据集会多一条提示。
    if groups and groups[0]["direction"] == "up":
        warnings.append("第一个容器（index 最小）被判为反向：reverse_direction_token 可能取反，"
                        f"拼出来上下颠倒就是这里（{groups[0]['name']}，token={groups[0]['token']!r}）")
    if major == "row":
        warnings.append("major_axis=row：容器仍存进 Tile.col、容器内顺序存进 nominal_row，"
                        "配准先验按列扫描假设；拼出来歪了先怀疑这里")

    # ---- 各列张数
    counts = [len(g["entries"]) for g in groups]
    if counts:
        med = float(np.median(counts))
        tol = max(2.0, count_tol * med)
        outliers = [f"{g['name']}: {len(g['entries'])} 张（中位数 {med:.0f}）"
                    for g in groups if abs(len(g["entries"]) - med) > tol]
        report["count_outliers"] = outliers
        if outliers:
            warnings.append("各列张数不一致：" + "; ".join(outliers))

    report["columns"] = [{"name": g["name"], "index": g["index"], "token": g["token"],
                          "direction": g["direction"], "n_tiles": len(g["entries"])}
                         for g in groups]
    report["ok"] = not errors
    return report, groups


def validate_layout(data_dir: str, layout: dict,
                    min_match: float = LAYOUT_MIN_MATCH,
                    count_tol: float = LAYOUT_COUNT_TOL) -> dict:
    """闭式校验一份布局：正则能编译且带命名组；group_regex 匹配 >= min_match 的
    容器、tile_regex 匹配 >= min_match 的文件；序号能解析成数字；各列张数在
    容差内（超了只提醒）。返回 JSON 可序列化的报告，ok=False 时 errors 说明
    原因，unmatched 列出**每一个**没对上的名字。"""
    report, _ = _parse_layout(data_dir, layout, min_match, count_tol)
    return report


def format_layout_report(report: dict, max_names: int = 30) -> str:
    """把校验报告排成给人看的文本。名单超过 max_names 时截断并给出总数，
    完整清单在 JSON 里。"""
    def names(xs):
        xs = list(xs)
        s = "\n".join(f"      {x}" for x in xs[:max_names])
        if len(xs) > max_names:
            s += f"\n      ...还有 {len(xs) - max_names} 个（完整清单在 JSON 报告里）"
        return s

    c, t = report.get("containers", {}), report.get("tiles", {})
    lines = [f"  布局校验：{'通过' if report.get('ok') else '不通过'}"]
    if c.get("total") is not None:
        lines.append(f"    容器 {c.get('matched', 0)}/{c.get('total', 0)} 匹配，"
                     f"瓦片 {t.get('matched', 0)}/{t.get('total', 0)} 匹配"
                     f"（跳过 {len(c.get('ignored', []))} 个目录项、"
                     f"{len(t.get('skipped', []))} 个 __MACOSX/点文件）")
    cols = report.get("columns", [])
    for col in cols[:max_names]:
        lines.append(f"    {col['name']:<28} index={col['index']:<4} "
                     f"{col['direction'] or '-':<5} {col['n_tiles']:>4} 张")
    if len(cols) > max_names:
        lines.append(f"    ...共 {len(cols)} 列")
    for e in report.get("errors", []):
        lines.append(f"    [错误] {e}")
    for w in report.get("warnings", []):
        lines.append(f"    [提醒] {w}")
    if c.get("unmatched"):
        lines.append("    没匹配上的容器：\n" + names(c["unmatched"]))
    if c.get("bad_index"):
        lines.append("    index 解析不了的容器：\n" + names(c["bad_index"]))
    if t.get("unmatched"):
        lines.append("    没匹配上的文件：\n" + names(t["unmatched"]))
    if t.get("bad_idx"):
        lines.append("    idx 解析不了的文件：\n" + names(t["bad_idx"]))
    return "\n".join(lines)


def scan_with_layout(data_dir: str, layout: dict,
                     min_match: float = LAYOUT_MIN_MATCH,
                     count_tol: float = LAYOUT_COUNT_TOL) -> list[Tile]:
    """按推断出的布局清点。先 validate，不通过就抛 LayoutError（消息里列着
    没对上的名字），绝不退回硬编码规则。通过了就按和硬编码路径完全相同的
    约定建 Tile：容器按 index 排序得 col_idx，容器内按 (key, name) 排序得
    order，反向列的 nominal_row 倒过来。"""
    report, groups = _parse_layout(data_dir, layout, min_match, count_tol)
    if not report["ok"]:
        raise LayoutError(format_layout_report(report))
    tiles: list[Tile] = []
    for col_idx, g in enumerate(groups):
        n = len(g["entries"])
        for order, (key, base, inner, size) in enumerate(g["entries"]):
            row = order if g["direction"] == "down" else (n - 1 - order)
            tiles.append(Tile(col=g["index"], col_idx=col_idx, direction=g["direction"],
                              order=order, key=key, name=base, zip_path=g["path"],
                              inner=inner, nominal_row=row, nbytes=size))
    return tiles


# ---------------------------------------------------------------- 缓存
def _read_raw(container: str, inner: str) -> bytes:
    """从容器里取一张原图的字节。容器是 zip 就按 namelist 读；是普通目录
    （按 layout 清点的文件夹数据集）就直接读文件。"""
    if os.path.isdir(container):
        with open(os.path.join(container, inner), "rb") as f:
            return f.read()
    with zipfile.ZipFile(container) as z:
        return z.read(inner)



def cache_path(cache_dir: str, t: Tile) -> str:
    return os.path.join(cache_dir, f"L{t.col:02d}_{t.direction}", f"{t.key:08.2f}.png")


# ---- Full-frame focus metrics: lazy, GUI-free loading. If unavailable,
# report missing scores explicitly; importing this backend never loads hardware.
FOCUS_INDEX = "focus_index.json"      # 缓存目录里的指标索引：重跑不再解原图
_FOCUS_BACKEND = None                 # None = 还没试；False = 不可用；否则 (focus_metrics, cfg)


def _focus_backend():
    global _FOCUS_BACKEND
    if _FOCUS_BACKEND is None:
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            from flakepipeline.focus_metrics import FocusConfig, focus_metrics
        except Exception:
            if here not in sys.path:
                sys.path.insert(0, here)
            try:
                from flakepipeline.focus_metrics import FocusConfig, focus_metrics
            except Exception as e:      # Missing numerical dependencies: report unavailability.
                print(f"  [提醒] 加载不了 flakepipeline.focus_metrics（{type(e).__name__}: {e}），"
                      f"全帧对焦分数为 NaN，defocus/focus_band 标记停用", flush=True)
                _FOCUS_BACKEND = False
                return False
        _FOCUS_BACKEND = (focus_metrics, FocusConfig())
    return _FOCUS_BACKEND


def full_frame_focus(arr: np.ndarray) -> dict:
    """已解码的全帧 -> {"focus_score", "contrast"}，与 focus_metrics.focus_metrics
    的 score / contrast 同数（最长边缩到 FocusConfig.max_side=1024，3840 -> 960）。"""
    be = _focus_backend()
    if not be:
        return {"focus_score": float("nan"), "contrast": float("nan")}
    focus_metrics, cfg = be
    m = focus_metrics(arr, cfg)
    return {"focus_score": float(m["score"]), "contrast": float(m["contrast"])}


def _small_stats(small: np.ndarray) -> dict:
    """1/scale_div 缓存图上的提示量：均值、标准差、Laplacian 方差、饱和占比。"""
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return dict(mean=float(g.mean()), std=float(g.std()),
                focus=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                clip_frac=float(((g >= 250) | (g <= 5)).mean()))


def _decode(args):
    """在子进程里解一张原图，缩到 1/scale_div 存成 PNG，并算好质检量
    （缓存图上的提示量 + 全帧对焦分数，同一趟解码里完成）。"""
    zip_path, inner, out_path, scale_div = args
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw = _read_raw(zip_path, inner)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return out_path, None
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (w // scale_div, h // scale_div), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, small, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    stats = dict(src_w=w, src_h=h, **_small_stats(small), **full_frame_focus(arr))
    return out_path, stats


def _score_full(args):
    """已缓存、但索引里还没有全帧分数的瓦片：只解原图打分，不重写缓存图。"""
    zip_path, inner, out_path = args
    try:
        raw = _read_raw(zip_path, inner)
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:                      # raw data moved / unmounted / bad zip:
        print(f"    [focus] 原图读不到，跳过打分 {inner}: {type(e).__name__}: {e}")
        return out_path, None                   # cached run continues without a score
    if arr is None:
        return out_path, None
    h, w = arr.shape[:2]
    return out_path, dict(src_w=w, src_h=h, **full_frame_focus(arr))


def _index_key(cache_dir: str, out_path: str) -> str:
    return os.path.relpath(out_path, cache_dir).replace(os.sep, "/")


def _load_index(cache_dir: str) -> dict:
    p = os.path.join(cache_dir, FOCUS_INDEX)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_index(cache_dir: str, index: dict):
    p = os.path.join(cache_dir, FOCUS_INDEX)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=0)
    os.replace(tmp, p)


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def build_cache(tiles: list[Tile], cache_dir: str, scale_div: int = 8,
                workers: int | None = None, force: bool = False) -> dict:
    """把每张瓦片解码并缩存成 1/scale_div 的 PNG，顺手填好质检指标。幂等。

    指标存进 cache_dir/focus_index.json：缓存图上的提示量（mean/std/focus/clip_frac）
    和全帧对焦分数（focus_score/contrast）。已缓存但索引里没有全帧分数的瓦片
    （旧缓存）只解原图打分、不重写缓存图；索引齐全的瓦片一张都不解。"""
    os.makedirs(cache_dir, exist_ok=True)
    index = _load_index(cache_dir)
    jobs, rescore, targets = [], [], []
    for t in tiles:
        out = cache_path(cache_dir, t)
        targets.append((t, out))
        if force or not os.path.exists(out):
            jobs.append((t.zip_path, t.inner, out, scale_div))
        elif not _finite((index.get(_index_key(cache_dir, out)) or {}).get("focus_score")):
            rescore.append((t.zip_path, t.inner, out))

    nw = workers or max(1, (os.cpu_count() or 4) - 2)

    def _run(pool_cls, fn, items, label, **kw):
        out_map = {}
        with pool_cls(max_workers=nw) as ex:
            for i, (out, stats) in enumerate(ex.map(fn, items, **kw), 1):
                out_map[out] = stats
                if i % 50 == 0 or i == len(items):
                    print(f"  {label} {i}/{len(items)}", flush=True)
        return out_map

    def _parallel(fn, items, label):
        if not items:
            return {}
        try:
            return _run(ProcessPoolExecutor, fn, items, label, chunksize=4)
        except BrokenProcessPool:
            # macOS 的进程池用 spawn 启动，子进程要重新 import __main__。
            # 在 Jupyter、python -c、或从 stdin 喂脚本的场景下 __main__ 不是
            # 一个真实文件，进程池会直接崩掉。退回线程池：cv2 的解码和缩放
            # 都释放 GIL，所以多线程照样是真并行，只是稍慢一点。
            print("  进程池启动失败（多半是在 Jupyter 或交互式环境里），"
                  "改用线程池", flush=True)
            return _run(ThreadPoolExecutor, fn, items, label)

    done = _parallel(_decode, jobs, "缓存")
    scored = _parallel(_score_full, rescore, "全帧对焦打分（旧缓存补算）")

    for t, out in targets:
        key = _index_key(cache_dir, out)
        if out in done:
            st = done[out]
            if st is None:                  # 原图解不开
                t.flags.append("unreadable")
                index.pop(key, None)
                continue
        else:
            # 已缓存的瓦片：提示量优先取索引，没有就从缓存图补算（比解原图快得多）；
            # 全帧分数来自这次的补算（或索引里已有的）
            st = dict(index.get(key) or {})
            if not all(k in st for k in ("mean", "std", "focus", "clip_frac")):
                small = cv2.imread(out, cv2.IMREAD_COLOR)
                if small is None:
                    t.flags.append("unreadable")
                    continue
                st.update(_small_stats(small))
                st.setdefault("src_w", TILE_W)
                st.setdefault("src_h", TILE_H)
            fs = scored.get(out)
            if fs:
                st.update(fs)
        st.setdefault("focus_score", float("nan"))
        st.setdefault("contrast", float("nan"))
        index[key] = st
        t.mean, t.std = st["mean"], st["std"]
        t.focus, t.clip_frac = st["focus"], st["clip_frac"]
        t.focus_score, t.contrast = float(st["focus_score"]), float(st["contrast"])
    _save_index(cache_dir, index)
    return {"cached": len(jobs), "rescored": len(rescore), "total": len(targets),
            "scale_div": scale_div, "index": os.path.join(cache_dir, FOCUS_INDEX),
            "focus_backend": bool(_focus_backend()) if (jobs or rescore) else None}


def load_small(cache_dir: str, t: Tile) -> np.ndarray | None:
    return cv2.imread(cache_path(cache_dir, t), cv2.IMREAD_COLOR)


def load_full(t: Tile) -> np.ndarray | None:
    raw = _read_raw(t.zip_path, t.inner)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- 质检
def _mad_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-9
    return (v - med) / (1.4826 * mad)


# 对焦门的阈值。数值与 flakepipeline/focus_metrics.FocusConfig 的 2026-09-02 标定
# 一致（tests/test_tile_gate.py 钉住这一点）：Figure 3a 全部 920 张，ref = p75 = 19.8，
# 0.4 x ref = 7.9 恰好抓出中央熔池周围 25 张全糊的（模型 25/25 全票 drop），
# 0.4–0.6 x ref 的 27 张真是混合的（抽 10 张 drop 3 张），得看图。
FOCUS_MIN_CONTRAST = 4.0        # contrast 低于它的帧不进参考池（空帧），也不判虚焦
FOCUS_REF_PERCENTILE = 75.0     # 参考值 = 有纹理瓦片 focus_score 的这个百分位
FOCUS_MIN_REF_TILES = 3         # 有纹理瓦片少于这个数就没有参考值，对焦标记停用
FOCUS_RATIO_DEFOCUS = 0.4       # score < 0.4 x ref -> defocus（闭式丢弃）
FOCUS_RATIO_BAND = 0.6          # 0.4 x ref <= score < 0.6 x ref -> focus_band（送模型）
FOCUS_REF_WARN_BELOW = 10.0     # ref 低于它 -> 整批偏糊/低对比度，相对规则抓不到，警告
FOCUS_FLAGS = ("defocus", "focus_band", "low_texture")
STAT_FLAGS = ("low-contrast", "small-file", "clipped", "irregular-index")


def focus_reference(tiles: list[Tile]) -> tuple[float | None, int]:
    """(参考值, 参考池张数)。参考池 = focus_score 有限且 contrast >= FOCUS_MIN_CONTRAST
    的瓦片；不足 FOCUS_MIN_REF_TILES 张或 p75 <= 0 时没有参考值。"""
    textured = [float(t.focus_score) for t in tiles
                if _finite(t.focus_score) and _finite(t.contrast)
                and float(t.contrast) >= FOCUS_MIN_CONTRAST]
    if len(textured) < FOCUS_MIN_REF_TILES:
        return None, len(textured)
    ref = float(np.percentile(textured, FOCUS_REF_PERCENTILE))
    return (ref if ref > 0 else None), len(textured)


def flag_tiles(tiles: list[Tile]) -> dict:
    """给瓦片打提名标签。可重复调用（每次先清掉自己上次打的标签）。

    对焦：比值规则（见上面的常量），defocus 是闭式丢弃、focus_band 送模型看图；
    contrast < FOCUS_MIN_CONTRAST 的帧几乎是平的（快门空拍、光路被挡），和
    focus_metrics 一样**不判虚焦**，打 low_texture 送模型看是空衬底还是空帧。
    以前那条 1/8 缓存图 Laplacian 方差的 MAD-z < -3 从来没触发过（Figure 3a 上
    最小 z = -1.83），已删。其余统计标签照旧、阈值故意宽松：low-contrast /
    small-file 按 MAD-z，clipped 按饱和占比。这一步只负责 *提名*，最终去留由
    step_qc 的三路门决定，仍然拿不准的才送 Kimi 看图。

    返回值除 flagged / irregular 外还带：focus_ref、focus_ref_n、defocus、
    focus_band、low_texture、warnings（整批偏糊、没有参考值、分数缺失）。"""
    own = set(FOCUS_FLAGS) | set(STAT_FLAGS)
    for t in tiles:
        t.flags = [f for f in t.flags if f not in own]

    warnings: list[str] = []
    ref, n_ref = focus_reference(tiles)
    n_nan = sum(1 for t in tiles if not _finite(t.focus_score))
    if tiles and n_nan:
        warnings.append(f"{n_nan}/{len(tiles)} 张瓦片没有全帧对焦分数（focus_score 为 NaN，"
                        f"缓存没重建或 focus_metrics 加载失败），这些瓦片不判虚焦")
    if tiles and ref is None:
        warnings.append(f"有纹理（contrast >= {FOCUS_MIN_CONTRAST:g}）且有分数的瓦片只有 {n_ref} 张"
                        f"（< {FOCUS_MIN_REF_TILES}），没有对焦参考值，defocus/focus_band 标记停用")
    elif ref is not None and ref < FOCUS_REF_WARN_BELOW:
        # 移植自 focus_metrics.check_focus_folder：相对规则在这两种情况下都抓不到东西
        warnings.append(f"参考 score {ref:.1f} 低于 {FOCUS_REF_WARN_BELOW:g}：整批都偏糊（焦面漂了？）"
                        f"或衬底本身对比度低，相对规则抓不到，抽几张肉眼看")

    std = np.array([t.std for t in tiles])
    nb = np.array([float(t.nbytes) for t in tiles])
    zs, zb = _mad_z(std), _mad_z(nb)

    for t, b, c in zip(tiles, zs, zb):
        textured = _finite(t.contrast) and float(t.contrast) >= FOCUS_MIN_CONTRAST
        if _finite(t.contrast) and not textured:
            t.flags.append("low_texture")
        elif ref is not None and textured and _finite(t.focus_score):
            r = float(t.focus_score) / ref
            if r < FOCUS_RATIO_DEFOCUS:
                t.flags.append("defocus")
            elif r < FOCUS_RATIO_BAND:
                t.flags.append("focus_band")
        if b < -3.0:
            t.flags.append("low-contrast")
        if c < -3.0:
            t.flags.append("small-file")
        if t.clip_frac > 0.35:
            t.flags.append("clipped")

    # 序号不是整数的，只做记号，不做判断。
    #
    # 这里踩过一个大坑，写下来以免重犯：最初的版本把同一整数格位上的多张
    # （L19 的 0035.5 和 0035.6）当成"重复拍摄，二选一"。实测把这个前提
    # 推翻了——它们是**两个不同的相邻格位**，都得留，而且真实物理顺序是
    #     0034 -> 0035.6 -> 0035.5 -> 0036
    # 恰好和文件名排序相反（配准链 NCC 0.66/0.55/0.53，反例对照只有 0.17）。
    #
    # 结论：文件名会撒谎，几何不会。所以重复的判定挪到配准之后，
    # 按解出来的位置是否重合来定（见 find_duplicates），这里只打记号。
    for t in tiles:
        if t.key != int(t.key):
            t.flags.append("irregular-index")

    return {"conflicts": [],           # 保留字段以兼容调用方；真冲突见 find_duplicates
            "irregular": [t for t in tiles if "irregular-index" in t.flags],
            "flagged": [t for t in tiles if t.flags],
            "focus_ref": ref, "focus_ref_n": n_ref,
            "defocus": [t for t in tiles if "defocus" in t.flags],
            "focus_band": [t for t in tiles if "focus_band" in t.flags],
            "low_texture": [t for t in tiles if "low_texture" in t.flags],
            "warnings": warnings}


def find_duplicates(tiles: list[Tile], positions: np.ndarray,
                    frac: float = 0.25) -> list[list[Tile]]:
    """配准之后，按**解出来的位置**找真正重合的瓦片。

    两张瓦片的中心距离小于 frac 个视野时，认为它们拍的是同一处，
    必须二选一。这才是可靠的重复判据：它不关心文件叫什么名字。
    """
    thr_x, thr_y = TILE_W * frac, TILE_H * frac
    groups, used = [], set()
    for i in range(len(tiles)):
        if i in used:
            continue
        g = [i]
        for j in range(i + 1, len(tiles)):
            if j in used:
                continue
            if (abs(positions[i, 0] - positions[j, 0]) < thr_x
                    and abs(positions[i, 1] - positions[j, 1]) < thr_y):
                g.append(j)
        if len(g) > 1:
            used.update(g)
            for k in g:
                tiles[k].flags.append("duplicate")
            groups.append([tiles[k] for k in g])
    return groups


def summarize(tiles: list[Tile]) -> str:
    by_col: dict[int, list[Tile]] = {}
    for t in tiles:
        by_col.setdefault(t.col, []).append(t)
    lines = [f"{'列':>4} {'方向':<5} {'张数':>4} {'序号范围':<16} {'focus中位':>9} {'可疑'}"]
    for col in sorted(by_col):
        ts = by_col[col]
        keys = [t.key for t in ts]
        bad = [t.name for t in ts if t.flags]
        lines.append(f"L{col:<3} {ts[0].direction:<5} {len(ts):>4} "
                     f"{min(keys):.1f}-{max(keys):.1f}".ljust(38)
                     + f"{np.median([t.focus for t in ts]):9.1f}  "
                     + (",".join(bad) if bad else "-"))
    return "\n".join(lines)
