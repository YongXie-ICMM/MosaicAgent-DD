# -*- coding: utf-8 -*-
"""register.py — Figure 3a 马赛克的成对配准与全局位置求解。

这里只解位置，不做融合。对外返回的位移一律换算成 **全分辨率像素**，
内部计算在 1/scale_div 缩略图上做（6.5 Gpx 的画布不可能整幅进内存）。

数据实测出来的坑，按踩到的顺序记在这里：

1. 这批数据上纯相位相关是不能用的。10% 重叠 + 高度自相似的 CVD 畴形貌，
   cv2.phaseCorrelate 的峰稳定地落在 (0, 0) 附近（实测 L20 连续四对全是
   |peak| < 4 px），真峰在 (-9, 237) 处 NCC 高达 0.96 却根本不在峰上。
   所以相位相关在这里只当"廉价的第二意见"，真正干活的是以先验为中心的
   窗口化归一化互相关搜索（cv2.matchTemplate + TM_CCOEFF_NORMED，
   一次 FFT 顶几千次点积）。

2. 相位相关的解永远是模 (W, H) 的：峰 (px, py) 同时也可能是 (px±W, py±H)。
   所有同余解释都要展开，逐个取 **真实重叠矩形** 算 NCC 再裁决
   （MIST / ASHLAR 的标准做法）。实测这个判据很干净：真解 0.96，
   零位移解 0.017。但"重叠不足"必须按 **厚度** 卡而不是按面积卡：
   实测互不相邻的瓦片对，重叠只剩 6 列（面积 1620，比 1.2% 的面积下限
   还大）时有 5.2% 能刷出 NCC >= 0.25，而重叠厚到 27 px 以上时是 0。
   详见 _min_overlap_px。

3. 显微图低频（照明不均）能量远大于形貌纹理，不做高通相关会被背景带跑。
   预处理 = 灰度 -> 减大核高斯 -> 零均值单位方差。相位相关另加 Hann 窗，
   NCC 不加窗（加了窗 NCC 就不再是"重叠区有多像"）。

4. 位移台可重复，全图 3600 多条边的真实位移几乎是同一个常数。所以两阶段：
   先在少量样本对上以标称值为中心宽搜，取 MAD 稳健中位数当全局先验；
   再让所有边在先验附近 tol 内搜。空片、低纹理区全靠先验兜住。

5. 物理行不能信文件名，必须实测。L19 的 0035.5 / 0035.6 不是"同一格位补拍
   两张"——实测 0034 -> 0035.6 差 -1 行（NCC 0.92）、0035.5 -> 0035.6
   差 +1 行（NCC 0.83）、0036 -> 0035.5 差 +1 行（NCC 0.78），也就是
   拍完 0034 后台子一次走了两步拍下 0035.5，发现漏了又退回一步补 0035.6，
   再跳两步接着拍 0036。两张都是有效的、不同的物理行，一张都不能删，
   而 tiles.nominal_row 给这两张的顺序恰好是反的。
   所以这里的做法是：列内逐链测 "这一步走了几行"（假设 k ∈ {+1, 0, -1}，
   实测 NCC 裁决），链上测不动的地方（跨了 2 行，压根没有重叠）就把列
   断成若干 **段**，每段的行偏移单独跟左邻列投票决定。这样漏拍、补拍、
   台子打滑全都由数据自己说了算。

6. 标称值（143 步 / 76 步 / 25.6 px每步）只当搜索中心，绝不写死。
   脚本里 FOV_STEPS_X = 151.04 和 3840/25.6 = 150.0 本来就不自洽，
   横向重叠到底多少必须由数据说了算 —— 就是 stats['measured_overlap_x_px']。
   实测结果是两个方向自洽地指向"每步 25.09 px、台面相对相机转了 2.3 度"
   （v 和 h 两个位移向量夹角 89.8 度，模长比标称都是 0.980）。

7. 图千万不能断。块与块之间一条边都没有时，最小二乘对它们的相对位置
   无话可说，而 residual_rms 照样是 0 —— 几块精确重叠、分数还很漂亮，
   是本流程最难发现的错误。所以 build_edges 保证相邻瓦片一定有边
   （测不准的换成先验、权重压到 1e-3），solve_positions 在真断了又没给
   anchors 时会告警并把各块摊开。判活儿好坏一律看 diag['disconnected']，
   不能只看 residual_rms。
"""
from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

try:
    from tiles import Tile, load_small, TILE_W, TILE_H
except ImportError:                                     # 被当成包导入时
    from .tiles import Tile, load_small, TILE_W, TILE_H


# ---------------------------------------------------------------- 标称几何
# 采集脚本 02Auto_Snake_Scan.py 命令位移台走的步数（这两个是确定的）
DY_STEPS = 76           # 列内步距
DX_STEPS = 143          # 列间步距

# 位移台标定。脚本里写的是 25.6 px/full-step，但那是错的：
# 两次完全独立的测量（本模块第一阶段的自由探测、以及另一路用全域零均值
# NCC 做的审计）分别得到 25.06 和 25.08，两个轴之间也自洽到 0.1%。
# 看起来是 25.06 被误写成了 25.6。
STEP_PX = 25.06

# 位移台 Y 轴与相机轴之间有约 2.24 度转角：每走一个纵向步，图像还会横向
# 平移 74.6 px；每走一个横向步，纵向平移 153 px。一列 53 步累积 3954 px，
# 比一整张瓦片还宽（3840），所以整幅图是平行四边形而不是矩形。
# 这两项必须进先验，否则搜索窗根本够不到真峰。
SHEAR_X_PER_ROW = -74.6
SHEAR_Y_PER_COL = 153.4
# (dx_v, dy_v, dx_h, dy_h)，全分辨率像素
#   v = "物理向下一行"的位移向量（dy 恒为正）
#   h = "col_idx 加一列"的位移向量
NOMINAL_FULL = (SHEAR_X_PER_ROW, DY_STEPS * STEP_PX,
                DX_STEPS * STEP_PX, SHEAR_Y_PER_COL)


@dataclass
class Edge:
    i: int          # tiles 列表下标
    j: int
    dx: float       # j 相对 i 的位移，全分辨率像素
    dy: float
    response: float # 0..1，越大越可信
    kind: str       # 'v' 列内 | 'h' 列间


def _log(on, *a):
    if on:
        print(*a, flush=True)


# ---------------------------------------------------------------- 预处理
_HANN: dict = {}


def _hann(shape):
    if shape not in _HANN:
        _HANN[shape] = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    return _HANN[shape]


def _prep(img, hp_sigma: float = 0.0):
    """灰度 -> 高通 -> 零均值单位方差。全灰空片会得到全零，
    后面 NCC 的分母保护会把它判成 -1，不会以 nan 的形式传染出去。"""
    if img is None:
        return None
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    h, w = g.shape
    # 核取短边的 1/24：比单个晶畴大、比照明不均的尺度小，只砍背景不砍信号
    sigma = hp_sigma or max(2.0, min(h, w) / 24.0)
    hp = g - cv2.GaussianBlur(g, (0, 0), sigma)
    hp -= float(hp.mean())
    s = float(hp.std())
    if s < 1e-5:
        return np.zeros_like(hp)
    return np.ascontiguousarray(hp / s)


# ---------------------------------------------------------------- NCC 判据
def _min_overlap_px(H, W) -> float:
    """重叠矩形在两个方向上各自的最小厚度（缩略图像素）。

    光有面积下限是不够的 —— 实测：拿本数据里 *互不相邻* 的瓦片对去算 NCC，
    重叠只剩 6 列（面积 1620，比 1.2% 的面积下限还大）时，5.2% 的随机对能
    刷出 NCC >= 0.25，p99 高到 0.34；重叠厚到 27 px 以上时这个比例是 0。
    虚高只跟"重叠有多薄"有关，跟面积关系不大（4 行 x 480 的面积 1920 比
    6 列 x 270 的 1620 还大，却一样虚）。所以真正该卡的是厚度。

    取 3% 的长边 = 14.4 px：本数据实测重叠 32 px，题面给的最坏情况横向
    179 px 全分辨率 = 22.4 缩略图像素，都还留着 1.5 倍以上的余量，不会把
    真解误杀；同时把 4-12 px 那一段虚高最严重的候选全部挡在外面。
    """
    return max(8.0, 0.03 * max(H, W))


def _ncc(A, B, dx, dy, min_area, min_lin=0.0) -> float:
    """按 b 相对 a 的整数位移取 **真实重叠矩形** 算归一化互相关。
    重叠不够的解一律判死 —— 否则总有一个同余解能靠"只重叠十几个像素"
    刷出高分，这正是必须用真实重叠矩形而不是固定模板的原因。
    面积和厚度两个下限都要卡，理由见 _min_overlap_px。"""
    H, W = A.shape
    dx, dy = int(dx), int(dy)
    x0, x1 = max(0, dx), min(W, W + dx)
    y0, y1 = max(0, dy), min(H, H + dy)
    if x1 - x0 < min_lin or y1 - y0 < min_lin:
        return -1.0
    if x1 - x0 <= 1 or y1 - y0 <= 1 or (x1 - x0) * (y1 - y0) < min_area:
        return -1.0
    pa = A[y0:y1, x0:x1].ravel()
    pb = B[y0 - dy:y1 - dy, x0 - dx:x1 - dx].ravel()
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    na = float(pa @ pa)
    nb = float(pb @ pb)
    if na < 1e-8 or nb < 1e-8:
        return -1.0
    return float((pa @ pb) / math.sqrt(na * nb))


def _parab(fm, f0, fp) -> float:
    """三点抛物线顶点；越界就放弃亚像素（弱纹理处相关面根本不是抛物线）。"""
    den = fm - 2.0 * f0 + fp
    if abs(den) < 1e-9:
        return 0.0
    d = 0.5 * (fm - fp) / den
    return d if -1.0 <= d <= 1.0 else 0.0


def _win_search(A, B, cx, cy, rx, ry, min_strip=8):
    """在位移盒 (cx±rx, cy±ry) 内做归一化互相关搜索，返回 (dx, dy, 峰值)。

    从 B 里裁一块"在盒内任何位移下都仍落在 A 内"的模板，再让它在 A 的
    对应窗口里滑 —— 这样整个盒子里模板面积恒定，峰值可比；用
    matchTemplate 走 FFT，比逐格点积快两个量级。盒子会先被夹到"至少还
    留得下 min_strip 宽的重叠带"，否则模板会退化成一条线。
    """
    H, W = A.shape

    def _rng(c, r, n):
        lim = float(n - min_strip)
        lo = max(c - r, -lim)
        hi = min(c + r, lim)
        return int(math.floor(lo)), int(math.ceil(hi))

    dx0, dx1 = _rng(cx, rx, W)
    dy0, dy1 = _rng(cy, ry, H)
    if dx1 < dx0 or dy1 < dy0:
        return None
    bx0, bx1 = max(0, -dx0), min(W, W - dx1)
    by0, by1 = max(0, -dy0), min(H, H - dy1)
    w, h = bx1 - bx0, by1 - by0
    if w < min_strip or h < min_strip:
        return None
    tpl = np.ascontiguousarray(B[by0:by1, bx0:bx1])
    ax0, ay0 = bx0 + dx0, by0 + dy0
    win = np.ascontiguousarray(A[ay0:ay0 + h + (dy1 - dy0), ax0:ax0 + w + (dx1 - dx0)])
    if win.shape[0] < h or win.shape[1] < w:
        return None
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    m, n = np.unravel_index(int(np.argmax(res)), res.shape)
    v = float(res[m, n])
    sx = _parab(float(res[m, n - 1]), v, float(res[m, n + 1])) if 0 < n < res.shape[1] - 1 else 0.0
    sy = _parab(float(res[m - 1, n]), v, float(res[m + 1, n])) if 0 < m < res.shape[0] - 1 else 0.0
    return dx0 + n + sx, dy0 + m + sy, v


def _pc_peak(A, B):
    """相位相关峰，即 b 相对 a 的画布位移。注意方向：若 src2 = src1 平移 +d，
    cv2.phaseCorrelate(src1, src2) 给 +d，而"b 在 a 的右下方"对应的内容
    平移是 -d，所以要反着调用（已实测验证）。"""
    win = _hann(A.shape)
    (px, py), _ = cv2.phaseCorrelate(np.ascontiguousarray(B * win),
                                     np.ascontiguousarray(A * win))
    return px, py


def _congruent(v, period, hi, expected=None, tol=None):
    """把峰展开成模 period 的同余候选，砍掉重叠必然不足的那些。
    tol 传 None 表示"有先验但不设容差"，此时不做距离筛选 ——
    以前这里会拿 float 跟 None 比大小直接抛 TypeError。"""
    out = []
    for k in (-1, 0, 1):
        c = v + k * period
        if abs(c) > hi:
            continue
        if expected is not None and tol is not None and abs(c - expected) > tol:
            continue
        out.append(c)
    return out


def _measure_prepped(A, B, expected_dx, expected_dy, tol,
                     min_overlap_frac=0.012, rad=3):
    """measure_pair 的内核，输入已经是 _prep 过的图。"""
    has_prior = expected_dx is not None and expected_dy is not None
    ex = float(expected_dx) if has_prior else 0.0
    ey = float(expected_dy) if has_prior else 0.0
    if A is None or B is None:
        return ex, ey, 0.0
    H, W = A.shape
    min_area = max(64.0, min_overlap_frac * W * H)
    min_lin = _min_overlap_px(H, W)
    t = float(tol) if (has_prior and tol is not None) else None

    px, py = _pc_peak(A, B)
    # 同余上限直接卡在厚度下限上：连 _ncc 都过不了的候选没必要生成
    cxs = _congruent(px, W, W - min_lin, ex if has_prior else None, t)
    cys = _congruent(py, H, H - min_lin, ey if has_prior else None, t)
    cands = [(cx, cy) for cx in cxs for cy in cys] or [(px, py)]
    # 同余候选先用真实重叠矩形的 NCC 粗排，只精修最像的几个
    cands.sort(key=lambda c: -_ncc(A, B, round(c[0]), round(c[1]), min_area, min_lin))

    trials = []
    for cx, cy in cands[:3]:
        r = _win_search(A, B, round(cx), round(cy), rad, rad)
        if r:
            trials.append(r)
    if has_prior:
        # 主力：以先验为中心、覆盖整个 tol 盒的窗口搜索。
        # tol 大的时候模板会被夹得很薄，所以命中后再用小半径复搜一次，
        # 让模板厚起来把亚像素坐稳。
        rr = max(int(math.ceil(t or rad)), rad)
        r = _win_search(A, B, ex, ey, rr, rr)
        if r:
            trials.append(r)
            if rr > 4:
                r2 = _win_search(A, B, round(r[0]), round(r[1]), rad, rad)
                if r2:
                    trials.append(r2)
    if not trials:
        return (ex if has_prior else px), (ey if has_prior else py), 0.0

    # 最终裁决一律回到"真实重叠矩形上的 NCC"，保证各同余解释可比
    scored = [(_ncc(A, B, round(x), round(y), min_area, min_lin), x, y)
              for x, y, _ in trials]
    v, bx, by = max(scored, key=lambda s: s[0])
    return float(bx), float(by), max(0.0, float(v))


def measure_pair(a, b, expected_dx, expected_dy, tol) -> tuple[float, float, float]:
    """对两张 BGR uint8 小图做相位相关，返回 (dx, dy, response)，
    单位是传入图像自身的像素。expected_* 也是该尺度下的像素。
    expected_* 传 None 表示不加先验、在全部同余候选里自由裁决。"""
    return _measure_prepped(_prep(a), _prep(b), expected_dx, expected_dy, tol)


def _measure_multi(A, B, hyps, tol, min_overlap_frac=0.012):
    """多假设版本：hyps 是若干个 (ex, ey)，返回 (dx, dy, ncc, 胜出假设下标)。
    列内"这一步走了几行"、列间"标称方向是正是负"都靠它裁决。"""
    best = (0.0, 0.0, -1.0, -1)
    for k, (ex, ey) in enumerate(hyps):
        dx, dy, v = _measure_prepped(A, B, ex, ey, tol, min_overlap_frac)
        if v > best[2]:
            best = (dx, dy, v, k)
    return best


# ---------------------------------------------------------------- 稳健统计
def _robust(vals, weights=None, z=3.0):
    """MAD 剔离群后的加权均值；样本少时退化成中位数。"""
    v = np.asarray(vals, float)
    if v.size == 0:
        return float("nan"), 0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad < 1e-6 or v.size < 4:
        return med, int(v.size)
    keep = np.abs(v - med) <= z * 1.4826 * mad
    if keep.sum() == 0:
        return med, 0
    if weights is None:
        return float(v[keep].mean()), int(keep.sum())
    w = np.asarray(weights, float)[keep]
    if w.sum() <= 0:
        return float(v[keep].mean()), int(keep.sum())
    return float((v[keep] * w).sum() / w.sum()), int(keep.sum())


def decide_segment_offset(acc, min_response=0.25, single_edge_ncc=0.5, offset_margin=0.1):
    """Row offset of one segment from its horizontal votes: acc maps offset -> NCC values.

    Returns (offset, n_good, mean_ncc). Offsets are ranked by the number of edges at or
    above min_response, then by mean NCC. Two guards keep the stage's own numbering unless
    the evidence is real: a single edge only counts when its NCC reaches single_edge_ncc
    (at the chip edge or on blank tiles the lateral correlations are noise, and one of the
    seven candidate offsets always happens to pass 0.25), and a non-zero offset must beat
    the chain hypothesis (offset 0) by offset_margin in mean NCC unless it has more good
    edges. n_good == 0 tells the caller to inherit the neighbouring segment's offset.
    """
    rank = sorted(acc.items(),
                  key=lambda kv: (-sum(1 for x in kv[1] if x >= min_response),
                                  -float(np.mean(kv[1]))))
    best = rank[0][0] if rank else 0
    nice = sum(1 for x in acc.get(best, []) if x >= min_response)
    mean_best = float(np.mean(acc[best])) if best in acc else 0.0
    if nice == 1 and mean_best < single_edge_ncc:
        nice = 0
    if best != 0 and 0 in acc:
        nice0 = sum(1 for x in acc[0] if x >= min_response)
        mean0 = float(np.mean(acc[0]))
        if nice0 >= nice and mean_best - mean0 < offset_margin:
            best, nice, mean_best = 0, nice0, mean0
    return best, nice, mean_best


def _spread(n, k):
    """在 [0, n) 里取 k 个尽量分散的下标，两端各留一点（列首列尾常有暗角）。

    这里必须用 floor 而不是 round：round 在 k 接近 n 时会把相邻两个格心
    撞到同一个下标上（n=2,k=2 时 round 给 {0,0} —— 只剩一个探针）。
    偏偏最需要探针的就是断链切出来的两三张的小段，样本本来就少，
    再被撞掉一半就投不出票了。floor 在 k <= n 时保证 k 个下标两两不同。
    """
    if n <= 0:
        return []
    k = max(1, min(k, n))
    return sorted({min(n - 1, int((i + 0.5) * n / k)) for i in range(k)})


# ---------------------------------------------------------------- 主流程
def build_edges(tiles, cache_dir, scale_div=8, workers=None,
                prior=None, verbose=True, *,
                nominal=None, tol_full=None, coarse_tol_full=None,
                min_response=0.25, n_probe_v=6, n_probe_h=7,
                offset_range=3, single_edge_ncc=0.5, offset_margin=0.1,
                row_policy="infer") -> tuple[list[Edge], dict]:
    """建立列内和列间的所有邻接边。返回 (edges, stats)。

    prior / nominal 都是全分辨率像素的 (dx_v, dy_v, dx_h, dy_h)：
    v = "物理向下一行"的位移向量（约定 dy_v > 0），h = "向右一列"的位移向量。
    prior 给了就跳过第一阶段；nominal 只当第一阶段的搜索中心。

    被判不可信的边不会被丢掉，而是换成先验位移、权重压到 1e-3。宁可让全局
    解在那里靠先验填空，也不能把图割断 —— 断了就得靠名义网格硬拼分量。

    段的行偏移投票（见下）要有把握才改链上假设：只有一条够格边时它的 NCC 必须
    达到 single_edge_ncc；最佳偏移比"不偏移"的证据没多出 offset_margin 时沿用
    不偏移。2026-10-07 的 18×50 扫描里，芯片边缘上两张只剩一半衬底的瓦片各自
    断成单元素段，被一条 NCC 0.27 的边投到了 -2/-3 行，造成空洞和错位重影；
    其余同类瓦片都是 n_good=0 而被正确继承。

    row_policy="infer" preserves the historical serpentine row reconstruction.
    "recorded-grid" uses integer, unique per-column nominal_row identities from a
    physical-coordinate acquisition. Correlations refine pixel displacements but
    cannot reverse, duplicate or compact those recorded rows.
    """
    tiles = list(tiles)
    n = len(tiles)
    if row_policy not in ("infer", "recorded-grid"):
        raise ValueError(f"Unknown registration row_policy: {row_policy!r}")
    recorded_grid = row_policy == "recorded-grid"
    if recorded_grid:
        identities = set()
        for t in tiles:
            row = t.nominal_row
            if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
                raise ValueError(f"recorded-grid requires integer nominal_row: {t.tid}={row!r}")
            identity = (t.col, int(row))
            if identity in identities:
                raise ValueError(f"Duplicate recorded-grid row in column {t.col}: {row}")
            identities.add(identity)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    sd = float(scale_div)
    nom = np.array(nominal if nominal is not None else NOMINAL_FULL, float) / sd

    # ---- 预处理全部缩略图并常驻内存。480x270 float32 只有 0.5 MB/张，
    #      920 张约 480 MB，比每条边重复解 PNG + 高斯模糊便宜太多。
    _log(verbose, f"[register] 预处理 {n} 张缩略图 (workers={workers})")
    P: list = [None] * n

    def _load(k):
        P[k] = _prep(load_small(cache_dir, tiles[k]))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_load, range(n)))
    missing = [k for k in range(n) if P[k] is None]
    shape = next((p.shape for p in P if p is not None), None)
    if shape is None:
        raise RuntimeError("缓存里一张缩略图都读不到，先跑 tiles.build_cache")
    Hs, Ws = shape

    def _recorded_measure(a, b, vector, search_tol):
        delta = int(tiles[b].nominal_row) - int(tiles[a].nominal_row)
        expected = (delta * vector[0], delta * vector[1])
        if abs(expected[0]) >= Ws or abs(expected[1]) >= Hs:
            # Missing rows can span beyond the field of view. Retain their true
            # spacing with a weak prior edge instead of searching an alias peak.
            return expected[0], expected[1], 0.0, -1
        result = _measure_multi(P[a], P[b], [expected], search_tol)
        if math.hypot(result[0] - expected[0], result[1] - expected[1]) > search_tol * 1.5:
            # The phase-correlation helper can return an unconstrained peak when
            # its congruent candidates miss the search window. Such a peak must
            # not become a recorded-grid prior or a reverse/zero-step identity.
            return expected[0], expected[1], 0.0, -1
        return result

    # 搜索半径按缩略图自身尺寸定，不能按 TILE_H/scale_div 推 ——
    # 合成测试和真数据的 scale_div 不一样，写死一定有一边太松或太紧。
    # 0.075*H 是在真数据上扫出来的：这批数据列间 dy 的实际跨度有 ±0.06 行
    # （台面倾斜 + 各列起点不齐），tol 定到 108 px 会把边界上的边削掉，
    # 放到 220 px 又开始出现锁错行的假峰（p99 残差从 36 px 跳到 153 px）。
    tol = (tol_full / sd) if tol_full else max(8.0, 0.075 * Hs)
    coarse_tol = (coarse_tol_full / sd) if coarse_tol_full else max(3.5 * tol, 32.0)
    _log(verbose, f"[register] 缩略图 {Ws}x{Hs}，缺失 {len(missing)} 张，"
                  f"tol={tol:.1f} coarse={coarse_tol:.1f} (缩略图 px)")

    def _pool(jobs, fn):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, jobs))

    # ---- 列结构：只按采集顺序排链，不做任何"文件名 = 物理行"的假设
    by_col: dict[int, list[int]] = {}
    for k, t in enumerate(tiles):
        by_col.setdefault(t.col, []).append(k)
    cols = []
    for col in sorted(by_col):
        seq = sorted(by_col[col], key=lambda k: tiles[k].key)
        cols.append({"col": col, "col_idx": tiles[seq[0]].col_idx,
                     "direction": tiles[seq[0]].direction, "seq": seq})
    cols.sort(key=lambda c: c["col_idx"])

    # ================================================== 第一阶段：全局先验
    stage1 = {}
    if prior is None:
        # 纵向：每列抽 n_probe_v 条链接，正负标称各试一次，真实重叠 NCC 裁决。
        # 不做完全自由搜索 —— 10% 重叠时零位移解重叠面积最大，背景一平就赢，
        # 这是这类流程最经典的翻车方式。
        vp = []
        for c in cols:
            s = c["seq"]
            for r in _spread(len(s) - 1, n_probe_v):
                vp.append((s[r], s[r + 1]))

        def _mv(p):
            if recorded_grid:
                r = _recorded_measure(p[0], p[1], nom[:2], coarse_tol)
                delta = abs(int(tiles[p[1]].nominal_row) - int(tiles[p[0]].nominal_row))
                return r[0] / delta, r[1] / delta, r[2], r[3]
            return _measure_multi(P[p[0]], P[p[1]],
                                  [(nom[0], nom[1]), (-nom[0], -nom[1])], coarse_tol)

        rv = _pool(vp, _mv)
        gv = [r for r in rv if r[2] >= min_response]
        if not gv:
            raise RuntimeError("第一阶段纵向先验全部失败：缓存或标称步距不对")
        # 约定 v 的 dy 恒为正（画布向下）。每条链接自己的正负由 NCC 定，
        # 这样 up / down 列不需要区别对待，蛇形也就不再是特例。
        sv = [(g[0] * math.copysign(1, g[1]), abs(g[1]), g[2]) for g in gv]
        dxv, nv = _robust([s[0] for s in sv], [s[2] for s in sv])
        dyv, _ = _robust([s[1] for s in sv], [s[2] for s in sv])
        stage1["v"] = {"n_probe": len(vp), "n_good": len(gv), "n_used": nv}

        # 横向：行对应关系还没解出来，先拿 tiles.nominal_row 当初值，
        # 在 ±2 行里试，用 NCC 挑出真正相邻的那些对再取稳健中位数。
        hp = []
        for ca, cb in zip(cols[:-1], cols[1:]):
            rb: dict[int, list[int]] = {}
            for k in cb["seq"]:
                rb.setdefault(tiles[k].nominal_row, []).append(k)
            for r in _spread(len(ca["seq"]), 4):
                i = ca["seq"][r]
                for off in ((0,) if recorded_grid else (-2, -1, 0, 1, 2)):
                    for j in rb.get(tiles[i].nominal_row + off, []):
                        hp.append((i, j))

        def _mh(p):
            return _measure_multi(P[p[0]], P[p[1]],
                                  [(nom[2], nom[3]), (-nom[2], -nom[3])], coarse_tol)

        rh = _pool(hp, _mh)
        gh = [r for r in rh if r[2] >= min_response]
        if not gh:
            raise RuntimeError("第一阶段横向先验全部失败")
        sgn = 1.0 if sum(1 for g in gh if g[0] > 0) >= len(gh) / 2 else -1.0
        sh = [g for g in gh if math.copysign(1, g[0]) == sgn]
        dxh, nh = _robust([g[0] for g in sh], [g[2] for g in sh])
        dyh, _ = _robust([g[1] for g in sh], [g[2] for g in sh])
        stage1["h"] = {"n_probe": len(hp), "n_good": len(gh), "n_used": nh, "sign": sgn}
        pri = np.array([dxv, dyv, dxh, dyh], float)
    else:
        pri = np.array(prior, float) / sd

    v_vec = (float(pri[0]), float(pri[1]))
    h_vec = (float(pri[2]), float(pri[3]))
    _log(verbose, f"[register] 先验(全分辨率) v=({pri[0]*sd:+.1f},{pri[1]*sd:+.1f}) "
                  f"h=({pri[2]*sd:+.1f},{pri[3]*sd:+.1f})")

    # ================================================== 列内链：一步走几行
    # 每条链接试 k ∈ {+1, -1, 0} 行。k=0 是"同一格位补拍"，k=±1 是正常相邻；
    # 跨了 2 行的链接根本没有重叠，测不出来，就在那里把列断成两段。
    links = []
    for ci, c in enumerate(cols):
        s = c["seq"]
        for a, b in zip(s[:-1], s[1:]):
            links.append((ci, a, b))

    def _ml(job):
        _, a, b = job
        if recorded_grid:
            return _recorded_measure(a, b, v_vec, tol)
        return _measure_multi(P[a], P[b],
                              [(v_vec[0], v_vec[1]), (-v_vec[0], -v_vec[1]), (0.0, 0.0)],
                              tol)

    _log(verbose, f"[register] 列内链 {len(links)} 条")
    rl = _pool(links, _ml)
    STEP_OF = (1, -1, 0)

    link_info: dict[tuple[int, int], dict] = {}
    for (ci, a, b), (dx, dy, v, kk) in zip(links, rl):
        ok = v >= min_response and kk >= 0
        step = ((int(tiles[b].nominal_row) - int(tiles[a].nominal_row))
                if recorded_grid else STEP_OF[kk]) if ok else None
        link_info[(a, b)] = {"ci": ci, "dx": dx, "dy": dy, "ncc": v,
                             "step": step if ok else None, "ok": ok}

    # 链积分 -> 列内相对行号 + 段划分
    for c in cols:
        s = c["seq"]
        default = 1 if c["direction"] == "down" else -1
        rows, seg = {s[0]: 0}, {s[0]: 0}
        cur, sid = 0, 0
        for a, b in zip(s[:-1], s[1:]):
            li = link_info[(a, b)]
            if li["step"] is None:
                cur += default          # 断链处先按标称走一格，稍后由横向证据纠正
                sid += 1
            else:
                cur += li["step"]
            rows[b], seg[b] = cur, sid
        lo = min(rows.values())
        c["row_prov"] = ({k: int(tiles[k].nominal_row) for k in s} if recorded_grid
                         else {k: rows[k] - lo for k in s})
        c["seg"] = seg
        c["n_seg"] = sid + 1

    nbreak = sum(c["n_seg"] - 1 for c in cols)
    _log(verbose, f"[register] 断链 {nbreak} 处，列被切成 "
                  f"{[c['n_seg'] for c in cols]} 段")

    # ================================================== 段的行偏移
    # 逐列从左往右推：每一段单独在 [-offset_range, +offset_range] 里搜，
    # 用它跟左邻列的横向 NCC 投票。段级而不是列级，才能纠正断链处那一格。
    cols[0]["row_abs"] = {k: cols[0]["row_prov"][k] for k in cols[0]["seq"]}
    cols[0]["seg_offset"] = {s: 0 for s in range(cols[0]["n_seg"])}
    seg_report = []
    for ca, cb in zip(cols[:-1], cols[1:]):
        ref: dict[int, list[int]] = {}
        for k in ca["seq"]:
            ref.setdefault(ca["row_abs"][k], []).append(k)
        members: dict[int, list[int]] = {}
        for k in cb["seq"]:
            members.setdefault(cb["seg"][k], []).append(k)

        cb["row_abs"], cb["seg_offset"] = {}, {}
        if recorded_grid:
            cb["row_abs"] = dict(cb["row_prov"])
            cb["seg_offset"] = {sid: 0 for sid in members}
            for sid, mem in sorted(members.items()):
                seg_report.append({"col": cb["col"], "seg": sid, "size": len(mem),
                                   "offset": 0, "how": "recorded-grid", "n_good": 0,
                                   "ncc_mean": None,
                                   "tids": [tiles[k].tid for k in mem] if len(mem) <= 4 else None})
            continue
        votes = {}
        for sid, mem in sorted(members.items()):
            probe = [mem[t] for t in _spread(len(mem), n_probe_h)]
            jobs = []
            for o in range(-offset_range, offset_range + 1):
                for k in probe:
                    for r in ref.get(cb["row_prov"][k] + o, []):
                        jobs.append((o, r, k))
            out = _pool(jobs, lambda jb: _measure_prepped(P[jb[1]], P[jb[2]],
                                                          h_vec[0], h_vec[1], tol))
            acc: dict[int, list[float]] = {}
            for (o, _, _), r in zip(jobs, out):
                acc.setdefault(o, []).append(r[2])
            # 先比"够格的边有几条"再比均值：某些 offset 只剩一两行可比，
            # 光看均值会被少数高分带偏
            votes[sid] = decide_segment_offset(acc, min_response, single_edge_ncc, offset_margin)

        # 一张糊片自己就会断成一个单元素段，它跟左邻列的相关全是噪声，
        # 这时候投票没有意义，硬取 argmax 就会把它挪错一行。没有任何一条
        # 够格边的段一律沿用链上假设（继承相邻段的偏移），别自作主张。
        order = sorted(votes)
        anchor = next((s for s in order if votes[s][1] > 0), None)
        for sid in order:
            best, nice, mean = votes[sid]
            if nice > 0:
                off, how = best, "voted"
            elif anchor is None:
                off, how = 0, "no-evidence"
            else:
                near = min((s for s in order if votes[s][1] > 0), key=lambda s: abs(s - sid))
                off, how = votes[near][0], "inherited"
            cb["seg_offset"][sid] = off
            for k in members[sid]:
                cb["row_abs"][k] = cb["row_prov"][k] + off
            seg_report.append({"col": cb["col"], "seg": sid, "size": len(members[sid]),
                               "offset": off, "how": how, "n_good": nice, "ncc_mean": mean,
                               "tids": [tiles[k].tid for k in members[sid]]
                               if len(members[sid]) <= 4 else None})
        _log(verbose, f"[register] L{ca['col']}->L{cb['col']} 段偏移 "
                      f"{ {s: cb['seg_offset'][s] for s in sorted(cb['seg_offset'])} }")

    # ================================================== 第二阶段：全部边
    edges: list[Edge] = []
    rejected = []

    def _emit(i, j, kind, dx, dy, v, ex_, ey_, why=None):
        if P[i] is None or P[j] is None:
            why = why or "missing"
        elif v < min_response:
            why = why or "low-ncc"
        elif math.hypot(dx - ex_, dy - ey_) > tol * 1.5:
            why = why or "out-of-tol"
        if why:
            rejected.append({"i": i, "j": j, "kind": kind, "why": why, "ncc": float(v),
                             "tid_i": tiles[i].tid, "tid_j": tiles[j].tid,
                             "dev_px": float(math.hypot(dx - ex_, dy - ey_) * sd)})
            edges.append(Edge(i, j, float(ex_ * sd), float(ey_ * sd), 1e-3, kind))
        else:
            edges.append(Edge(i, j, float(dx * sd), float(dy * sd),
                              float(min(1.0, v)), kind))

    # 列内：链接结果直接复用；断链的用"两端解出的行差"当先验，权重压到底
    for c in cols:
        for a, b in zip(c["seq"][:-1], c["seq"][1:]):
            li = link_info[(a, b)]
            if li["step"] is None:
                k = c["row_abs"][b] - c["row_abs"][a]
                _emit(a, b, "v", li["dx"], li["dy"], li["ncc"],
                      k * v_vec[0], k * v_vec[1], why="broken-link")
            else:
                k = li["step"]
                _emit(a, b, "v", li["dx"], li["dy"], li["ncc"],
                      k * v_vec[0], k * v_vec[1])

    # 列间：同一绝对行号的配对
    hjobs = []
    bridges = []
    for ca, cb in zip(cols[:-1], cols[1:]):
        ra: dict[int, list[int]] = {}
        for k in ca["seq"]:
            ra.setdefault(ca["row_abs"][k], []).append(k)
        rb: dict[int, list[int]] = {}
        for k in cb["seq"]:
            rb.setdefault(cb["row_abs"][k], []).append(k)
        shared = sorted(set(ra) & set(rb))
        for r in shared:
            for i in ra[r]:
                for j in rb[r]:
                    hjobs.append((i, j))
        if not shared:
            # 两列一行都对不上（整列漏拍、或行号解崩了）。这里不补一条边的话
            # 图就真的断了，solve_positions 只能把两块各自锚回原点、精确叠在
            # 一起，而残差还是漂亮的 —— 正是最难发现的那种错。所以按解出来的
            # 行差摆一条纯先验的桥边，权重压到底，让它只负责连通。
            r1 = sorted(ra)[len(ra) // 2]
            r2 = min(rb, key=lambda r: abs(r - r1))
            bridges.append((ra[r1][0], rb[r2][0], r2 - r1))
    _log(verbose, f"[register] 列间边 {len(hjobs)} 条"
                  + (f"，无公共行的桥边 {len(bridges)} 条" if bridges else ""))
    hres = _pool(hjobs, lambda p: _measure_prepped(P[p[0]], P[p[1]],
                                                   h_vec[0], h_vec[1], tol))
    for (i, j), (dx, dy, v) in zip(hjobs, hres):
        _emit(i, j, "h", dx, dy, v, h_vec[0], h_vec[1])
    for i, j, dk in bridges:
        ex_ = h_vec[0] + dk * v_vec[0]
        ey_ = h_vec[1] + dk * v_vec[1]
        _emit(i, j, "h", ex_, ey_, 0.0, ex_, ey_, why="no-shared-row")

    # ================================================== 实测步距与重叠
    # 只用"走了正好一行"的列内边来算纵向步距，断链和补拍那种 0 行 / 2 行的
    # 链接混进来会把中位数拉歪
    one_row = []
    for c in cols:
        for a, b in zip(c["seq"][:-1], c["seq"][1:]):
            li = link_info[(a, b)]
            if li["ok"] and abs(li["step"]) == 1:
                s = float(li["step"])
                one_row.append((li["dx"] * s * sd, li["dy"] * s * sd, li["ncc"]))
    # 样本一条都没有时 _robust 会给 nan，而 measured_* 会被 nominal_positions
    # 直接拿去算坐标 —— nan 一旦进了画布尺寸就到处传染。退回先验并把 n_used
    # 记成 0，让总控看得见"这个数是先验不是实测"。
    mdxv, nv_used = _robust([a[0] for a in one_row], [a[2] for a in one_row])
    mdyv, _ = _robust([a[1] for a in one_row], [a[2] for a in one_row])
    if not one_row:
        mdxv, mdyv = v_vec[0] * sd, v_vec[1] * sd
    ah = [(e.dx, e.dy, e.response) for e in edges if e.kind == "h" and e.response > min_response]
    mdxh, nh_used = _robust([a[0] for a in ah], [a[2] for a in ah])
    mdyh, _ = _robust([a[1] for a in ah], [a[2] for a in ah])
    if not ah:
        mdxh, mdyh = h_vec[0] * sd, h_vec[1] * sd

    # ================================================== 冲突瓦片打分
    # 同一个 floor(key) 下有多张（L19 的 0035.5 / 0035.6）在这里只打分不删片。
    # 注意实测结论是它们并非同格位补拍，而是两个不同的物理行，所以除了
    # 一致性分数，还要把解出来的 row 和它们之间的实测行差报出去，
    # 让总控看清楚"到底该不该二选一"。
    cscore: dict[str, dict] = {}
    for c in cols:
        buckets: dict[int, list[int]] = {}
        for k in c["seq"]:
            buckets.setdefault(int(math.floor(tiles[k].key)), []).append(k)
        for _, grp in sorted(buckets.items()):
            if len(grp) < 2:
                continue
            for k in grp:
                inc = [e for e in edges if e.i == k or e.j == k]
                sib = [x for x in grp if x != k]
                rel = None
                if sib and P[k] is not None and P[sib[0]] is not None:
                    d = _measure_multi(P[k], P[sib[0]],
                                       [(0.0, 0.0), v_vec, (-v_vec[0], -v_vec[1])], tol)
                    rel = {"dx": d[0] * sd, "dy": d[1] * sd, "ncc": d[2],
                           "rows": (0, 1, -1)[d[3]] if d[3] >= 0 else None}
                # 偏差要跟"这条边应该走几行"比：列内边先按实测 dy 反推行数
                vf = (v_vec[0] * sd, v_vec[1] * sd)
                hf = (h_vec[0] * sd, h_vec[1] * sd)
                dev = []
                for e in inc:
                    if e.kind == "h":
                        base = hf
                    else:
                        kk = round(e.dy / vf[1]) if abs(vf[1]) > 1e-6 else 0
                        base = (kk * vf[0], kk * vf[1])
                    dev.append(math.hypot(e.dx - base[0], e.dy - base[1]))
                cscore[tiles[k].tid] = {
                    "index": k,
                    "row_abs": c["row_abs"][k],
                    "seg": c["seg"][k],
                    "n_edges": len(inc),
                    "ncc_sum": float(sum(e.response for e in inc)),
                    "ncc_mean": float(np.mean([e.response for e in inc])) if inc else 0.0,
                    "ncc_min": float(min([e.response for e in inc])) if inc else 0.0,
                    "n_strong": int(sum(1 for e in inc if e.response >= min_response)),
                    "dev_mean_px": float(np.mean(dev)) if dev else float("nan"),
                    "focus": float(tiles[k].focus),
                    "siblings": [tiles[x].tid for x in sib],
                    "sibling_rel": rel,
                    "same_row": bool(sib and c["row_abs"][k] == c["row_abs"][sib[0]]),
                }

    row_abs_all = {}
    for c in cols:
        for k in c["seq"]:
            row_abs_all[tiles[k].tid] = c["row_abs"][k]
    resp = np.array([e.response for e in edges], float)
    stats = {
        "scale_div": scale_div,
        "row_policy": row_policy,
        "n_tiles": n,
        "n_missing": len(missing),
        "missing_tids": [tiles[k].tid for k in missing],
        "small_shape": (int(Ws), int(Hs)),
        "tol_full_px": tol * sd,
        "stage1": stage1,
        "prior_full": {"dx_v": pri[0] * sd, "dy_v": pri[1] * sd,
                       "dx_h": pri[2] * sd, "dy_h": pri[3] * sd},
        # ---- 交付数字：实测步距与重叠（全分辨率像素）
        "measured_dx": float(mdxh),             # 列间 x 步距
        "measured_dy": float(mdyv),             # 列内 y 步距
        "measured_dx_v": float(mdxv),           # 列内 x 漂移（台面倾斜）
        "measured_dy_h": float(mdyh),           # 列间 y 漂移
        "measured_overlap_x_px": float(TILE_W - abs(mdxh)),
        "measured_overlap_y_px": float(TILE_H - mdyv),
        "nominal_overlap_x_px": float(TILE_W - abs(NOMINAL_FULL[2])),
        "nominal_overlap_y_px": float(TILE_H - NOMINAL_FULL[1]),
        "n_used_v": nv_used, "n_used_h": nh_used,
        "n_edges": len(edges),
        "n_edges_v": sum(1 for e in edges if e.kind == "v"),
        "n_edges_h": sum(1 for e in edges if e.kind == "h"),
        "n_rejected": len(rejected),
        "rejected": rejected,
        # ---- 行对齐
        # row_offsets 只是每列 *最大那一段* 相对左邻列的偏移，纯诊断用。
        # L19 这种被断链切成三段、三段偏移各不相同的列，照它摆一定错位；
        # 要摆瓦片请一律用 row_abs（每张一个绝对行号），细节看 seg_offsets。
        "row_offsets": {c["col"]: c["seg_offset"][max(c["seg_offset"],
                        key=lambda s: sum(1 for k in c["seq"] if c["seg"][k] == s))]
                        for c in cols},
        "seg_offsets": {c["col"]: dict(c["seg_offset"]) for c in cols},
        "segments": seg_report,
        "n_breaks": nbreak,
        "row_abs": row_abs_all,
        "conflict_scores": cscore,
        "response_pct": {p: float(np.percentile(resp, p)) for p in (5, 25, 50, 75, 95)}
        if len(resp) else {},
    }
    _log(verbose, f"[register] 实测 dy={mdyv:.1f} dx={mdxh:.1f} -> "
                  f"纵向重叠 {TILE_H - mdyv:.1f} px, 横向重叠 {TILE_W - abs(mdxh):.1f} px; "
                  f"拒绝 {len(rejected)}/{len(edges)}")
    return edges, stats


def nominal_positions(tiles, stats) -> np.ndarray:
    """按实测先验和解出来的绝对行号摆一个名义网格，供 solve_positions
    的 anchors 用：图一旦断成几块，块内相对位置还是对的，只能靠名义网格
    把各块摆回大致的位置。"""
    p = stats["prior_full"]
    rows = stats["row_abs"]
    out = np.zeros((len(tiles), 2), float)
    for k, t in enumerate(tiles):
        r, c = rows.get(t.tid, t.nominal_row), t.col_idx
        out[k] = (c * p["dx_h"] + r * p["dx_v"], c * p["dy_h"] + r * p["dy_v"])
    out -= out.min(axis=0)
    return out


# ---------------------------------------------------------------- 全局求解
def _resolve_anchors(n_tiles, anchors):
    """anchors 支持：None / 单个下标 / 下标列表 / {下标: (x, y)} /
    (n, 2) 名义网格坐标数组。数组形式最有用，见 nominal_positions。"""
    if anchors is None:
        return {}, None
    if isinstance(anchors, (int, np.integer)):
        return {int(anchors): (0.0, 0.0)}, None
    if isinstance(anchors, dict):
        return {int(k): (float(v[0]), float(v[1])) for k, v in anchors.items()}, None
    arr = np.asarray(anchors)
    if arr.ndim == 2 and arr.shape == (n_tiles, 2):
        return {}, arr.astype(float)
    return {int(k): (0.0, 0.0) for k in arr.ravel()}, None


def solve_positions(n_tiles, edges, anchors=None) -> tuple[np.ndarray, dict]:
    """加权最小二乘全局求解。返回 (positions[n,2] float64 全分辨率像素,
    诊断 dict 含 residual_rms, n_components, per_edge_residual)。

    x 和 y 完全解耦，同一个设计矩阵解两次。权重取 response：被判不可信的边
    权重是 1e-3，只负责保持图连通，不会把好边拽歪。

    图断成多块时的处理值得单说。块与块之间一条边都没有，最小二乘对它们的
    相对位置无话可说，残差也一样是 0 —— 如果各块都锚回原点，出来的就是几块
    精确重叠、而 residual_rms 漂亮得不行的图，这是本模块最危险的静默错误。
    所以：给了 anchors（尤其是 nominal_positions 的名义网格）就按它摆；
    没给就把各块沿 y 摞开，并在 diag 里立 disconnected 旗，宁可让图一眼看着
    不对，也不能让它看着对。
    """
    edges = list(edges)
    m = len(edges)
    pos = np.zeros((n_tiles, 2), float)
    fixed, nominal = _resolve_anchors(n_tiles, anchors)

    if m == 0:
        return pos, {"residual_rms": 0.0, "residual_rms_strong": 0.0,
                     "residual_max": 0.0, "residual_p95": 0.0,
                     "per_edge_residual": np.zeros(0), "n_components": n_tiles,
                     "components": [], "anchors": {}, "n_tiles": n_tiles,
                     "n_edges": 0, "disconnected": n_tiles > 1,
                     "layout": "empty"}

    ii = np.array([e.i for e in edges])
    jj = np.array([e.j for e in edges])
    w = np.array([max(e.response, 1e-6) for e in edges])
    bx = np.array([e.dx for e in edges])
    by = np.array([e.dy for e in edges])

    adj = sp.coo_matrix((np.ones(m), (ii, jj)), shape=(n_tiles, n_tiles))
    ncomp, labels = connected_components(adj, directed=False)
    deg = np.bincount(np.concatenate([ii, jj]), weights=np.concatenate([w, w]),
                      minlength=n_tiles)

    comps, used, solved = [], {}, []
    for c in range(ncomp):
        members = np.flatnonzero(labels == c)
        loc = {int(g): k for k, g in enumerate(members)}
        k = len(members)
        sel = np.flatnonzero(labels[ii] == c)
        comps.append({"label": c, "size": int(k), "n_edges": int(sel.size),
                      "members": members.tolist()})
        if sel.size == 0:                       # 孤立瓦片，只能靠名义位置
            if nominal is not None:
                pos[members] = nominal[members]
            used[int(members[0])] = "isolated"
            solved.append(members)
            continue

        rows = np.arange(sel.size)
        A = sp.coo_matrix(
            (np.concatenate([-w[sel], w[sel]]),
             (np.concatenate([rows, rows]),
              np.concatenate([[loc[int(x)] for x in ii[sel]],
                              [loc[int(x)] for x in jj[sel]]]))),
            shape=(sel.size, k)).tocsr()

        # 必须钉一个锚点消掉平移自由度：优先用调用方指定的，否则挑连接最强的
        anch = next((loc[int(g)] for g in members if int(g) in fixed), None)
        if anch is None:
            anch = int(np.argmax(deg[members]))
        wa = float(w[sel].sum()) + 1.0
        Aa = sp.vstack([A, sp.coo_matrix(([wa], ([0], [anch])), shape=(1, k))]).tocsr()
        sx = lsqr(Aa, np.concatenate([w[sel] * bx[sel], [0.0]]), atol=1e-12, btol=1e-12)[0]
        sy = lsqr(Aa, np.concatenate([w[sel] * by[sel], [0.0]]), atol=1e-12, btol=1e-12)[0]
        sol = np.stack([sx, sy], axis=1)

        gid = int(members[anch])
        if gid in fixed:
            sol += np.array(fixed[gid]) - sol[anch]
            used[gid] = "explicit"
        elif nominal is not None:
            # 整块平移到名义网格：用中位数而不是单点，免得锚点自己就是坏片
            sol += np.median(nominal[members] - sol, axis=0)
            used[gid] = "nominal-median"
        else:
            used[gid] = "origin"
        pos[members] = sol
        solved.append(members)

    layout = "single" if ncomp == 1 else ("anchored" if (fixed or nominal is not None)
                                          else "components-stacked")
    if layout == "components-stacked":
        # 谁也没告诉我们各块之间该差多少，那就至少别让它们精确重叠。
        # 沿 y 依次摞开，块间留一个瓦片高的空隙 —— 渲染出来是几段断开的
        # 竖条，一眼就知道该回去看 n_components，而不是当成一张好图发论文。
        cursor = 0.0
        for members in solved:
            blk = pos[members]
            pos[members] = blk - blk.min(axis=0) + np.array([0.0, cursor])
            cursor += float(np.ptp(blk[:, 1])) + TILE_H
        warnings.warn(
            f"solve_positions: 图断成 {ncomp} 块且未提供 anchors，"
            f"块间相对位置无解，已沿 y 摞开。正确做法是传 "
            f"anchors=nominal_positions(tiles, stats)。", RuntimeWarning, stacklevel=2)

    if not fixed and nominal is None:
        pos -= pos.min(axis=0)

    res = (pos[jj] - pos[ii]) - np.stack([bx, by], axis=1)
    per_edge = np.hypot(res[:, 0], res[:, 1])
    strong = w > 1e-2
    diag = {
        "residual_rms": float(math.sqrt(float((w * per_edge ** 2).sum() / w.sum()))),
        "residual_rms_strong": float(math.sqrt(float(
            (w[strong] * per_edge[strong] ** 2).sum() / w[strong].sum()))) if strong.any() else 0.0,
        "residual_max": float(per_edge.max()),
        "residual_p95": float(np.percentile(per_edge, 95)),
        "per_edge_residual": per_edge,
        "n_components": int(ncomp),
        # residual_rms 只在块内有意义：块之间没有边，断图时它照样是 0，
        # 所以判活儿好坏必须同时看 disconnected
        "disconnected": bool(ncomp > 1),
        "layout": layout,
        "components": comps,
        "anchors": used,
        "n_tiles": int(n_tiles),
        "n_edges": m,
    }
    return pos, diag


# ---------------------------------------------------------------- 自测
# 真数据只能验"看起来自洽"，验不了"确实对"。所以主力是合成测试：造一张已知
# 大图切成蛇形采集的瓦片，把这批数据真实存在的三个坑原样注入 —— L19 式的
# 走两步/退一步补拍、L26/L27 式的多一行、以及一张糊到没纹理的空片 —— 再看
# 解出来的行号和坐标能不能对上真值。
_SYN_DX, _SYN_DY, _SYN_DXV, _SYN_DYH = 457, 243, -1, 2      # 缩略图尺度真值
_SYN_SW, _SYN_SH = 480, 270


def _make_synthetic(cache_dir, seed=3):
    """5 列 x 8 行的蛇形马赛克，真值全部已知。返回 (tiles, truth)。"""
    import tiles as _T
    rng = np.random.default_rng(seed)
    big_w = 4 * _SYN_DX + _SYN_SW + 60
    big_h = 9 * _SYN_DY + _SYN_SH + 60
    base = cv2.GaussianBlur(rng.random((big_h, big_w, 3), dtype=np.float32), (0, 0), 18)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    tex = cv2.GaussianBlur(rng.random((big_h, big_w, 3), dtype=np.float32), (0, 0), 1.1)
    big = np.clip(50 + 130 * base + 100 * (tex - .5), 0, 255).astype(np.uint8)

    truth, tiles = {}, []
    for ci, col in enumerate(range(18, 23)):
        direction = "down" if col % 2 == 0 else "up"
        rows = list(range(-1, 8)) if col == 21 else list(range(8))   # L21 多一行且在顶上
        seq = rows if direction == "down" else rows[::-1]
        keys = [float(i + 1) for i in range(len(seq))]
        if col == 19:
            # 复刻 L19：拍完 row4 一次走两步到 row2(=4.5)，退一步补 row3(=4.6)，
            # 再跳两步到 row1。两个 2 行的跨越都没有重叠，列必然被切成三段。
            seq = [7, 6, 5, 4, 2, 3, 1, 0]
            keys = [1.0, 2.0, 3.0, 4.0, 4.5, 4.6, 5.0, 6.0]
        n = len(seq)
        for order, (r, key) in enumerate(zip(seq, keys)):
            jx, jy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            x = 20 + ci * _SYN_DX + r * _SYN_DXV + jx
            y = 20 + _SYN_DY + ci * _SYN_DYH + r * _SYN_DY + jy
            name = f"{key:07.1f}.png" if key != int(key) else f"{int(key):04d}.png"
            t = _T.Tile(col=col, col_idx=ci, direction=direction, order=order, key=key,
                        name=name, zip_path="", inner="",
                        nominal_row=order if direction == "down" else n - 1 - order)
            patch = big[y:y + _SYN_SH, x:x + _SYN_SW].copy()
            if col == 22 and r == 4:                 # 一张纯灰空片
                patch[:] = 128
            p = _T.cache_path(cache_dir, t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, patch)
            tiles.append(t)
            truth[t.tid] = (float(x) * 8, float(y) * 8, r)
    return tiles, truth


def _selftest():
    import shutil
    import tempfile
    from collections import Counter

    print("=" * 70)
    print("(a) 基元：相位相关方向、_ncc 索引、_win_search 亚像素")
    print("=" * 70)
    rng = np.random.default_rng(0)
    b = cv2.GaussianBlur(rng.random((600, 900), dtype=np.float32), (0, 0), 1.5) * 200
    dx, dy = 40, 25
    A = _prep(np.ascontiguousarray(b[100:370, 100:580]))
    B = _prep(np.ascontiguousarray(b[100 + dy:370 + dy, 100 + dx:580 + dx]))
    px, py = _pc_peak(A, B)
    print(f"_pc_peak = ({px:+.2f}, {py:+.2f})  期望 ({dx:+d}, {dy:+d})")
    print(f"_ncc 真值处 {_ncc(A, B, dx, dy, 64):+.3f}  零位移处 {_ncc(A, B, 0, 0, 64):+.3f}")
    mx, my, mv = _measure_prepped(A, B, 35.0, 20.0, 12.0)
    print(f"_measure_prepped = ({mx:+.2f}, {my:+.2f}) ncc={mv:.3f}")
    ok_a = (abs(px - dx) < 1 and abs(py - dy) < 1
            and abs(mx - dx) < .5 and abs(my - dy) < .5 and mv > .9)
    # 有先验但 tol=None 不能炸（曾经在这里抛 TypeError）
    try:
        _measure_prepped(A, B, 35.0, 20.0, None)
        ok_tol = True
    except TypeError:
        ok_tol = False
    print(f"tol=None + 有先验 不抛异常: {ok_tol}")
    print(f"_spread(2,2)={_spread(2, 2)} _spread(3,3)={_spread(3, 3)} (不许撞点)")
    ok_sp = len(_spread(2, 2)) == 2 and len(_spread(3, 3)) == 3
    print("结论:", "通过" if (ok_a and ok_tol and ok_sp) else "失败")

    print()
    print("=" * 70)
    print("(b) 极薄重叠的伪 NCC：厚度下限是否真的挡住了")
    print("=" * 70)
    H, W = A.shape
    ml = _min_overlap_px(H, W)
    print(f"厚度下限 = {ml:.1f} px（缩略图），本数据实测重叠约 32 px")
    for thick in (4, 8, 14, 32):
        v = _ncc(A, B, W - thick, 0, max(64.0, 0.012 * W * H), ml)
        print(f"  重叠 {thick:2d} 列 -> _ncc = {v:+.3f}"
              + ("  (被厚度下限判死)" if v <= -1 else ""))
    ok_b = (_ncc(A, B, W - 4, 0, max(64.0, 0.012 * W * H), ml) <= -1
            and _ncc(A, B, W - 8, 0, max(64.0, 0.012 * W * H), ml) <= -1)
    print("结论:", "通过" if ok_b else "失败")

    print()
    print("=" * 70)
    print("(c) 合成蛇形马赛克：注入 L19 式补拍 + 多一行的列 + 空片")
    print("=" * 70)
    work = tempfile.mkdtemp(prefix="reg_selftest_")
    try:
        cache = os.path.join(work, "cache")
        tiles, truth = _make_synthetic(cache)
        edges, st = build_edges(tiles, cache, scale_div=8, verbose=True)
        print(f"实测 dy={st['measured_dy'] / 8:.2f} (真值 {_SYN_DY})  "
              f"dx={st['measured_dx'] / 8:.2f} (真值 {_SYN_DX})  "
              f"dx_v={st['measured_dx_v'] / 8:.2f} (真值 {_SYN_DXV})  "
              f"dy_h={st['measured_dy_h'] / 8:.2f} (真值 {_SYN_DYH})")
        drow = Counter(st["row_abs"][k] - truth[k][2] for k in truth)
        print(f"row_abs - 真值行号 分布: {dict(drow)}  (只许有一个常数)")
        pos, diag = solve_positions(len(tiles), edges,
                                    anchors=nominal_positions(tiles, st))
        tp = np.array([[truth[t.tid][0], truth[t.tid][1]] for t in tiles])
        e = np.hypot(*((pos - pos.mean(0)) - (tp - tp.mean(0))).T)
        print(f"连通块 {diag['n_components']}  残差 rms {diag['residual_rms']:.2f} px")
        print(f"位置误差(全分辨率 px) 中位 {np.median(e):.2f}  p95 "
              f"{np.percentile(e, 95):.2f}  max {e.max():.2f}")
        ok_c = (len(drow) == 1 and diag["n_components"] == 1
                and np.percentile(e, 95) < 8.0)
        print("结论:", "通过" if ok_c else "失败")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    print("=" * 70)
    print("(d) 断图不许静默：两块互不相连时不能精确重叠")
    print("=" * 70)
    ed = [Edge(0, 1, 100, 0, .9, "h"), Edge(1, 2, 100, 0, .9, "h"),
          Edge(3, 4, 100, 0, .9, "h"), Edge(4, 5, 100, 0, .9, "h")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p2, d2 = solve_positions(6, ed, anchors=None)
    ov = bool(np.allclose(p2[:3], p2[3:]))
    print(f"n_components={d2['n_components']} disconnected={d2['disconnected']} "
          f"layout={d2['layout']}")
    print(f"residual_rms={d2['residual_rms']:.2e} (断图时它天然是 0，不能拿来判好坏)")
    print(f"两块是否精确重叠: {ov}   发出警告: {len(caught) > 0}")
    ok_d = (not ov) and d2["disconnected"] and len(caught) > 0
    print("结论:", "通过" if ok_d else "失败")

    print()
    print("总计:", "全部通过" if all([ok_a, ok_tol, ok_sp, ok_b, ok_c, ok_d]) else "有失败项")


if __name__ == "__main__":
    _selftest()
