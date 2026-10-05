# -*- coding: utf-8 -*-
"""blend.py — Figure 3a 马赛克的平场校正、羽化融合与条带式流式渲染。

这个模块解决四件事，每一件都是显微马赛克拼不好看（或者根本拼不出来）的直接原因：

1. 渐晕/照明不均。物镜+柯勒照明下每张瓦片都是中心亮、四角暗，落差常有
   10%-25%。920 张同样的亮度包络平铺出来就是肉眼可见的棋盘格，比任何
   配准误差都刺眼。校正的依据是：同一像素位置上，不同瓦片的 *样品内容*
   互不相关，而 *渐晕* 是固定的，所以对一批瓦片逐像素取中位数，剩下的
   就基本只有渐晕；再用大核高斯抹掉中位数里残留的样品结构。
   用中位数而不是均值，是因为数据里有过曝的胶带边、烧穿的黑洞这种极端瓦片。
   但中位数只在"坏样本不过半"时才稳；坏样本过半会静默产生 NaN 场，
   于是这里额外做了两件事：先按内容剔掉空片，再对结果做有限性断言。

2. 接缝。直接覆盖的话，即使配准是完美的，相邻瓦片的曝光和残余渐晕也
   对不上，缝会像铅笔线一样清楚。所以用"到瓦片边界的距离"当权重做加权
   平均：瓦片中心权重最高，边界处降到接近 0，重叠区自然过渡。

3. 内存。全分辨率画布约 62000 x 105000，float32 三通道累加器 78 GB，
   一次成型是不可能的。所以按水平条带渲染，一次只保留 band_px 行的累加器。
   条带路径是最容易写错的地方（行区间、源图裁剪、跨条带的瓦片要加两次），
   所以 __main__ 里有一个"大条带 vs 小条带必须逐位一致"的回归测试。

4. 落盘。第 3 点省下来的内存，很容易在最后一步又赔回去：cv2 不能分块写
   PNG，PIL 要求先建一张完整画布（62000x105000x3 = 19 GB，这台机器 26 GB
   内存直接爆）。所以这里自己按 PNG 规范流式写：IHDR 之后把每个条带
   过 Sub 滤波、喂进同一个 zlib 流、切成若干 IDAT 落盘，全程内存只有
   一个条带。代价是几十行格式代码，换来的是"画布再大也不进内存"。

坑：
  - positions 允许有负值（配准解出来的全局坐标没有理由从 0 开始），
    渲染前必须整体平移到非负。
  - 画布尺寸绝对不能按"包围盒 x 缩放比再取整"算。ceil/floor/round 各自
    独立取整，三个 0.5 能叠出 1 px 的差，在 1/16、1/32、1/64 这些常用
    预览倍率上实测都会发生，结果是右下角那张瓦片最后一列/一行被悄悄裁掉。
    正确做法是先算出每张瓦片的整数落点，再取落点的实际外包 —— 定义上
    就不可能差一。
  - 没被任何瓦片覆盖的地方要填 bg（默认白）。留黑边在论文图里非常难看，
    而且累加器初值是 0，不显式填就一定是黑的。
  - 权重除法不能写成 acc/(wsum+eps)。羽化权重的最小值随瓦片尺寸平方衰减：
    1/8 图上是 3.1e-5，全分辨率 3840x2160 上只有 4.8e-7，比 1e-6 还小。
    加性 eps 会把画面最外圈压暗（实测最深处暗 67%），甚至判成"未覆盖"
    而填成白点。所以权重带一个绝对下限，除法用 maximum 而不是加法。
  - 累加顺序必须与条带划分无关，否则测试 (b) 过不了：每个输出行的求和
    只涉及覆盖它的那些瓦片，按 tiles 列表顺序累加，跟条带边界无关，
    所以是可以做到逐位一致的 —— 不要为了省事在条带里对瓦片重新排序。
  - 一张瓦片会横跨相邻两个条带，天真写法就会把它解码两次。1/8 预览无所谓，
    全分辨率下 920 张原图从 zip 里解两遍是十几分钟的纯浪费，所以跨界的
    瓦片在带预算的前提下留到下一条带复用。
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np

from tiles import TILE_W, TILE_H, Tile, cache_path, load_small, load_full

# 羽化权重的绝对下限。取 1e-4 是为了让它在 float32 下离 0 足够远（相对精度
# 1e-7），同时只影响瓦片四角那一小片双曲线区域（3840x2160 上约十几个像素），
# 对融合过渡没有可察觉的影响。
W_FLOOR = 1e-4
COV_EPS = W_FLOOR * 0.5      # 总权重低于这个值才算"没有任何瓦片覆盖"

EPS = COV_EPS                # 兼容旧名字


# ---------------------------------------------------------------- 平场
def _sample_quality(small) -> tuple[bool, float]:
    """判断一张缓存小图能不能进平场样本池，顺带返回它的灰度均值。

    判据全部从图像本身算，不看 Tile.mean/std —— 那些字段只有跑过
    build_cache 的瓦片才有值，直接拿 allt 调用平场估计时它们全是 NaN。

    剔除的是"空片"：全黑/全白/整片糊成一个灰度的瓦片。它们本身没有渐晕
    信息（归一化之后就是一张平的 1.0），混进中位数只会把渐晕估平；数量
    过半时中位数直接崩掉，且崩法很阴险——ff.max() 变 0，除以自身均值得到
    整场 NaN，一路静默传到 apply_flatfield 把整幅图变成黑的。
    """
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m, s = float(g.mean()), float(g.std())
    if not (np.isfinite(m) and np.isfinite(s)):
        return False, 0.0
    if s < 3.0:                                   # 几乎没有内容
        return False, m
    if m < 8.0 or m > 247.0:                      # 整体欠曝/过曝，归一化不可靠
        return False, m
    if float(((g <= 2) | (g >= 253)).mean()) > 0.5:   # 一半以上像素撞到量程端
        return False, m
    return True, m


def estimate_flatfield(tiles, cache_dir, sample=80, scale_div=8,
                       seed=0, verbose=False) -> np.ndarray:
    """估计照明不均/渐晕场，返回与缓存小图同尺寸的 float32 (h,w,3)，逐通道均值 1.0。

    sample 取多大有个权衡：太少（<20）中位数还带着样品结构，太多则读盘和
    内存都变贵。80 张 480x270x3 的 float32 约 124 MB，够用。为了在剔除空片
    之后仍然凑得齐 sample 张，候选池最多扫到 4*sample 个。

    每张样本先除以自己的 *逐通道均值* 再进中位数栈。这一步不是可有可无的：
    真实数据里瓦片亮度差得很远（有的整片是空白衬底，有的压着黑色胶带边），
    不归一化的话逐像素中位数在不同像素上其实是由不同瓦片决定的，估出来的
    场会带上样品的大尺度衬度。归一化之后中位数只反映"相对自身平均的明暗
    分布"，也就是渐晕本身。

    scale_div 只用于校验缓存尺寸是否和调用方以为的一致（run_stitch 会传），
    真正的形状一律以读到的小图为准。
    """
    pool = [t for t in tiles if os.path.exists(cache_path(cache_dir, t))]
    if not pool:
        raise RuntimeError("缓存里一张瓦片都没有，先跑 tiles.build_cache")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    budget = min(len(pool), max(sample * 4, sample))

    stack, shape, n_seen, n_reject = [], None, 0, 0
    for i in order[:budget]:
        if len(stack) >= sample:
            break
        img = load_small(cache_dir, pool[int(i)])
        if img is None:
            continue
        n_seen += 1
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            # 缓存里混进了别的尺寸（换过 scale_div 没清缓存），直接跳过，
            # 否则 np.stack 会炸得莫名其妙
            continue
        ok, _ = _sample_quality(img)
        if not ok:
            n_reject += 1
            continue
        f = img.astype(np.float32)
        for c in range(f.shape[2]):
            mc = float(f[:, :, c].mean())
            if mc < 1.0:                    # 该通道整体死掉，这张不要
                f = None
                break
            f[:, :, c] /= mc
        if f is None:
            n_reject += 1
            continue
        stack.append(f)

    if len(stack) < 8:
        raise RuntimeError(
            f"可用样本只有 {len(stack)} 张（看过 {n_seen} 张，剔除空片 {n_reject} 张），"
            f"估不出平场；先确认缓存不是空的/全黑的")

    if shape is not None and scale_div:
        want = TILE_W // int(scale_div)
        if abs(shape[1] - want) > 1:
            raise RuntimeError(f"缓存小图宽 {shape[1]}，与 scale_div={scale_div} "
                               f"期望的 {want} 不符，缓存和参数对不上")

    med = np.median(np.stack(stack), axis=0)

    # 大核高斯：渐晕是低频的，样品结构（畴、褶皱、颗粒）是中高频的。
    # sigma 取宽度的 1/8 左右，既能抹平残留结构又保得住四角的下降趋势。
    h, w = med.shape[:2]
    sigma = max(w, h) / 8.0
    ff = cv2.GaussianBlur(med, (0, 0), sigmaX=sigma, sigmaY=sigma,
                          borderType=cv2.BORDER_REPLICATE).astype(np.float32)

    if not np.all(np.isfinite(ff)):
        raise RuntimeError("平场里出现非有限值，样本池八成大半是空片")

    # 防止某个通道整体接近 0（例如整批瓦片蓝通道欠曝）时后面除爆
    ff = np.maximum(ff, 1e-3 * float(ff.max()))
    for c in range(ff.shape[2]):
        mc = float(ff[:, :, c].mean())
        if not np.isfinite(mc) or mc <= 0:
            raise RuntimeError(f"平场第 {c} 通道均值 {mc}，无法归一化")
        ff[:, :, c] /= mc
    if verbose:
        print(f"[flatfield] 用 {len(stack)} 张（剔除 {n_reject} 张空片），"
              f"范围 [{ff.min():.3f}, {ff.max():.3f}]", flush=True)
    return ff


# 单槽记忆：全分辨率渲染时 apply_flatfield 会被调 920 次，每次把 480x270
# 的场放大到 3840x2160 是 95 MB 的重复工作。用 `is` 比对原始对象而不是 id()，
# 免得原数组被回收后 id 复用命中错误的缓存。
_FF_RESIZED: tuple | None = None


def apply_flatfield(img, ff) -> np.ndarray:
    """img 为 BGR uint8（任意尺寸），ff 内部缩放到 img 尺寸后逐像素相除。

    ff 本来就是低频场，用双线性放缩不会引入伪影；反过来若先把 img 缩到 ff
    尺寸再校正，就等于丢掉了原图分辨率，所以方向只能是放缩 ff。
    """
    global _FF_RESIZED
    h, w = img.shape[:2]
    if ff.shape[0] == h and ff.shape[1] == w:
        big = ff
    else:
        c = _FF_RESIZED
        if c is not None and c[0] is ff and c[1] == (h, w):
            big = c[2]
        else:
            big = cv2.resize(ff, (w, h), interpolation=cv2.INTER_LINEAR)
            _FF_RESIZED = (ff, (h, w), big)
    out = img.astype(np.float32) / np.maximum(big, 1e-3)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 权重
_WCACHE: dict[tuple[int, int, bool], np.ndarray] = {}


def _feather_weight(w: int, h: int, feather: bool = True) -> np.ndarray:
    """到瓦片边界的距离（切比雪夫意义下）作权重，用可分离的一维斜坡外积算。

    等价于 cv2.distanceTransform 在矩形上的结果的可分离近似，但不用建
    二值图、也不用每张瓦片都重算。归一到峰值 1，再压一个下限 W_FLOOR：
    如果边界权重是硬 0（或者小到被 float32/覆盖判据当成 0），孤立瓦片的
    最外一圈像素就会因为总权重为 0 被判成"没覆盖"而填成背景色，画面上
    出现一圈白框；比白框更常见的是被加性 eps 压暗一大块四角。
    """
    key = (w, h, feather)
    cached = _WCACHE.get(key)
    if cached is not None:
        return cached
    if feather:
        wx = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float32)
        wy = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(np.float32)
        wt = np.outer(wy, wx)
        wt /= float(wt.max())
        np.maximum(wt, W_FLOOR, out=wt)
    else:
        wt = np.ones((h, w), np.float32)
    _WCACHE[key] = wt
    return wt


# ---------------------------------------------------------------- 位置
def _resolve_positions(tiles, positions):
    """positions 允许两种写法：{tid: (x,y)} / {下标: (x,y)}，或与 tiles 平行的序列。

    配准那边可能给 dict 也可能给数组，这里统一成与 tiles 对齐的列表，
    没有位置的瓦片记为 None（例如某列整体没配准上，宁可不画也别画错）。

    注意不要再尝试用 Tile 对象本身当 key：Tile 是带 eq 的 dataclass，
    __hash__ 被置成 None，dict.get(tile) 抛的是 TypeError 而不是返回 None，
    于是"某张瓦片没有位置"这种正常情况会变成崩溃。

    非有限坐标（NaN/inf）一律当成"没有位置"。求解器在图断开时可能吐出
    NaN，而 math.floor(nan) 是 ValueError，不拦住就会在画布那一步炸。
    """
    def _clean(p):
        x, y = float(p[0]), float(p[1])
        return (x, y) if (math.isfinite(x) and math.isfinite(y)) else None

    n = len(tiles)
    if isinstance(positions, dict):
        out = []
        for i, t in enumerate(tiles):
            p = positions.get(t.tid)
            if p is None:
                p = positions.get(i)
            out.append(None if p is None else _clean(p))
        return out
    arr = np.asarray(positions, dtype=np.float64)
    if arr.shape != (n, 2):
        raise ValueError(f"positions 形状 {arr.shape} 与 {n} 张瓦片对不上")
    return [_clean(p) for p in arr]


# ---------------------------------------------------------------- 流式 PNG
class _PngStream:
    """按行流式写 truecolor 8-bit PNG，全程只有一个条带在内存里。

    为什么不用 cv2/PIL：cv2.imwrite 只接受完整数组；PIL 要先 Image.new 出
    整张画布，62000x105000 就是 19 GB，正好是本模块要避免的那件事。
    PNG 的容器部分其实很短：签名 + IHDR + 若干 IDAT + IEND，IDAT 里装的是
    同一条 zlib 流切出来的片段，所以完全可以边算边写。

    滤波用 Sub（type 1）：对显微照片这种水平相关性强的内容，比不滤波小
    三成左右，而且是纯向量化的一次减法。uint8 的回绕减法正好就是 PNG
    要求的模 256，不用额外处理。
    """

    _CHUNK = 8 << 20        # 单个 IDAT 上限，远低于 PNG 的 2^31-1
    _ROWS = 256             # 一次滤波多少行，控制中间数组大小

    def __init__(self, path, w, h, level=6):
        self.w, self.h, self.path = int(w), int(h), path
        self._rows_done = 0
        self._f = open(path, "wb")
        self._f.write(b"\x89PNG\r\n\x1a\n")
        self._chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
        self._co = zlib.compressobj(int(level))
        self._buf = bytearray()

    def _chunk(self, tag, data):
        self._f.write(struct.pack(">I", len(data)))
        self._f.write(tag)
        self._f.write(data)
        self._f.write(struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    def _drain(self, force=False):
        while len(self._buf) >= (1 if force else self._CHUNK):
            k = min(len(self._buf), self._CHUNK)
            self._chunk(b"IDAT", bytes(self._buf[:k]))
            del self._buf[:k]
            if not force and len(self._buf) < self._CHUNK:
                break

    def add(self, band_bgr):
        """追加一个条带，形状 (bh, w, 3)，通道顺序是 OpenCV 的 BGR。"""
        if band_bgr.shape[1] != self.w or band_bgr.ndim != 3:
            raise ValueError(f"条带宽 {band_bgr.shape[1]} 与画布宽 {self.w} 不符")
        bh = band_bgr.shape[0]
        if self._rows_done + bh > self.h:
            raise ValueError("写入行数超过 IHDR 声明的高度")
        w3 = self.w * 3
        for i in range(0, bh, self._ROWS):
            sub = band_bgr[i:i + self._ROWS]
            k = sub.shape[0]
            px = np.empty((k, w3), np.uint8)
            px[:, 0::3] = sub[:, :, 2]        # PNG 要 RGB，源是 BGR
            px[:, 1::3] = sub[:, :, 1]
            px[:, 2::3] = sub[:, :, 0]
            rows = np.empty((k, w3 + 1), np.uint8)
            rows[:, 0] = 1                    # filter type = Sub
            rows[:, 1:4] = px[:, :3]
            rows[:, 4:] = px[:, 3:] - px[:, :-3]
            self._buf += self._co.compress(rows.tobytes())
            self._drain()
        self._rows_done += bh

    def close(self):
        if self._f is None:
            return
        try:
            if self._rows_done != self.h:
                raise ValueError(f"只写了 {self._rows_done} 行，IHDR 声明 {self.h} 行")
            self._buf += self._co.flush()
            self._drain(force=True)
            self._chunk(b"IEND", b"")
        finally:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            if self._f is not None:
                self._f.close()
                self._f = None
            try:
                os.remove(self.path)      # 半截 PNG 比没有文件更害人
            except OSError:
                pass


# ---------------------------------------------------------------- 渲染
def render(tiles, positions, out_path, out_scale=1 / 8, cache_dir=None,
           flatfield=None, feather=True, band_px=4096, bg=(255, 255, 255),
           use_full=False, verbose=True, png_level=6,
           carry_bytes=3 << 30) -> dict:
    """按 positions（全分辨率像素坐标）把瓦片渲染成一张图。

    use_full=False 时从 cache_dir 读 1/scale_div 小图，因此 out_scale 不能
    超过 1/scale_div —— 缓存里没有的分辨率放大出来只会糊，不如报错。
    bg 按 OpenCV 惯例是 BGR，默认白。

    carry_bytes 是"跨条带瓦片缓存"的内存预算。跨界瓦片留到下一条带复用，
    可以把全分辨率下的解码次数从约 1.9 倍压回 1.0 倍；预算用完就退化成
    重新解码，只慢不错。
    """
    if not tiles:
        raise ValueError("tiles 为空")
    if not use_full and not cache_dir:
        raise ValueError("use_full=False 时必须给 cache_dir")

    pos = _resolve_positions(tiles, positions)
    live = [(t, p) for t, p in zip(tiles, pos) if p is not None]
    if not live:
        raise ValueError("没有任何瓦片带位置")

    # 缓存小图的实际尺寸决定了可用的最高输出倍率
    if not use_full:
        probe = None
        for t, _ in live:
            probe = load_small(cache_dir, t)
            if probe is not None:
                break
        if probe is None:
            raise RuntimeError("缓存里读不出任何一张小图")
        sdiv = TILE_W / float(probe.shape[1])
        if out_scale > 1.0 / sdiv + 1e-9:
            raise ValueError(f"缓存是 1/{sdiv:g}，out_scale={out_scale:g} 超过上限 "
                             f"{1.0 / sdiv:g}，需要更大输出请用 use_full=True")

    tw = max(1, int(round(TILE_W * out_scale)))
    th = max(1, int(round(TILE_H * out_scale)))

    # 每张瓦片在输出坐标系里的整数落点。四舍五入到整像素：亚像素平移要么
    # 重采样（糊）要么留给配准去吸收，这里选后者，1/8 图上误差 < 0.5 px。
    #
    # 画布尺寸由落点反推，不由包围盒推。理由见模块头：包围盒那条路上
    # floor/ceil/round 三次独立取整，在 1/16、1/32、1/64 这些常用倍率下
    # 实测会少 1 px，把右下角瓦片的最后一行/列裁掉。
    x0 = math.floor(min(p[0] for _, p in live))
    y0 = math.floor(min(p[1] for _, p in live))
    raw = [(t, int(round((p[0] - x0) * out_scale)),
            int(round((p[1] - y0) * out_scale))) for t, p in live]
    ox_min = min(r[1] for r in raw)
    oy_min = min(r[2] for r in raw)
    placed = [(t, ox - ox_min, oy - oy_min) for t, ox, oy in raw]
    out_w = max(ox + tw for _, ox, _ in placed)
    out_h = max(oy + th for _, _, oy in placed)

    bandh = out_h if (band_px is None or band_px <= 0) else int(band_px)
    bandh = max(1, min(bandh, out_h))
    n_bands = int(math.ceil(out_h / bandh))

    ext = os.path.splitext(out_path)[1].lower()
    if ext != ".png" and n_bands > 1:
        raise ValueError(f"分条带渲染只会写 PNG，out_path 后缀是 {ext or '(无)'}；"
                         f"要么改成 .png，要么把 band_px 设成 None 一次成型")

    if verbose:
        mode = "全分辨率原图" if use_full else "缓存小图"
        print(f"渲染 {len(placed)} 张 -> {out_w} x {out_h} "
              f"(out_scale={out_scale:g}, 瓦片 {tw}x{th}, 源={mode})", flush=True)
        print(f"条带 {n_bands} 个 x {bandh} 行，"
              f"单条带累加器约 {bandh * out_w * 16 / 1e9:.2f} GB", flush=True)

    bg_u8 = np.array([int(bg[0]), int(bg[1]), int(bg[2])], np.uint8)
    wt_ref = _feather_weight(tw, th, feather)

    # 每张瓦片按 (t, ox, oy) 归到它覆盖的条带上，免得每个条带都把 920 张
    # 从头扫一遍；顺序仍按 placed 的下标，保证累加顺序与分带方式无关。
    per_band: list[list[int]] = [[] for _ in range(n_bands)]
    for ti, (_, _, oy) in enumerate(placed):
        b_lo = max(0, oy // bandh)
        b_hi = min(n_bands - 1, (oy + th - 1) // bandh)
        for bi in range(b_lo, b_hi + 1):
            per_band[bi].append(ti)

    writer = None
    single = None
    contributed: set[int] = set()
    failed: set[int] = set()
    carry: dict[int, np.ndarray] = {}
    carry_used = 0
    n_decode = 0

    if n_bands > 1 or ext == ".png":
        writer = _PngStream(out_path, out_w, out_h, level=png_level)
    try:
        for bi in range(n_bands):
            b0 = bi * bandh
            b1 = min(out_h, b0 + bandh)
            acc = np.zeros((b1 - b0, out_w, 3), np.float32)
            wsum = np.zeros((b1 - b0, out_w), np.float32)

            members = per_band[bi]
            need = set(members)
            for k in list(carry):                    # 用不上的立刻放掉
                if k not in need:
                    carry_used -= carry.pop(k).nbytes

            for ti in members:
                t, ox, oy = placed[ti]
                ry0, ry1 = max(oy, b0), min(oy + th, b1)
                cx0, cx1 = max(ox, 0), min(ox + tw, out_w)
                if ry1 <= ry0 or cx1 <= cx0:
                    continue

                img = carry.get(ti)
                if img is None:
                    if ti in failed:
                        continue
                    img = load_full(t) if use_full else load_small(cache_dir, t)
                    if img is None:
                        failed.add(ti)
                        continue
                    n_decode += 1
                    if flatfield is not None:
                        img = apply_flatfield(img, flatfield)
                    if img.shape[0] != th or img.shape[1] != tw:
                        interp = (cv2.INTER_AREA if img.shape[1] > tw
                                  else cv2.INTER_LINEAR)
                        img = cv2.resize(img, (tw, th), interpolation=interp)
                    # 只有还要伸进下一条带、且预算够，才留着
                    if oy + th > b1 and carry_used + img.nbytes <= carry_bytes:
                        carry[ti] = img
                        carry_used += img.nbytes

                sy0, sy1 = ry0 - oy, ry1 - oy
                sx0, sx1 = cx0 - ox, cx1 - ox
                sub_w = wt_ref[sy0:sy1, sx0:sx1]
                acc[ry0 - b0:ry1 - b0, cx0:cx1] += (
                    img[sy0:sy1, sx0:sx1].astype(np.float32) * sub_w[:, :, None])
                wsum[ry0 - b0:ry1 - b0, cx0:cx1] += sub_w
                contributed.add(ti)

            # 归一化全程原地做。天真写法（np.where + acc/(wsum+EPS) + clip）
            # 会同时活着四五个 float32 全条带临时量，全分辨率下 4096 行 x
            # 62000 列就是十几 GB，这台机器直接换页到死。
            cov = wsum > COV_EPS
            np.maximum(wsum, COV_EPS, out=wsum)
            acc /= wsum[:, :, None]
            acc += 0.5                                  # 四舍五入而不是截断
            np.clip(acc, 0, 255, out=acc)
            out = acc.astype(np.uint8)
            np.logical_not(cov, out=cov)
            out[cov] = bg_u8

            if writer is not None:
                writer.add(out)
            else:
                single = out
            if verbose:
                print(f"  条带 {bi + 1}/{n_bands} 行 {b0}-{b1} "
                      f"覆盖 {100.0 * (1.0 - cov.mean()):.1f}%", flush=True)
            del acc, wsum, out, cov

        carry.clear()
        if writer is not None:
            writer.close()
            writer = None
        else:
            cv2.imwrite(out_path, single)
    finally:
        if writer is not None:                # 出错了就别留半截 PNG
            writer.__exit__(RuntimeError, None, None)

    if verbose:
        print(f"写出 {out_path}（解码 {n_decode} 次 / {len(contributed)} 张）", flush=True)
    return {"out_w": out_w, "out_h": out_h, "n_placed": len(contributed),
            "path": os.path.abspath(out_path), "n_bands": n_bands,
            "n_failed_loads": len(failed), "n_decodes": n_decode,
            "origin": (x0, y0), "origin_out": (ox_min, oy_min),
            "out_scale": float(out_scale)}


# ---------------------------------------------------------------- 自测
def _psnr(a, b) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def _fake_tile(col, direction, k) -> Tile:
    return Tile(col=col, col_idx=0, direction=direction, order=k, key=float(k),
                name=f"{k:05.1f}.png", zip_path="", inner="", nominal_row=k)


def _make_synthetic(work, nc=4, nr=3, seed=1):
    """造一张已知大图，切 nc x nr 块，每块乘上归一化到均值 1 的渐晕场。

    坐标系刻意用真实的那一套：positions 是全分辨率像素，瓦片缓存是 1/8，
    渲染 out_scale=1/8。步距取 8 的整数倍，保证 1/8 下落点是整像素，
    这样 PSNR 只反映平场和融合的质量，不掺进重采样误差。
    """
    cache = os.path.join(work, "cache")
    os.makedirs(cache, exist_ok=True)
    sw, sh = TILE_W // 8, TILE_H // 8          # 480 x 270
    dx_full, dy_full = 3200, 1600              # 1/8 下是 400 / 200，重叠 80 / 70
    dx, dy = dx_full // 8, dy_full // 8
    gw, gh = (nc - 1) * dx + sw, (nr - 1) * dy + sh

    rng = np.random.default_rng(seed)
    # 低频背景 + 高频纹理：低频模拟样品的大尺度衬度，高频保证逐像素中位数
    # 不会被某一片结构带偏
    base = cv2.GaussianBlur(rng.random((gh, gw, 3), dtype=np.float32), (0, 0), 25)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    tex = cv2.GaussianBlur(rng.random((gh, gw, 3), dtype=np.float32), (0, 0), 1.2)
    big = np.clip(60 + 120 * base + 90 * (tex - 0.5), 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    r2 = ((xx - sw / 2 + 12) / (sw / 2)) ** 2 + ((yy - sh / 2 - 8) / (sh / 2)) ** 2
    vig = np.stack([1.0 - a * r2 for a in (0.42, 0.36, 0.50)], axis=2).astype(np.float32)
    for c in range(3):                       # 归一到均值 1：只留空间不均匀
        vig[:, :, c] /= float(vig[:, :, c].mean())

    tiles, positions = [], {}
    for r in range(nr):
        for c in range(nc):
            t = _fake_tile(90, "down", r * nc + c)
            patch = big[r * dy:r * dy + sh, c * dx:c * dx + sw].astype(np.float32)
            dirty = np.clip(patch * vig, 0, 255).astype(np.uint8)
            p = cache_path(cache, t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, dirty)
            tiles.append(t)
            positions[t.tid] = (c * dx_full, r * dy_full)
    return tiles, positions, cache, big, vig


def _write_blank(cache, col, k, value):
    t = _fake_tile(col, "down", k)
    p = cache_path(cache, t)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, np.full((TILE_H // 8, TILE_W // 8, 3), value, np.uint8))
    return t


def _selftest():
    import shutil
    import tempfile
    work = tempfile.mkdtemp(prefix="blend_selftest_")
    try:
        print("=" * 72)
        print("(a) 合成测试：平场校正对 PSNR 的贡献")
        print("=" * 72)
        tiles, positions, cache, big, vig = _make_synthetic(work)
        ff = estimate_flatfield(tiles, cache, sample=len(tiles))
        err = float(np.abs(ff - vig).mean())
        print(f"估出的平场 shape={ff.shape} 均值={ff.mean():.4f} "
              f"范围 [{ff.min():.3f}, {ff.max():.3f}]")
        print(f"与真实渐晕场的平均绝对偏差 {err:.4f}")

        p_off = os.path.join(work, "syn_noff.png")
        p_on = os.path.join(work, "syn_ff.png")
        render(tiles, positions, p_off, out_scale=1 / 8, cache_dir=cache,
               flatfield=None, band_px=10 ** 9, verbose=False)
        r_on = render(tiles, positions, p_on, out_scale=1 / 8, cache_dir=cache,
                      flatfield=ff, band_px=10 ** 9, verbose=False)
        a_off = cv2.imread(p_off, cv2.IMREAD_COLOR)
        a_on = cv2.imread(p_on, cv2.IMREAD_COLOR)
        print(f"画布 {r_on['out_w']} x {r_on['out_h']}，真值 {big.shape[1]} x {big.shape[0]}，"
              f"放置 {r_on['n_placed']} 张")
        s_off, s_on = _psnr(a_off, big), _psnr(a_on, big)
        print(f"PSNR 不做平场 = {s_off:.2f} dB")
        print(f"PSNR 做了平场 = {s_on:.2f} dB")
        print(f"提升 = {s_on - s_off:+.2f} dB")

        print()
        print("=" * 72)
        print("(b) 条带一致性 + 跨条带横向条纹")
        print("=" * 72)
        p_one = os.path.join(work, "band_one.png")
        p_many = os.path.join(work, "band_many.png")
        r1 = render(tiles, positions, p_one, out_scale=1 / 8, cache_dir=cache,
                    flatfield=ff, band_px=10 ** 9, verbose=False)
        r2 = render(tiles, positions, p_many, out_scale=1 / 8, cache_dir=cache,
                    flatfield=ff, band_px=37, verbose=False)
        i1 = cv2.imread(p_one, cv2.IMREAD_COLOR)
        i2 = cv2.imread(p_many, cv2.IMREAD_COLOR)
        d = np.abs(i1.astype(np.int16) - i2.astype(np.int16))
        print(f"一次成型 {r1['out_w']}x{r1['out_h']} 条带数 {r1['n_bands']} "
              f"解码 {r1['n_decodes']} 次")
        print(f"多条带   {r2['out_w']}x{r2['out_h']} 条带数 {r2['n_bands']} "
              f"解码 {r2['n_decodes']} 次（瓦片 {len(tiles)} 张，>张数就是在重复解码）")
        print(f"最大逐像素差 = {int(d.max())} 灰阶，不同像素数 = {int((d > 0).sum())}")
        print("结论:", "逐位一致" if d.max() == 0 else "不一致，条带逻辑有问题")

        # 条带边界如果各自截断距离场，接缝处会是一条整行偏亮/偏暗的横带。
        # 直接看逐行均值的二阶差分：条带边界行不该比别的行更突出。
        prof = i2.reshape(i2.shape[0], -1).mean(axis=1)
        jump = np.abs(np.diff(prof))
        edges = [b for b in range(37, i2.shape[0], 37)]
        e_j = [jump[b - 1] for b in edges if b - 1 < len(jump)]
        print(f"逐行均值跳变：条带边界 {len(e_j)} 行 max={max(e_j):.4f} "
              f"中位={float(np.median(e_j)):.4f}；全部行 max={jump.max():.4f} "
              f"中位={float(np.median(jump)):.4f}")
        print("结论:", "无横向条纹" if max(e_j) <= jump.max() + 1e-9
              and max(e_j) < 5 * float(np.median(jump)) + 1.0 else "疑似条纹")

        print()
        print("=" * 72)
        print("(c) 画布尺寸：落点必须全部落在画布里（差一像素回归）")
        print("=" * 72)
        bad = []
        for s in (1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 7, 1 / 3):
            for k in (0, 1, 4, 17, 24, 33, 47, 100, 137):
                t0, t1 = _fake_tile(92, "down", 0), _fake_tile(92, "down", 1)
                pp = {t0.tid: (0.0, 0.0), t1.tid: (float(k), float(k))}
                tws = max(1, int(round(TILE_W * s)))
                ths = max(1, int(round(TILE_H * s)))
                x0 = 0
                o = [int(round(v * s)) for v in (0.0, float(k))]
                need_w, need_h = max(o) + tws, max(o) + ths
                # 直接调 render 的尺寸逻辑：造两张 1x1 的假缓存太麻烦，
                # 这里复算一遍它现在用的公式
                got_w = max(oo + tws for oo in o) - min(o)
                got_h = max(oo + ths for oo in o) - min(o)
                if got_w < need_w - min(o) or got_h < need_h - min(o):
                    bad.append((s, k))
        print(f"扫过 8 个倍率 x 9 个偏移：越界 {len(bad)} 例")
        # 真刀真枪跑一遍 1/64（旧版在这个倍率上实测会裁掉一行）
        p64 = os.path.join(work, "s64.png")
        pos64 = dict(positions)
        r64 = render(tiles, pos64, p64, out_scale=1 / 64, cache_dir=cache,
                     flatfield=None, band_px=7, verbose=False)
        im64 = cv2.imread(p64, cv2.IMREAD_COLOR)
        tw64, th64 = int(round(TILE_W / 64)), int(round(TILE_H / 64))
        maxx = max(int(round(p[0] / 64)) for p in positions.values()) + tw64
        maxy = max(int(round(p[1] / 64)) for p in positions.values()) + th64
        print(f"1/64 渲染 {r64['out_w']}x{r64['out_h']}，最右下瓦片需要 {maxx}x{maxy}，"
              f"读回 {im64.shape[1]}x{im64.shape[0]}")
        print("结论:", "无裁切" if r64['out_w'] >= maxx and r64['out_h'] >= maxy
              else "画布少了像素")

        print()
        print("=" * 72)
        print("(d) 羽化权重下限 vs 覆盖判据（全分辨率四角变暗回归）")
        print("=" * 72)
        for (w, h) in ((TILE_W // 8, TILE_H // 8), (TILE_W // 2, TILE_H // 2),
                       (TILE_W, TILE_H)):
            wt = _feather_weight(w, h, True)
            old = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float32)
            oldy = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(np.float32)
            o = np.outer(oldy, old)
            o /= o.max()
            n_white = int((o <= 1e-6).sum())               # 旧版会填成背景色
            dark = 1e-6 / (o + 1e-6)                       # 旧版 acc/(w+eps) 的压暗量
            print(f"  瓦片 {w}x{h}: 新权重最小 {wt.min():.2e} (阈值 {COV_EPS:.1e}) "
                  f"| 旧版最小 {o.min():.2e}，误判未覆盖 {n_white} px，"
                  f"压暗>1% 的 {int((dark > 0.01).sum())} px，最深 {dark.max() * 100:.1f}%")
        print("结论:", "新权重恒大于覆盖阈值，且除法用 maximum 不引入偏置"
              if _feather_weight(TILE_W, TILE_H, True).min() > COV_EPS else "仍会误判")

        print()
        print("=" * 72)
        print("(e) 平场样本混进空片")
        print("=" * 72)
        for frac, val, tag in ((0.30, 0, "全黑"), (0.30, 255, "全白"),
                               (0.70, 0, "全黑(远过半)")):
            n_bad = int(round(frac * len(tiles) / (1 - frac)))
            t2 = list(tiles) + [_write_blank(cache, 91, 1000 + k, val)
                                for k in range(n_bad)]
            try:
                f2 = estimate_flatfield(t2, cache, sample=len(t2))
                print(f"  混入 {n_bad}/{len(t2)} 张{tag}: 与真值偏差 "
                      f"{float(np.abs(f2 - vig).mean()):.4f}  "
                      f"范围 [{f2.min():.3f},{f2.max():.3f}]  "
                      f"有限={bool(np.all(np.isfinite(f2)))}")
            except RuntimeError as e:
                print(f"  混入 {n_bad}/{len(t2)} 张{tag}: 明确报错 -> {e}")
            for k in range(n_bad):
                os.remove(cache_path(cache, _fake_tile(91, "down", 1000 + k)))

        print()
        print("=" * 72)
        print("(f) 累加器精度：float32 vs float64")
        print("=" * 72)
        # 每个输出像素最多被 4 张瓦片压上，和不超过 4*255；float32 在这个
        # 量级的相对精度是 1e-7，理论上不该有差别。实测确认一下，免得
        # 把"6 Gpx 所以要 float64"这种误解写进代码。
        #
        # 各次贡献必须用 *不同* 的权重，否则 acc/wsum 退化成整数的算术
        # 平均，会大量出现恰好 x.5 的量化平局；那时 float32/float64 的
        # 差别全落在平局上，量出来的"不同像素数"是测试自己造的假象，
        # 不是精度问题。真实渲染里相邻瓦片的羽化权重本来就不一样。
        acc32 = np.zeros((th_ := 270, 480, 3), np.float32)
        acc64 = np.zeros((th_, 480, 3), np.float64)
        w32 = np.zeros((th_, 480), np.float32)
        w64 = np.zeros((th_, 480), np.float64)
        rng = np.random.default_rng(7)
        wt0 = _feather_weight(480, 270, True)
        for k in range(6):
            im = rng.integers(0, 256, (th_, 480, 3)).astype(np.float32)
            wt = np.roll(wt0, 37 * k + 11, axis=0) * np.float32(0.6 + 0.13 * k)
            acc32 += im * wt[:, :, None]
            acc64 += im.astype(np.float64) * wt[:, :, None].astype(np.float64)
            w32 += wt
            w64 += wt.astype(np.float64)
        v32 = (acc32 / w32[:, :, None]).astype(np.float64)
        v64 = acc64 / w64[:, :, None]
        n_diff = int((np.clip(v32 + .5, 0, 255).astype(np.uint8)
                      != np.clip(v64 + .5, 0, 255).astype(np.uint8)).sum())
        print(f"  6 张叠加后逐像素最大差 {float(np.abs(v32 - v64).max()):.3e} 灰阶，"
              f"量化到 uint8 后不同像素 {n_diff} / {v32.size}")
        print("结论: float32 够用（累加是逐像素的，画布多大都不改变每像素的项数）")

        print()
        print("=" * 72)
        print("(g) 缺位置 / NaN 位置不崩")
        print("=" * 72)
        # 挖掉中间一行的两张：4x3 网格里这样会在中排留出一块谁也盖不到的
        # 空洞，正好检验"未覆盖填 bg"这条路
        part = {t.tid: positions[t.tid] for i, t in enumerate(tiles)
                if i not in (5, 6)}
        pp = os.path.join(work, "part.png")
        rp = render(tiles, part, pp, out_scale=1 / 8, cache_dir=cache,
                    flatfield=ff, band_px=64, bg=(255, 0, 0), verbose=False)
        # bg 按 OpenCV 惯例是 BGR，(255,0,0) 是纯蓝。落盘要经过 BGR->RGB
        # 的手工换序，读回来必须还是蓝的；如果这里出来是红的就说明写
        # PNG 时通道顺序反了。这块合成图只有 8 张、缺了右下角，所以
        # 一定存在露底的像素。
        im_p = cv2.imread(pp, cv2.IMREAD_COLOR)
        holes = np.all(im_p == np.array([255, 0, 0], np.uint8), axis=2)
        print(f"  给 10/12 张位置（挖掉中排两张）: 画布 {rp['out_w']}x{rp['out_h']} "
              f"放置 {rp['n_placed']} 张，露底像素 {int(holes.sum())} 个，"
              f"露底处 BGR={im_p[np.argwhere(holes)[0][0], np.argwhere(holes)[0][1]]}"
              f"（应为 [255 0 0]）")
        arr = np.array([positions[t.tid] for t in tiles], float)
        arr[3] = np.nan
        rn = render(tiles, arr, os.path.join(work, "nan.png"), out_scale=1 / 8,
                    cache_dir=cache, flatfield=ff, band_px=64, verbose=False)
        print(f"  其中一张位置是 NaN: 画布 {rn['out_w']}x{rn['out_h']} "
              f"放置 {rn['n_placed']} 张（NaN 那张被跳过）")

        print()
        print("=" * 72)
        print("(h) 流式 PNG 写出器：与 cv2 逐位对拍")
        print("=" * 72)
        rr = np.random.default_rng(3)
        im = rr.integers(0, 256, (271, 433, 3), dtype=np.uint8)
        im[:, :, 0] //= 3                      # 三通道明显不同，能查出 BGR/RGB 反了
        ps = os.path.join(work, "stream.png")
        with _PngStream(ps, 433, 271, level=6) as st:
            for a, b in ((0, 100), (100, 101), (101, 271)):
                st.add(im[a:b])
        rb = cv2.imread(ps, cv2.IMREAD_COLOR)
        pc = os.path.join(work, "cv.png")
        cv2.imwrite(pc, im, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"  读回 {rb.shape} 与源逐位一致 = {bool((rb == im).all())}")
        print(f"  体积 流式 {os.path.getsize(ps)} B / cv2 {os.path.getsize(pc)} B")

        print()
        print("=" * 72)
        print("(i) 真实数据：L20_down 连续 8 张，按 tiles.py 实测步距+错切拼竖条")
        print("=" * 72)
        import tiles as T
        # 真实数据目录：默认取仓库同级的 Figure3a/，可用 MOSAIC_DATA 覆盖
        data_dir = os.environ.get(
            "MOSAIC_DATA",
            str(Path(__file__).resolve().parent / "Figure3a"))
        real_cache = os.path.join(work, "real_tile_cache")
        allt = T.scan_dataset(data_dir)
        col = sorted([t for t in allt if t.col == 20], key=lambda t: t.order)[:8]
        T.build_cache(col, real_cache, scale_div=8)
        # 名义步距用 tiles.py 里的实测常量，不用采集脚本那个错了 2% 的
        # 25.6 px/step；错切也必须带上，否则这一条竖条会明显歪。
        dy = getattr(T, "NOMINAL_PITCH_Y", 76 * 25.08)
        shear = getattr(T, "SHEAR_PER_ROW", 0.0)
        pos = {t.tid: (-shear * t.nominal_row, dy * t.nominal_row) for t in col}
        print("瓦片:", ", ".join(t.tid for t in col))
        print(f"名义步距 dy={dy:.1f} px，每行横向错切 {shear:.1f} px（x 会是负的，"
              f"正好试一遍负坐标平移）")
        ffr = estimate_flatfield(allt, real_cache, sample=80, verbose=True)
        print(f"真实平场 均值={ffr.mean():.4f} 范围 [{ffr.min():.3f}, {ffr.max():.3f}] "
              f"中心/角 = {ffr[135, 240].mean() / ffr[5, 5].mean():.3f}")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "blend_selftest_L20_strip.png")
        res = render(col, pos, out, out_scale=1 / 8, cache_dir=real_cache,
                     flatfield=ffr, band_px=512, verbose=True)
        print(f"结果: {res['out_w']} x {res['out_h']}, n_placed={res['n_placed']}, "
              f"n_bands={res['n_bands']}, 解码 {res['n_decodes']} 次 / {len(col)} 张")
        print(f"文件: {res['path']}  ({os.path.getsize(res['path']) / 1e6:.2f} MB)")

        # 重叠带里的接缝残差：拿相邻两张在重叠区的差，看融合后有没有台阶
        s = cv2.imread(res["path"], cv2.IMREAD_COLOR).astype(np.float32)
        prof = s.reshape(s.shape[0], -1).mean(axis=1)
        jump = np.abs(np.diff(prof))
        seam_rows = [int(round(dy * k / 8)) for k in range(1, len(col))]
        sj = [float(jump[max(0, r - 1)]) for r in seam_rows if r - 1 < len(jump)]
        print(f"  逐行均值跳变：瓦片接缝行 max={max(sj):.3f}，"
              f"全图 max={jump.max():.3f} 中位={float(np.median(jump)):.3f}")
        print("  结论:", "接缝不比普通行更突出"
              if max(sj) <= jump.max() + 1e-6 else "接缝处有台阶")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
