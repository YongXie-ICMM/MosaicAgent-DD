"""Offline focus metrics and image-folder checks for MosaicAgent-DD.

The numerical routines are extracted without changing their calculations from
MosaicAgent's archived focus checker. This module does not import a GUI, camera
or motion controller. Historical thresholds are adjustable settings, not a
validation of a new camera mode or sample.
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MTIME_SLACK_S = 2.0            # exFAT/FAT 的 mtime 精度是 2 s，按 since 筛文件时留这个余量


@dataclass
class FocusConfig:
    """对焦判定的全部可调参数。数字的尺度：8-bit 灰度、最长边缩到 max_side 像素。

    默认值来自 2026-09-02 对 Figure 3a 全部 920 张瓦片（3840x2160，20x，SiO2/Si）
    的实测分布：score p0/p10/p50/p75/p100 = 4.8/12.4/15.5/19.8/72.6。参考 p75=19.8
    乘 ratio 0.4 得阈值 7.9，恰好抓出中央熔池周围那 25 张全糊的瓦片（score
    4.8–7.9；再往上 8.2 是熔池边缘半糊半清，9.4 起全清）——就是发表那次事后
    才被 tile_inspector 抓出来、只能救回的那 25 张。floor 故意留得很低（这套
    显微镜上 8 就能单独分开，但换蓝宝石衬底对比度会低），跨设置靠相对规则。
    换物镜或换相机时先用 --check-focus 跑一批已知清晰的瓦片，把 focus.json 里的
    reference 记下来，下次用 --focus-ref 喂回去。
    """
    max_side: int = 1024          # 打分前把最长边缩到这个像素数（3840 -> 960）
    floor: float = 4.0            # 绝对下限：score 低于它一律判虚焦（本显微镜 20x 可设 8）
    ratio: float = 0.4            # 相对规则：score < ratio * reference 判虚焦
    min_contrast: float = 4.0     # 灰度标准差低于它 -> low_texture（空帧），不判虚焦
    ref_percentile: float = 75.0  # 参考值 = 有纹理帧 score 的这个百分位
    min_ref_tiles: int = 3        # 有纹理帧少于这个数就不算参考值，只用绝对下限
    ref_warn_below: float = 10.0  # 参考值低于它 -> 警告整批偏糊/低对比度（清晰批约 20）
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def to_dict(self):
        d = asdict(self)
        d["extensions"] = list(self.extensions)
        return d


def _require_np():
    if np is None:
        raise RuntimeError("对焦自检需要 numpy（pip install numpy）；走位功能不受影响")


def to_gray8(img):
    """任意 BGR/BGRA/灰度、uint8/uint16/float 图 -> float32 灰度，0..255 尺度。

    16-bit 整数按位深缩到 255；float 若最大值 <= 1 视为 0..1 也缩到 255，
    否则原样使用。这样不同位深的相机给出的 score 才能互相比较。"""
    _require_np()
    src = np.asarray(img)
    if src.ndim == 3:
        if src.shape[2] >= 3:
            b = src[..., 0].astype(np.float32)
            g = src[..., 1].astype(np.float32)
            r = src[..., 2].astype(np.float32)
            a = 0.114 * b + 0.587 * g + 0.299 * r
        else:
            a = src[..., 0].astype(np.float32)
    elif src.ndim == 2:
        a = src.astype(np.float32)
    else:
        raise ValueError(f"expected HxW or HxWxC image, got shape {src.shape}")
    if np.issubdtype(src.dtype, np.integer) and src.dtype.itemsize > 1:
        a *= 255.0 / float(np.iinfo(src.dtype).max)
    elif np.issubdtype(src.dtype, np.floating) and a.size and float(a.max()) <= 1.0:
        a *= 255.0
    return a


def downsample_gray(gray, max_side=1024):
    """整数倍分块平均，把最长边缩到 <= max_side。

    不用 cv2.resize：INTER_AREA 输出 uint8 会四舍五入，而且没有 cv2 的机器算出
    来的数就不一样。分块平均在哪台机器上都是同一个数。"""
    _require_np()
    g = np.asarray(gray, np.float32)
    h, w = g.shape
    f = int(math.ceil(max(h, w) / float(max_side)))
    if f <= 1:
        return g
    h2, w2 = (h // f) * f, (w // f) * f
    return g[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def prep_gray(img, max_side=1024):
    """打分前的统一预处理：灰度 + 缩放。"""
    return downsample_gray(to_gray8(img), max_side)


def _laplacian(g):
    g = np.ascontiguousarray(g, dtype=np.float32)
    if cv2 is not None:
        # ksize=1 就是 3x3 的 [[0,1,0],[1,-4,1],[0,1,0]]，边界 BORDER_REFLECT_101
        return cv2.Laplacian(g, cv2.CV_32F)
    p = np.pad(g, 1, mode="reflect")          # 与 cv2 的 BORDER_REFLECT_101 相同
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * g


def _sobel_sq(g):
    g = np.ascontiguousarray(g, dtype=np.float32)
    if cv2 is not None:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    else:
        p = np.pad(g, 1, mode="reflect")
        gx = ((p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:])
              - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2]))
        gy = ((p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
              - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:]))
    return gx * gx + gy * gy


def focus_score(img, max_side=1024):
    """Laplacian 方差：对焦锐度的闭式指标，越大越锐。img 是 BGR/BGRA 或灰度。"""
    return float(np.var(_laplacian(prep_gray(img, max_side))))


def tenengrad(img, max_side=1024):
    """Sobel 梯度平方的均值，第二个锐度指标（对噪声不如 Laplacian 方差敏感）。"""
    return float(np.mean(_sobel_sq(prep_gray(img, max_side))))


def focus_metrics(img, cfg=None):
    """一次预处理算齐三个量：score / tenengrad / contrast，外加缩放后的尺寸。"""
    cfg = cfg or FocusConfig()
    g = prep_gray(img, cfg.max_side)
    return {"score": float(np.var(_laplacian(g))),
            "tenengrad": float(np.mean(_sobel_sq(g))),
            "contrast": float(np.std(g)),
            "shape": [int(g.shape[0]), int(g.shape[1])]}


def is_defocused(score, reference=None, floor=FocusConfig.floor, ratio=FocusConfig.ratio):
    """绝对下限 + 相对参考值。reference 为 None（还没有参考）时只看下限。"""
    score = float(score)
    if not math.isfinite(score) or score < floor:
        return True
    if reference is not None:
        reference = float(reference)
        if math.isfinite(reference) and reference > 0 and score < ratio * reference:
            return True
    return False


def focus_reference(scores, cfg=None):
    """有纹理帧的 score 取 ref_percentile 百分位；不够 min_ref_tiles 张就没有参考值。"""
    cfg = cfg or FocusConfig()
    s = [float(v) for v in scores if v is not None and math.isfinite(float(v))]
    if len(s) < cfg.min_ref_tiles:
        return None
    return float(np.percentile(s, cfg.ref_percentile))


def focus_decision(score, contrast, reference, cfg=None):
    """(decision, reason)，decision ∈ {ok, defocused, low_texture}。规则见上面的注释。"""
    cfg = cfg or FocusConfig()
    score, contrast = float(score), float(contrast)
    if not math.isfinite(contrast) or contrast < cfg.min_contrast:
        return "low_texture", f"contrast {contrast:.1f} < {cfg.min_contrast:g}"
    if is_defocused(score, reference, cfg.floor, cfg.ratio):
        if not math.isfinite(score) or score < cfg.floor:
            return "defocused", f"score {score:.1f} < floor {cfg.floor:g}"
        return "defocused", f"score {score:.1f} < {cfg.ratio:g} x reference {float(reference):.1f}"
    return "ok", ""


class RunningFocusReference:
    """边拍边判的参考值：最近 n 张判为 ok 的帧的 score 中位数。

    这是给**能拿到相机帧**的采集程序用的接口（每拍一张调一次 judge_frame；
    判虚焦就重拍，最多 K 次）。本脚本快门是人按的，拿不到帧，所以它在这里
    没有调用方，只在 tests/test_focus.py 里验证规则本身。reference 可以用上一次
    focus.json 里的值做种子，这样第一张就有参考。"""

    def __init__(self, n=10, reference=None):
        self.n = int(n)
        self.scores = []
        self.seed = None if reference is None else float(reference)

    @property
    def value(self):
        if self.scores:
            return float(np.median(self.scores))
        return self.seed

    def accept(self, score):
        self.scores.append(float(score))
        del self.scores[:-self.n]


def judge_frame(img, ref, cfg=None):
    """一帧的判定：算指标 -> focus_decision -> ok 的帧进参考。

    返回 dict：focus_score / tenengrad / contrast / decision / reason /
    focus_ok（不是 defocused 就算 ok，low_texture 也放行）/ reference（判定时用的）。"""
    cfg = cfg or FocusConfig()
    m = focus_metrics(img, cfg)
    used = ref.value
    d, why = focus_decision(m["score"], m["contrast"], used, cfg)
    if d == "ok":
        ref.accept(m["score"])
    return {"focus_score": m["score"], "tenengrad": m["tenengrad"], "contrast": m["contrast"],
            "decision": d, "reason": why, "focus_ok": d != "defocused", "reference": used}


def load_image(path):
    """读一张瓦片成 BGR；读不出来返回 None。

    用 np.fromfile + imdecode 而不是 cv2.imread：Windows 上 imread 读不了带中文的
    路径，而采集电脑上的文件夹名很可能有中文。没有 cv2 就退到 PIL。"""
    _require_np()
    if cv2 is not None:
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            a = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        except Exception:
            a = None
        if a is not None:
            return a
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))[..., ::-1].copy()
    except Exception:
        return None


def list_tiles(folder, cfg=None, since=None, order="name"):
    """文件夹里的瓦片路径。since（time.time() 时间戳）给了就只要之后新出现的文件；
    order="mtime" 按拍摄先后排（对扫描点用），"name" 按文件名排（离线复查用）。"""
    cfg = cfg or FocusConfig()
    rows = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p) or os.path.splitext(name)[1].lower() not in cfg.extensions:
            continue
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if since is not None and m < float(since) - MTIME_SLACK_S:
            continue
        rows.append((m, name, p) if order == "mtime" else (name, m, p))
    rows.sort()
    return [r[2] for r in rows]


def focus_summary_text(report):
    """一行人能读的总结，放进 GUI 弹窗、命令行输出和扫描留档。"""
    if not report.get("available"):
        return report.get("summary") or "对焦自检不可用"
    c = report["counts"]
    parts = [f"{c['total']} 张：ok {c['ok']}，虚焦 {c['defocused']}，"
             f"无纹理 {c['low_texture']}，读不出 {c['error']}"]
    ref = report.get("reference")
    if ref is not None:
        parts.append(f"参考 score {ref:.1f}（{report.get('reference_source')}）")
    else:
        parts.append(f"没有参考值（{report.get('reference_source')}）")
    if report.get("retake"):
        items = []
        for r in report["retake"][:8]:
            loc = (f"(点 {r['point_i']}, X={r['x_steps']}, Y={r['y_steps']})"
                   if r.get("point_i") is not None else "")
            items.append(f"{r['file']}{loc} score={r['score']:.1f}")
        more = f" ... 共 {len(report['retake'])} 张" if len(report["retake"]) > 8 else ""
        parts.append("需补拍: " + ", ".join(items) + more)
    for w in report.get("warnings") or []:
        parts.append("警告: " + w)
    return "；".join(parts)


def _write_focus_report(report, folder, out_path):
    path = out_path or (os.path.join(folder, "focus.json") if os.path.isdir(folder) else None)
    report["out_path"] = None
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            report["out_path"] = os.path.abspath(path)
        except OSError as e:
            report["write_error"] = str(e)
    return report


def check_focus_folder(folder, cfg=None, out_path=None, reference=None, since=None,
                       points=None, order="name"):
    """对一个文件夹里的瓦片逐张打分、判定，写 focus.json，返回报告 dict。

    reference 给了就用给的（比如上一次 focus.json 里的值），否则从这批有纹理的
    瓦片里按 cfg.ref_percentile 估。since 给了就只看之后新出现的文件。points
    给了（扫描留档里的 points 列表）且张数对得上，就把每张瓦片按顺序对到扫描点，
    虚焦的连坐标一起列进 retake。没有 numpy 时返回 available=False 的报告，不抛。
    """
    cfg = cfg or FocusConfig()
    folder = os.path.abspath(folder)
    report = {
        "folder": folder,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "metric": f"laplacian_variance(gray8, block-mean to max_side={cfg.max_side})",
        "backend": "cv2" if cv2 is not None else ("numpy" if np is not None else "none"),
        "available": np is not None,
        "config": cfg.to_dict(),
        "since": (datetime.fromtimestamp(float(since)).isoformat(timespec="seconds")
                  if since is not None else None),
        "order": order,
        "reference": None, "reference_source": None, "point_map": None,
        "warnings": [], "tiles": [], "retake": [],
        "counts": {"total": 0, "ok": 0, "defocused": 0, "low_texture": 0, "error": 0},
        "summary": "",
    }
    if np is None:
        report["summary"] = "numpy 不可用，没做对焦自检（pip install numpy）"
        return _write_focus_report(report, folder, out_path)
    if not os.path.isdir(folder):
        report["summary"] = f"文件夹不存在: {folder}"
        return _write_focus_report(report, folder, out_path)

    tiles = []
    for p in list_tiles(folder, cfg, since, order):
        rec = {"file": os.path.basename(p),
               "mtime": datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
               "score": None, "tenengrad": None, "contrast": None,
               "decision": "error", "reason": "unreadable"}
        img = load_image(p)
        if img is not None:
            rec.update(focus_metrics(img, cfg))
            rec["decision"], rec["reason"] = None, ""
        tiles.append(rec)

    textured = [t["score"] for t in tiles
                if t["score"] is not None and t["contrast"] >= cfg.min_contrast]
    if reference is not None:
        ref, src = float(reference), "given"
    else:
        ref = focus_reference(textured, cfg)
        src = (f"p{cfg.ref_percentile:g} of {len(textured)} textured tiles" if ref is not None
               else f"none: {len(textured)} textured tiles < min_ref_tiles={cfg.min_ref_tiles}, floor only")
    if ref is not None and ref < cfg.ref_warn_below:
        report["warnings"].append(
            f"参考 score {ref:.1f} 低于 {cfg.ref_warn_below:g}：整批都偏糊（焦面漂了？）"
            f"或衬底本身对比度低，相对规则抓不到，抽几张肉眼看")
    elif ref is None and len(tiles) - report["counts"]["error"] > 0 and tiles:
        report["warnings"].append("没有参考值，只用了绝对下限")

    mapped = False
    if points is not None:
        mapped = bool(tiles) and len(points) == len(tiles)
        report["point_map"] = (f"1:1 by {order} order" if mapped
                               else f"none ({len(tiles)} tiles vs {len(points)} points)")
    for k, t in enumerate(tiles):
        if mapped:
            pt = points[k]
            t["point_i"], t["x_steps"], t["y_steps"] = pt.get("i"), pt.get("x_steps"), pt.get("y_steps")
        if t["score"] is not None:
            t["decision"], t["reason"] = focus_decision(t["score"], t["contrast"], ref, cfg)

    counts = report["counts"]
    counts["total"] = len(tiles)
    for t in tiles:
        counts[t["decision"]] += 1
    report["retake"] = [{k: t.get(k) for k in ("file", "score", "point_i", "x_steps", "y_steps")}
                        for t in tiles if t["decision"] == "defocused"]
    report.update(tiles=tiles, reference=ref, reference_source=src)
    report["summary"] = focus_summary_text(report)
    return _write_focus_report(report, folder, out_path)


def focus_check_for_run(run, tile_dir, since_ts, cfg=None):
    """start_auto 的自检阶段调用：只看扫描开始后新出现的文件，按拍摄先后对到扫描点。

    返回写进 run["self_check"]["focus"] 的精简 dict（逐张明细在 focus.json 里）；
    有虚焦、有读不出、或者扫描期间根本没有新文件（相机存图路径填错了）时
    problem 非空，扫描自检就不算通过。"""
    rep = check_focus_folder(tile_dir, cfg, since=since_ts, points=run.get("points"), order="mtime")
    c = rep["counts"]
    info = {"tile_dir": rep["folder"], "focus_json": rep.get("out_path"), "counts": c,
            "reference": rep["reference"], "reference_source": rep["reference_source"],
            "point_map": rep["point_map"], "retake": rep["retake"],
            "warnings": rep["warnings"], "summary": rep["summary"], "problem": None}
    if not rep["available"]:
        info["note"] = rep["summary"]
    elif c["total"] == 0:
        info["problem"] = "扫描期间瓦片文件夹里没有新文件（相机存图路径填对了吗？）"
    elif c["defocused"] or c["error"]:
        names = [(f"{r['file']}(点 {r['point_i']})" if r.get("point_i") is not None else r["file"])
                 for r in rep["retake"][:8]]
        more = f" ... 共 {len(rep['retake'])} 张" if len(rep["retake"]) > 8 else ""
        msg = f"{c['defocused']} 张虚焦需补拍"
        if c["error"]:
            msg += f"，{c['error']} 张读不出"
        if names:
            msg += ": " + ", ".join(names) + more
        info["problem"] = msg
    return info


def focus_as_result(report):
    """把 focus.json 的报告包成 agents.Result，给编排层用（自检 + 上报的统一契约）。"""
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from agents import Result
    c = report.get("counts", {})
    n_def, n_err = c.get("defocused", 0), c.get("error", 0)
    if not report.get("available"):
        return Result(ok=False, data=report, confidence=0.0, evidence=report.get("summary", ""),
                      escalate=True, escalate_reason="对焦自检不可用（没有 numpy）")
    ok = n_def == 0 and n_err == 0
    warnings = report.get("warnings") or []
    conf = 1.0 if (report.get("reference") is not None and not warnings) else 0.5
    reasons = [] if ok else [f"{n_def} 张虚焦需补拍，{n_err} 张读不出"]
    reasons += warnings
    return Result(ok=ok, data=report, confidence=conf, evidence=report.get("summary", ""),
                  escalate=bool(reasons), escalate_reason="；".join(reasons))



def _build_argparser():
    ap = argparse.ArgumentParser(description="Offline focus checks; no camera or stage control.")
    ap.add_argument("--check-focus", required=True, metavar="TILE_DIR",
                    help="Inspect saved images and write focus.json.")
    ap.add_argument("--focus-out", metavar="PATH", help="Output JSON (default: TILE_DIR/focus.json).")
    ap.add_argument("--focus-ref", type=float, metavar="SCORE", help="Use a supplied reference score.")
    ap.add_argument("--focus-floor", type=float, default=FocusConfig.floor)
    ap.add_argument("--focus-ratio", type=float, default=FocusConfig.ratio)
    ap.add_argument("--min-contrast", type=float, default=FocusConfig.min_contrast)
    ap.add_argument("--order", choices=("name", "mtime"), default="name")
    return ap


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    cfg = FocusConfig(floor=args.focus_floor, ratio=args.focus_ratio, min_contrast=args.min_contrast)
    rep = check_focus_folder(args.check_focus, cfg, out_path=args.focus_out,
                             reference=args.focus_ref, order=args.order)
    print(rep["summary"])
    print(f"focus.json -> {rep.get('out_path')}")
    c = rep["counts"]
    checked = c.get("ok", 0) + c.get("defocused", 0) + c.get("low_texture", 0)
    if not rep["available"] or c.get("total", 0) == 0 or checked == 0:
        return 2
    return 1 if c.get("defocused", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
