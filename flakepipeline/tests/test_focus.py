"""对焦自检（focus_metrics 的闭式部分）：合成纹理 vs 同一纹理高斯模糊 vs 平帧，
验证指标排序、相对规则、numpy 退路与 cv2 同数、逐帧参考、以及对一个临时
文件夹的事后检查（focus.json、扫描点映射、CLI 退出码、Result 契约）。

标定记录（2026-09-02，Figure 3a 全部 920 张 3840x2160 瓦片，20x，SiO2/Si，
max_side=1024 -> 960x540）：score p0/p10/p50/p75/p100 = 4.8/12.4/15.5/19.8/72.6，
contrast p0/p50 = 7.3/21.2。参考 p75=19.8，ratio 0.4 -> 阈值 7.9，恰好抓出中央
熔池周围那 25 张全糊的瓦片（score 4.8–7.9；再往上 8.2 是熔池边缘半糊半清，
9.4 起全清）。空衬底 score >= 12（颗粒划痕是锐的），contrast 最低的反而是
虚焦片（7–11），所以 min_contrast 只留 4，只挡真正的空帧。

脚本本身不接相机（快门是人按的），所以是"扫完立刻查"（focus_check_for_run），
不是"拍完立刻判"；逐帧接口 judge_frame/RunningFocusReference 留给接了相机的版本。
不联网，不需要 KIMI_API_KEY。"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

FP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FP))

import focus_metrics as A        # noqa: E402

H, W = 1440, 2560                 # 缩到 max_side=1024 要走整数分块平均（f=3 -> 480x853）
PSF = 1.5                         # 真实光学图像本来就受 PSF 限制，没有像素级台阶；"清晰"也要先过这一道


def texture(seed, thr=0.4, h=H, w=W, bg=120.0, fg=170.0):
    """块状随机图形（锐边）：模拟衬底上的三角片。thr 越大晶体越少。返回 float 图和 rng。"""
    rng = np.random.default_rng(seed)
    low = rng.normal(size=(h // 24 + 1, w // 24 + 1))
    low = np.kron(low, np.ones((24, 24)))[:h, :w]
    return np.where(low > thr, fg, bg).astype(np.float32), rng


def add_noise(img, rng, sigma=2.0):
    """传感器噪声在光学之后加，和真实相机一样（模糊帧的噪声不模糊）。"""
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img, sigma):
    if A.cv2 is not None:
        return A.cv2.GaussianBlur(img, (0, 0), sigma)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma)


def sharp_frame(seed, thr=0.4):
    img, rng = texture(seed, thr)
    return add_noise(blur(img, PSF), rng)


def write_png(path, img):
    if A.cv2 is not None:
        assert A.cv2.imwrite(str(path), img)
    else:
        from PIL import Image
        Image.fromarray(img).save(str(path))


@pytest.fixture(scope="module")
def frames():
    img, rng = texture(0)
    sharp = add_noise(blur(img, PSF), rng)
    blurred = add_noise(blur(img, 12.0), rng)    # 明显虚焦
    sparse = sharp_frame(0, thr=0.9)             # 同倍率、清晰，只是晶体少一半：不该报
    flat = add_noise(np.full((H, W), 128.0, np.float32), rng)
    return {"sharp": sharp, "sparse": sparse, "blurred": blurred, "flat": flat}


# ------------------------------------------------------------------ 指标
def test_score_ordering(frames):
    s = {k: A.focus_score(v) for k, v in frames.items()}
    t = {k: A.tenengrad(v) for k, v in frames.items()}
    assert s["sharp"] > s["sparse"] > s["blurred"] > s["flat"] > 0
    assert t["sharp"] > t["sparse"] > t["blurred"] > t["flat"] > 0
    assert s["blurred"] < 0.1 * s["sharp"]            # 明显虚焦掉一个数量级以上
    assert s["sparse"] > 0.6 * s["sharp"]             # 内容少一半只掉三成，在相对规则的容差内


def test_prep_gray_shapes_and_dtypes(frames):
    g = A.prep_gray(frames["sharp"])
    assert g.shape == (480, 853) and g.dtype == np.float32
    small = frames["sharp"][:300, :400]
    assert A.prep_gray(small).shape == (300, 400)     # 不放大
    ref = A.focus_score(small)
    u16 = small.astype(np.uint16) * 257
    f01 = small.astype(np.float32) / 255.0
    bgr = np.stack([small, small, small], -1)
    bgra = np.dstack([bgr, np.full_like(small, 255)])
    for other in (u16, f01, bgr, bgra):
        assert A.focus_score(other) == pytest.approx(ref, rel=1e-4)
    with pytest.raises(ValueError):
        A.focus_score(np.zeros((2, 3, 4, 5)))


def test_numpy_fallback_matches_cv2(frames, monkeypatch):
    if A.cv2 is None:
        pytest.skip("cv2 不可用，无从比较")
    with_cv2 = A.focus_metrics(frames["blurred"])
    monkeypatch.setattr(A, "cv2", None)
    without = A.focus_metrics(frames["blurred"])
    assert without["score"] == pytest.approx(with_cv2["score"], rel=1e-4)
    assert without["tenengrad"] == pytest.approx(with_cv2["tenengrad"], rel=1e-4)
    assert without["contrast"] == pytest.approx(with_cv2["contrast"], rel=1e-6)


def test_deterministic(frames):
    a = A.focus_metrics(frames["sharp"])
    b = A.focus_metrics(frames["sharp"].copy())
    assert a == b


# ------------------------------------------------------------------ 规则
def test_relative_rule_and_floor(frames):
    s_sharp = A.focus_score(frames["sharp"])
    s_blur = A.focus_score(frames["blurred"])
    s_sparse = A.focus_score(frames["sparse"])
    assert A.is_defocused(s_blur, reference=s_sharp)
    assert not A.is_defocused(s_sparse, reference=s_sharp)
    assert not A.is_defocused(s_sharp, reference=s_sharp)
    assert s_blur > A.FocusConfig.floor                # 这张只有相对规则能抓，绝对下限抓不到
    # 没有参考时只看绝对下限
    assert A.is_defocused(A.FocusConfig.floor - 0.5, reference=None)
    assert not A.is_defocused(50.0, reference=None)
    assert A.is_defocused(float("nan"), reference=100.0)
    # 参数可调：比例放宽到 1.0 时同一张 sharp 仍不报（score == reference 不算低于）
    assert not A.is_defocused(s_sharp, reference=s_sharp, ratio=1.0)
    assert A.is_defocused(s_sharp * 0.9, reference=s_sharp, ratio=1.0)


def test_focus_decision_three_states(frames):
    cfg = A.FocusConfig()
    m = {k: A.focus_metrics(v, cfg) for k, v in frames.items()}
    ref = m["sharp"]["score"]
    assert A.focus_decision(m["sharp"]["score"], m["sharp"]["contrast"], ref, cfg) == ("ok", "")
    assert A.focus_decision(m["sparse"]["score"], m["sparse"]["contrast"], ref, cfg) == ("ok", "")
    d, why = A.focus_decision(m["blurred"]["score"], m["blurred"]["contrast"], ref, cfg)
    assert d == "defocused" and "reference" in why
    d, why = A.focus_decision(m["flat"]["score"], m["flat"]["contrast"], ref, cfg)
    assert d == "low_texture" and "contrast" in why
    assert m["flat"]["contrast"] < cfg.min_contrast < m["blurred"]["contrast"]
    d, why = A.focus_decision(cfg.floor - 1, 30.0, None, cfg)
    assert d == "defocused" and "floor" in why
    assert A.focus_decision(cfg.floor + 1, 30.0, None, cfg)[0] == "ok"


def test_focus_reference_percentile():
    cfg = A.FocusConfig(min_ref_tiles=3, ref_percentile=75.0)
    assert A.focus_reference([10.0, 20.0], cfg) is None
    assert A.focus_reference([1, 2, 3, 4, 100], cfg) == pytest.approx(4.0)
    assert A.focus_reference([5, float("nan"), None, 7, 9], cfg) == pytest.approx(8.0)


def test_running_reference_and_judge_frame(frames):
    cfg = A.FocusConfig()
    ref = A.RunningFocusReference(n=5)
    first = A.judge_frame(frames["sharp"], ref, cfg)
    assert first["decision"] == "ok" and first["reference"] is None and first["focus_ok"]
    for seed in (1, 2):
        assert A.judge_frame(sharp_frame(seed), ref, cfg)["decision"] == "ok"
    assert len(ref.scores) == 3 and ref.value == pytest.approx(float(np.median(ref.scores)))
    before = ref.value
    bad = A.judge_frame(frames["blurred"], ref, cfg)
    assert bad["decision"] == "defocused" and not bad["focus_ok"]
    assert bad["reference"] == pytest.approx(before) and ref.value == pytest.approx(before)
    empty = A.judge_frame(frames["flat"], ref, cfg)
    assert empty["decision"] == "low_texture" and empty["focus_ok"] and ref.value == pytest.approx(before)
    assert A.judge_frame(frames["sparse"], ref, cfg)["decision"] == "ok"
    # 用上一次的 reference 做种子，第一帧就能判
    seeded = A.RunningFocusReference(n=3, reference=before)
    assert A.judge_frame(frames["blurred"], seeded, cfg)["decision"] == "defocused"
    # 窗口长度
    for _ in range(10):
        seeded.accept(1.0)
    assert len(seeded.scores) == 3


# ------------------------------------------------------------------ 事后检查
def make_folder(root, frames, n_sharp=4, blurred="tile_05.png", flat="tile_06.png"):
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for k in range(n_sharp):
        p = root / f"tile_{k + 1:02d}.png"
        write_png(p, sharp_frame(10 + k)); names.append(p.name)
    write_png(root / blurred, frames["blurred"]); names.append(blurred)
    write_png(root / flat, frames["flat"]); names.append(flat)
    (root / "notes.txt").write_text("not a tile")
    (root / "broken.png").write_bytes(b"\x89PNG not really")
    return names


def test_check_focus_folder(tmp_path, frames):
    root = tmp_path / "tiles"
    make_folder(root, frames)
    rep = A.check_focus_folder(root)
    assert rep["available"] and rep["out_path"] == str(root / "focus.json")
    assert rep["counts"] == {"total": 7, "ok": 4, "defocused": 1, "low_texture": 1, "error": 1}
    # 糊掉的那张 contrast 仍 >= min_contrast，所以参考池是 5 张（4 清晰 + 1 糊）——
    # 这正是取 p75 而不是中位数的原因：糊片混进池里也拖不低参考值
    assert rep["reference_source"].startswith("p75 of 5 textured tiles")
    dec = {t["file"]: t["decision"] for t in rep["tiles"]}
    assert dec == {"broken.png": "error", "tile_01.png": "ok", "tile_02.png": "ok", "tile_03.png": "ok",
                   "tile_04.png": "ok", "tile_05.png": "defocused", "tile_06.png": "low_texture"}
    assert [t["file"] for t in rep["tiles"]] == sorted(dec)           # order="name"
    assert [r["file"] for r in rep["retake"]] == ["tile_05.png"]
    assert "需补拍: tile_05.png" in rep["summary"] and "虚焦 1" in rep["summary"]
    assert rep["warnings"] == []
    # focus.json 是同一份东西，逐张分数与内存路径一致（PNG 无损）
    disk = json.loads((root / "focus.json").read_text(encoding="utf-8"))
    assert disk["counts"] == rep["counts"] and disk["config"] == A.FocusConfig().to_dict()
    t05 = next(t for t in disk["tiles"] if t["file"] == "tile_05.png")
    assert t05["score"] == pytest.approx(A.focus_score(frames["blurred"]), rel=1e-6)
    assert t05["tenengrad"] > 0 and t05["contrast"] > 0 and t05["shape"] == [480, 853]
    # 给定参考值就不从这批里估
    rep2 = A.check_focus_folder(root, reference=1e6, out_path=str(tmp_path / "given.json"))
    assert rep2["reference_source"] == "given" and rep2["counts"]["defocused"] == 5
    assert rep2["counts"]["low_texture"] == 1                          # 平帧仍是 low_texture
    assert (tmp_path / "given.json").exists()
    # 参考值太低会警告（比如整批都糊）
    rep3 = A.check_focus_folder(root, cfg=A.FocusConfig(ref_warn_below=1e9))
    assert rep3["warnings"] and "整批" in rep3["warnings"][0] and "警告" in rep3["summary"]


def test_check_focus_folder_edge_cases(tmp_path, frames, monkeypatch):
    missing = A.check_focus_folder(tmp_path / "nope")
    assert missing["counts"]["total"] == 0 and "不存在" in missing["summary"] and missing["out_path"] is None
    empty = tmp_path / "empty"; empty.mkdir()
    rep = A.check_focus_folder(empty)
    assert rep["counts"]["total"] == 0 and rep["reference"] is None and (empty / "focus.json").exists()
    # 有纹理帧不够 min_ref_tiles：没有参考值，只用绝对下限，并警告
    two = tmp_path / "two"; two.mkdir()
    write_png(two / "a.png", frames["sharp"]); write_png(two / "b.png", frames["blurred"])
    rep = A.check_focus_folder(two)
    assert rep["reference"] is None and rep["reference_source"].startswith("none")
    assert rep["counts"]["defocused"] == 0 and "没有参考值" in rep["summary"]
    # 没有 numpy：不抛，报告标 available=False，Result 上报
    monkeypatch.setattr(A, "np", None)
    rep = A.check_focus_folder(two)
    assert rep["available"] is False and "numpy" in rep["summary"]
    res = A.focus_as_result(rep)
    assert res.ok is False and res.escalate and "numpy" in res.escalate_reason


def test_focus_check_for_run_maps_points(tmp_path, frames):
    root = tmp_path / "camera"
    names = make_folder(root, frames)                       # tile_01..04 清晰, 05 糊, 06 平
    os.remove(root / "broken.png")
    since = time.time() - 1000.0
    # 扫描开始前就在文件夹里的旧照片不该被算进来
    write_png(root / "old.png", frames["sharp"])
    os.utime(root / "old.png", (since - 50, since - 50))
    for k, n in enumerate(names):                            # 拍摄先后 = mtime 先后
        os.utime(root / n, (since + 10 * (k + 1), since + 10 * (k + 1)))
    run = {"points": [{"i": k + 1, "x_steps": 10 * k, "y_steps": 3, "t": ""} for k in range(6)]}
    info = A.focus_check_for_run(run, str(root), since)
    assert info["counts"]["total"] == 6 and info["point_map"] == "1:1 by mtime order"
    assert info["retake"] == [{"file": "tile_05.png", "score": pytest.approx(A.focus_score(frames["blurred"]), rel=1e-6),
                               "point_i": 5, "x_steps": 40, "y_steps": 3}]
    assert info["problem"].startswith("1 张虚焦需补拍") and "tile_05.png(点 5)" in info["problem"]
    assert os.path.exists(info["focus_json"])
    disk = json.loads(Path(info["focus_json"]).read_text(encoding="utf-8"))
    assert [t["file"] for t in disk["tiles"]] == names and disk["tiles"][0]["point_i"] == 1
    # 张数对不上就不硬对
    run["points"].append({"i": 7, "x_steps": 0, "y_steps": 0, "t": ""})
    info = A.focus_check_for_run(run, str(root), since)
    assert info["point_map"].startswith("none") and info["retake"][0]["point_i"] is None
    # 扫描期间没有新文件：相机存图路径填错了，这要算自检问题
    info = A.focus_check_for_run(run, str(root), time.time() + 3600)
    assert info["counts"]["total"] == 0 and "没有新文件" in info["problem"]
    # 只有清晰帧：没有 problem
    clean = tmp_path / "clean"; clean.mkdir()
    for k in range(3):
        write_png(clean / f"c{k}.png", sharp_frame(20 + k))
    info = A.focus_check_for_run({"points": []}, str(clean), None)
    assert info["problem"] is None and info["counts"]["ok"] == 3


def test_focus_as_result_contract(tmp_path, frames):
    root = tmp_path / "tiles"
    make_folder(root, frames)
    res = A.focus_as_result(A.check_focus_folder(root))
    assert res.ok is False and res.escalate and "1 张虚焦" in res.escalate_reason
    assert res.data["counts"]["defocused"] == 1 and res.evidence
    clean = tmp_path / "clean"; clean.mkdir()
    for k in range(3):
        write_png(clean / f"c{k}.png", sharp_frame(30 + k))
    res = A.focus_as_result(A.check_focus_folder(clean))
    assert res.ok and not res.escalate and res.confidence == 1.0


def test_cli_check_focus(tmp_path, frames, capsys):
    root = tmp_path / "tiles"
    make_folder(root, frames)
    out = tmp_path / "f.json"
    assert A.main(["--check-focus", str(root), "--focus-out", str(out)]) == 1
    assert out.exists() and "需补拍: tile_05.png" in capsys.readouterr().out
    # 阈值可从命令行改：比例调到 0 就只剩绝对下限，报不出来
    assert A.main(["--check-focus", str(root), "--focus-ratio", "0", "--focus-out", str(out)]) == 0
    empty = tmp_path / "empty"; empty.mkdir()
    assert A.main(["--check-focus", str(empty)]) == 2
    assert A.main(["--check-focus", str(tmp_path / "missing")]) == 2
