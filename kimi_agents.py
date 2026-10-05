# -*- coding: utf-8 -*-
"""kimi_agents.py — 拼接流水线里的 Kimi 智能体池。

为什么需要它
------------
纯统计判据在这个任务上会系统性误判。实测：L24_down/0022-0028、
L25_up/0027-0032、L28_down/0023-0029 这些瓦片 focus 只有 3-10，
文件也明显偏小，纯按阈值会被判成"坏片"。但它们其实是**没有晶体的
裸衬底**——无特征的图 PNG 本来就压得小、Laplacian 方差本来就低。
真正该丢的虚焦片和空帧，统计量长得和它们一模一样。这个区别只能看图。

所以这里把"看图做判断"的环节交给 Kimi，并且照搬多智能体的做法：
同一个问题独立问 N 次、投票取多数（vote()），而不是信单次输出。
LLM 在边界样本上会摇摆，投票能把摇摆压下去。

设计原则
--------
1. **降级而不是崩溃**。没有 API key、网络不通、模型没配好，一律回退到
   纯统计判据，流水线继续跑完，只在报告里注明"AI 环节被跳过"。
   学生手上没 key 也要能出图。
2. **幂等**。每次调用按 (角色, 模型, 输入指纹) 落盘缓存，重跑不重复花钱。
3. **可移植**。GridPlanner 让学生把工具指向一个全新的数据目录，
   由 Kimi 从文件名清单里反推网格布局，不用改代码。

Kimi 的两个坑（实测踩过）
-------------------------
  - kimi-k3 / kimi-k2.x 只接受 temperature=1，传别的值直接 400。
    moonshot-v1-* 系列没有这个限制。所以温度按模型名分流。
  - 投票需要多样性，但 k3 温度锁死在 1，改不了。所以多样性靠给每个
    投票者不同的 seed 提示语（"你是第 k 位独立评审"）来制造，
    而不是靠调温度。

Configuration: explicit environment variables, MOSAIC_ENV, or this repository's .env.
  KIMI_API_KEY / KIMI_BASE_URL / KIMI_MODEL / KIMI_VISION_MODEL
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Credentials are explicit and local to this project. Do not inspect a user's
# home directory, Desktop, Downloads, working directory or knowledge base.
def _env_candidates() -> list[Path]:
    """Read an explicitly selected MOSAIC_ENV file, then this repo's .env only."""
    paths = []
    if os.environ.get("MOSAIC_ENV"):
        paths.append(Path(os.environ["MOSAIC_ENV"]).expanduser())
    paths.append(Path(__file__).resolve().parent / ".env")
    return list(dict.fromkeys(paths))


# 注意：不在这里求值。import 时就锁死的话，调用方在 import 之后
# 才设置 MOSAIC_ENV 就不生效了。load_env() 每次现算。
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k3"          # 实测支持图像输入
FIXED_TEMP_PREFIXES = ("kimi-",)   # 这些模型只吃 temperature=1；也是推理模型（思维链算 completion token）
# 推理模型的 max_tokens 下限。kimi-k3 的思维链也算 completion token，4096/8192 的预算
# 经常在内容还没写出来时就 finish_reason='length'（光谱实验：23 次里 8 次截断后重发）。
# 与其截断再重试（每次重试都是一整轮推理往返），不如一开始就给够。
REASONING_MIN_TOKENS = 16384
# 429 的退避：5, 10, 20, 40 s。限流不是瞬时故障，2^n*3 的短退避只会再撞一次同一个限。
RATE_LIMIT_BACKOFF_S = 5.0


def load_env() -> dict:
    env = {}
    for p in _env_candidates():
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    for k in ("KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL", "KIMI_VISION_MODEL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


# ------------------------------------------------------------------ 客户端
@dataclass
class Usage:
    calls: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: int = 0
    failed: int = 0

    def __str__(self):
        return (f"Kimi 调用 {self.calls} 次（缓存命中 {self.cached}，失败 {self.failed}），"
                f"token 入 {self.prompt_tokens} 出 {self.completion_tokens}")


class KimiClient:
    """OpenAI 兼容的 Moonshot 客户端。只用标准库发请求，避免给学生增加依赖。"""

    def __init__(self, model: str | None = None, cache_dir: str | None = None,
                 timeout: int = 180, max_retries: int = 5):
        env = load_env()
        self.key = env.get("KIMI_API_KEY", "")
        self.base = env.get("KIMI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        # 注意优先级：仓库 .env 里的 KIMI_MODEL 是给纯文本工具用的
        # （当前是 moonshot-v1-128k，不支持图像输入）。这个流水线必须看图，
        # 所以不继承它，改用专门的 KIMI_VISION_MODEL，缺省 kimi-k3。
        self.model = model or env.get("KIMI_VISION_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _temp(self) -> float:
        return 1.0 if self.model.startswith(FIXED_TEMP_PREFIXES) else 0.2

    def _is_reasoning(self) -> bool:
        return self.model.startswith(FIXED_TEMP_PREFIXES)

    def _cache_key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def chat(self, text: str, images: list[np.ndarray] | None = None,
             system: str | None = None, max_tokens: int = 2048,
             want_json: bool = True) -> dict | str | None:
        """发一次请求。images 是 BGR uint8 数组列表，内部压成 JPEG 再走 base64。
        返回解析好的 dict（want_json）或原始字符串；彻底失败返回 None。"""
        if not self.available:
            return None

        content: list = [{"type": "text", "text": text}]
        for im in images or []:
            content.append({"type": "image_url",
                            "image_url": {"url": _to_data_uri(im)}})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": self._temp()}

        # 缓存键按调用方要的预算算，在下面的下限提升之前：预算只是传输层细节，
        # 同一个问题不该因为预算从 8192 变 16384 就重新掷一次 temperature=1 的骰子
        # （截断重试早就是这么做的：提到 16384 重发，结果仍记在原键下）。
        ck = self._cache_key(payload)
        cfile = self.cache_dir / f"{ck}.json" if self.cache_dir else None
        if cfile and cfile.exists():
            self.usage.cached += 1
            raw = json.loads(cfile.read_text(encoding="utf-8"))["raw"]
            return _extract_json(raw) if want_json else raw
        if self._is_reasoning() and int(payload["max_tokens"]) < REASONING_MIN_TOKENS:
            payload["max_tokens"] = REASONING_MIN_TOKENS

        raw = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.base + "/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                ch = d["choices"][0]
                raw = ch["message"].get("content") or ""
                # kimi-k3 是推理模型：思维链也算 completion token。预算不够时
                # content 会是空串而 finish_reason='length'，静默失败很难查，
                # 所以这里显式报出来并加倍预算重试一次。
                if not raw.strip() and ch.get("finish_reason") == "length":
                    self.usage.truncated += 1
                    if (attempt < self.max_retries - 1
                            and int(payload["max_tokens"]) < REASONING_MIN_TOKENS):
                        # 一次拉到顶，不要 3 倍 3 倍地爬——每爬一次就是一整轮
                        # 推理往返，无人值守时这是最大的时间浪费来源。
                        payload["max_tokens"] = REASONING_MIN_TOKENS
                        print(f"    [kimi] 输出被 max_tokens 截断（思维链吃光了），"
                              f"直接提到 {REASONING_MIN_TOKENS} 重试")
                        continue
                    # 已经在顶了还被截断：再发一次只会再烧一整轮推理。当失败处理，
                    # 不写缓存——空回复一旦落盘，这个问题以后每次运行都静默继承它。
                    print(f"    [kimi] 输出在 max_tokens={payload['max_tokens']} 处仍被截断，"
                          f"本次调用作失败处理")
                    raw = None
                    break
                u = d.get("usage", {})
                self.usage.calls += 1
                self.usage.prompt_tokens += u.get("prompt_tokens", 0)
                self.usage.completion_tokens += u.get("completion_tokens", 0)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                # 429 / 5xx 值得重试，4xx 参数错重试没意义
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    # 限流退避更长（5/10/20/40 s）；5xx 是瞬时故障，短退避就够
                    time.sleep(RATE_LIMIT_BACKOFF_S * 2 ** attempt if e.code == 429
                               else 2 ** attempt * 3)
                    continue
                print(f"    [kimi] HTTP {e.code}: {body}")
                self.usage.failed += 1
                return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 2)
                    continue
                print(f"    [kimi] {type(e).__name__}: {e}")
                self.usage.failed += 1
                return None

        if raw is None:
            self.usage.failed += 1
            return None
        if cfile:
            cfile.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8")
        return _extract_json(raw) if want_json else raw


def _to_data_uri(img: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    """BGR 数组 -> data URI。缩到 max_side 以内：视觉判断用不着原分辨率，
    而图像 token 是按面积算的，不缩会白烧钱。"""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _extract_json(text: str):
    """模型经常把 JSON 包在 ```json 里，或者前面加一段解释。都要能吃下。"""
    if text is None:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # 退而求其次：抓第一个平衡的 { } 或 [ ]
    for op, cl in (("{", "}"), ("[", "]")):
        i = t.find(op)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(t)):
            if t[j] == op:
                depth += 1
            elif t[j] == cl:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:j + 1])
                    except Exception:
                        break
    return {"_unparsed": text}


# ------------------------------------------------------------------ 角色
@dataclass
class Role:
    """一个智能体角色 = 系统提示 + 输出契约 + 一个把输入拼成请求的函数。"""
    name: str
    system: str
    schema_hint: str
    max_tokens: int = 2048


ROLE_TILE_INSPECTOR = Role(
    name="tile_inspector",
    system=(
        "你是二维材料 CVD 生长的显微图像质检员。你在看一张光学显微镜拍的瓦片，"
        "它属于一张大面积拼接图。衬底是 SiO2/Si 或蓝宝石，上面长着 MoS2 晶体"
        "（三角形或六边形的片状物）。\n\n"
        "关键区分（这是你存在的唯一理由，统计指标做不到）：\n"
        "  - empty_substrate：对焦正常、只是这块区域没长晶体。衬底纹理、划痕、"
        "灰尘边缘都是清晰的。这种**必须保留**，它是真实数据，代表该处覆盖率为零。\n"
        "  - defocus：整幅模糊，边缘发虚，没有任何清晰的细节。丢弃。\n"
        "  - blank_frame：几乎纯色，连衬底纹理都没有，像是快门空拍或光路被挡。丢弃。\n"
        "  - sample_edge：画面很大一部分是死黑，但**边上有一条对焦清晰的窄带**，"
        "带里能看到正常的衬底和晶体。这是扫描走出了衬底边缘，黑的那部分是芯片外面。"
        "这种**必须保留**——拼接图的边界就是由它们定义的，丢掉会在边缘啃出缺口。"
        "判它的标志是：黑区和亮区之间有一条笔直、锐利的分界线。\n"
        "  - overexposed / underexposed：整幅死白或死黑，**没有**任何一条清晰的窄带，"
        "细节不可恢复。丢弃。注意先排除 sample_edge 再判这一类。\n"
        "  - artifact：有大块污染、气泡、纤维、油渍盖住视野。视遮挡比例决定。\n"
        "  - normal：有晶体且对焦正常。保留。\n\n"
        "两条容易搞错的判据：\n"
        "  1. 判 defocus 之前先问：画面里有没有任何一处是锐利的？只要衬底的划痕"
        "或颗粒是锐利的，那就不是虚焦，是空衬底。\n"
        "  2. keep 的标准是**这张图能不能为拼接图贡献有效像素**，"
        "不是画面里有意思的内容占多大比例。一张七成是芯片外黑区、"
        "三成是清晰衬底的图，仍然要保留。"
    ),
    schema_hint=('{"category": "normal|empty_substrate|sample_edge|defocus|blank_frame|'
                 'overexposed|underexposed|artifact", "keep": true/false, '
                 '"confidence": 0.0-1.0, "reason": "一句话，中文"}'),
)

ROLE_CONFLICT_ADJUDICATOR = Role(
    name="conflict_adjudicator",
    system=(
        "同一个网格位置上出现了多张候选照片（通常是操作者当场补拍过）。"
        "只能留一张。你会看到：候选图，以及它们在网格里上下相邻位置的照片。\n\n"
        "判据按优先级：\n"
        "  1. 与上下邻居的视野连续性——留下的那张，其上下边缘的特征必须能和"
        "邻居对上。这是最硬的判据。\n"
        "  2. 对焦质量。\n"
        "  3. 曝光与邻居是否一致（补拍如果隔了几分钟，光源可能已经漂移，"
        "亮度和邻居差很多的那张更可疑）。\n"
        "只输出 JSON。"
    ),
    schema_hint=('{"winner_index": 0起的整数, "confidence": 0.0-1.0, '
                 '"reason": "一句话，中文", "runner_up_usable": true/false}'),
)

ROLE_SEAM_INSPECTOR = Role(
    name="seam_inspector",
    system=(
        "你在检查一张拼接结果的局部裁剪。要找的是拼接缺陷，不是样品本身的特征。\n"
        "  - seam_line：明显的直线亮度台阶，横平竖直，横跨整个裁剪区。\n"
        "  - misregistration：特征被错位、断开或出现重影（同一个三角形晶体"
        "在接缝两侧对不上）。\n"
        "  - vignette_grid：规则的棋盘状明暗，说明平场校正没做好。\n"
        "  - ghosting：半透明的重影，说明该处两张瓦片位置解错了。\n"
        "  - clean：没有可见缺陷。\n"
        "注意：MoS2 晶体本身是三角形、边界锐利，那是样品，不是缺陷。"
        "衬底上的划痕也是样品。只有沿着规则网格出现的东西才是拼接缺陷。\n"
        "只输出 JSON。"
    ),
    schema_hint=('{"defects": ["clean|seam_line|misregistration|vignette_grid|ghosting"], '
                 '"severity": "none|minor|major", "confidence": 0.0-1.0, '
                 '"where": "缺陷在裁剪图中的大致位置，中文", '
                 '"suggested_fix": "increase_overlap_search|redo_flatfield|'
                 'lower_response_threshold|none"}'),
)

ROLE_GRID_PLANNER = Role(
    name="grid_planner",
    system=(
        "你要从一份文件清单里反推出显微拼接数据集的网格布局，好让拼接程序"
        "不用改代码就能处理一个新数据集。\n\n"
        "常见的组织方式：\n"
        "  - 按列（或行）分文件夹/压缩包，文件夹名里带序号和扫描方向；\n"
        "  - 蛇形扫描：相邻列方向相反，所以偶数列的最后一张和奇数列的第一张"
        "在物理上相邻；\n"
        "  - 文件名里的序号是**采集顺序**，不一定等于物理行号。\n\n"
        "要特别当心：序号可能不连续、可能出现小数（补拍）、"
        "各列张数可能不等、压缩包内层文件夹名可能和压缩包名不一致。"
        "你要在输出里明确指出这些异常，不要假装数据是规整的。\n"
        "只输出 JSON。"
    ),
    schema_hint=(
        '{"group_regex": "匹配列/行容器名的 python 正则，用命名组 index 和 direction",'
        ' "tile_regex": "匹配瓦片文件名的正则，用命名组 idx",'
        ' "serpentine": true/false,'
        ' "major_axis": "column|row",'
        ' "reverse_direction_token": "表示反向的字符串，如 up",'
        ' "anomalies": ["中文描述，每条一个异常"],'
        ' "confidence": 0.0-1.0,'
        ' "notes": "中文，给使用者的提醒"}'),
    max_tokens=8192,
)


# ------------------------------------------------------------------ 智能体池
ADAPTIVE_GATE = 0.8        # 单票置信度低于它就再补两票（adaptive_votes 实验，23 区域上调出）
ADAPTIVE_MAX_VOTES = 5     # 自适应投票的上限：1 -> 3 -> 5
DEFAULT_VOTES = 3


def _as_bool(x) -> bool:
    """模型偶尔把布尔写成字符串（"false"）。bool("false") 是 True，这里不能靠它。"""
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("false", "0", "no", "n", "否", "none", ""):
            return False
        if t in ("true", "1", "yes", "y", "是"):
            return True
    return bool(x)


def _conf(a: dict) -> float:
    try:
        return float(a.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _valid(a, key: str, decision_key: str | None = None) -> bool:
    """一份能计票的回答：是 dict、带 key，且（若要按去留计票）带 decision_key。"""
    if not (isinstance(a, dict) and key in a):
        return False
    if decision_key is None:
        return True
    v = a.get(decision_key)
    # a null / unrecognised decision value must not be counted as a False vote
    return isinstance(v, bool) or (isinstance(v, str) and v.strip().lower() in
                                   ("true", "1", "yes", "y", "是", "false", "0", "no", "n", "否"))


def tally_answers(answers: list, key: str = "category", decision_key: str | None = None):
    """计票。返回 (winner, tally, decision_tally)。

    tally 永远是 key 在全部回答上的计数（调用方一直存的那个形状）。

    decision_key=None：winner 是票数最多的 key 取值；**最高票并列**（1-1、1-1-1 ……，
    典型情形是一票失败之后）没有多数，返回 None——三方平票不能由谁先插入 dict 来决定
    （adaptive_votes 实验：S05mg_r03 在 multilayer/damage/contamination 三方平票时被首答
    保留，尽管 2/3 票说剔除）。

    decision_key 给定（一个布尔字段，如 "exclude" / "keep"）：多数在这个字段上取——二元，
    没有三方平票；恰好平票（1-1、2-2）返回 None。winner 是**胜出一侧**回答里最常见的 key
    取值；同频时取置信度更高的那个，再同就按名字，绝不按到达顺序。decision_tally 是
    {"true": n, "false": m}。这样熔池/连续膜/污染之间的标签摆动（全在剔除侧）不再算分歧。
    """
    tally: dict = {}
    for a in answers:
        tally[str(a[key])] = tally.get(str(a[key]), 0) + 1
    if not tally:
        return None, {}, {}
    if decision_key is None:
        top = max(tally.values())
        leaders = [k for k, n in tally.items() if n == top]
        return (leaders[0] if len(leaders) == 1 else None), tally, {}
    dtally = {"true": 0, "false": 0}
    for a in answers:
        dtally["true" if _as_bool(a[decision_key]) else "false"] += 1
    if dtally["true"] == dtally["false"]:
        return None, tally, dtally
    side = dtally["true"] > dtally["false"]
    on_side = [a for a in answers if _as_bool(a[decision_key]) == side]
    sub: dict = {}
    for a in on_side:
        sub[str(a[key])] = sub.get(str(a[key]), 0) + 1
    top = max(sub.values())
    leaders = [k for k, n in sub.items() if n == top]
    if len(leaders) > 1:
        best = {k: max(_conf(a) for a in on_side if str(a[key]) == k) for k in leaders}
        leaders.sort(key=lambda k: (-best[k], k))
    return leaders[0], tally, dtally


def is_split(answers: list, key: str = "category", decision_key: str | None = None) -> bool:
    """没有多数、去留不一致（有 decision_key 时）、或标签不一致（没有时）。"""
    w, tally, dtally = tally_answers(answers, key, decision_key)
    if w is None:
        return True
    if decision_key is not None:
        return min(dtally.values()) > 0
    return tally[w] < len(answers)


class AgentPool:
    """并行跑多个 Kimi 角色调用，并支持对同一个问题做 N 路独立投票。

    投票是这里的核心：单次 LLM 判断在边界样本上不稳定，N 路多数票能把
    方差压下来。因为 kimi-k3 的 temperature 锁死为 1、无法调低，
    多样性只能靠给每个投票者不同的身份提示来制造。

    votes 可以是整数（固定 N 路，默认 3），也可以是 "adaptive"（vote_adaptive：
    1 票，置信度低于 ADAPTIVE_GATE 或回答无效再补 2 票，去留分歧再补 2 票，上限 5）。
    "adaptive" 只影响 map() 的缺省路径；vote() 始终是固定票数（此时取 3）。
    """

    def __init__(self, client: KimiClient, workers: int = 6, votes: int | str = DEFAULT_VOTES):
        self.client = client
        self.workers = workers
        self.adaptive = isinstance(votes, str) and votes.strip().lower() == "adaptive"
        self.votes = DEFAULT_VOTES if self.adaptive else max(1, int(votes))

    # -------- 单次
    def ask(self, role: Role, prompt: str, images=None, voter: int | None = None):
        seed = ""
        if voter is not None:
            seed = (f"\n\n[你是第 {voter + 1} 位独立评审，不知道其他评审的结论。"
                    f"请只依据眼前的图像和数据独立判断。]")
        full = (f"{prompt}\n\n只输出 JSON，不要任何解释文字、不要代码块标记。"
                f"格式：\n{role.schema_hint}{seed}")
        return self.client.chat(full, images=images, system=role.system,
                                max_tokens=role.max_tokens)

    def _ask_many(self, role: Role, prompt: str, images, voters) -> list:
        """并行问一组投票者，按投票者顺序返回原始回答（失败为 None）。"""
        voters = list(voters)
        if not voters:
            return []
        with cf.ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(voters)))) as ex:
            futs = [ex.submit(self.ask, role, prompt, images, v) for v in voters]
            return [f.result() for f in futs]

    # -------- N 路投票
    def vote(self, role: Role, prompt: str, images=None, key: str = "category",
             decision_key: str | None = None):
        """独立问 votes 次，按 key 取多数。返回 (共识值, 票数明细, 全部有效回答)。

        最高票并列返回共识值 None（不再由插入顺序决定）。给了 decision_key 时多数按
        该布尔字段取、共识值是胜出侧最常见的 key 取值，见 tally_answers。
        """
        if not self.client.available:
            return None, {}, []
        answers = self._ask_many(role, prompt, images, range(self.votes))
        answers = [a for a in answers if _valid(a, key, decision_key)]
        if not answers:
            return None, {}, []
        winner, tally, _ = tally_answers(answers, key, decision_key)
        return winner, tally, answers

    # -------- 自适应投票
    def _vote_adaptive(self, role: Role, prompt: str, images, key: str,
                       decision_key: str | None, gate: float, max_votes: int):
        """返回 (winner, tally, answers, n_asked)。投票者编号从 0 起顺序使用，
        所以在固定三票（0,1,2）跑过的样品上重跑，前三票全部命中缓存。"""
        if not self.client.available:
            return None, {}, [], 0
        max_votes = max(1, int(max_votes))
        first = self.ask(role, prompt, images, voter=0)
        asked = 1
        answers = [first] if _valid(first, key, decision_key) else []
        if answers and _conf(first) >= float(gate):
            w, tally, _ = tally_answers(answers, key, decision_key)
            return w, tally, answers, asked
        # 置信度不够或回答无效：补到 3 票；之后每次去留（或标签）仍分歧就再补 2 票，到上限为止
        while asked < max_votes:
            n_more = min(2, max_votes - asked)
            more = self._ask_many(role, prompt, images, range(asked, asked + n_more))
            asked += n_more
            answers += [a for a in more if _valid(a, key, decision_key)]
            if answers and not is_split(answers, key, decision_key):
                break
        w, tally, _ = tally_answers(answers, key, decision_key) if answers else (None, {}, {})
        return w, tally, answers, asked

    def vote_adaptive(self, role: Role, prompt: str, images=None, key: str = "category",
                      decision_key: str | None = None, gate: float = ADAPTIVE_GATE,
                      max_votes: int = ADAPTIVE_MAX_VOTES):
        """1 票；置信度 < gate 或回答无效则再 2 票取多数；3 票在 decision_key 上分歧
        （没有 decision_key 时：标签不一致）再 2 票，上限 max_votes。返回形状同 vote()。
        adaptive_votes 实验：23 个区域、720 种投票顺序上去留与固定三票 100 % 一致，
        1.73 次调用/区域。"""
        w, tally, answers, _ = self._vote_adaptive(role, prompt, images, key,
                                                   decision_key, gate, max_votes)
        return w, tally, answers

    # -------- 代表性回答
    @staticmethod
    def _decorate(winner: str, tally: dict, answers: list, asked: int,
                  key: str, decision_key: str | None) -> dict:
        """取胜出类别（胜出侧）里置信度最高的那份回答作为代表，附上票数。"""
        _, _, dtally = tally_answers(answers, key, decision_key)
        cands = [a for a in answers if str(a.get(key)) == winner]
        if decision_key is not None:
            side = dtally["true"] > dtally["false"]
            cands = [a for a in cands if _as_bool(a[decision_key]) == side]
        best = max(cands, key=_conf)
        out = {**best, "_tally": tally, "_n_votes": len(answers), "_n_asked": int(asked)}
        if decision_key is not None:
            out[decision_key] = _as_bool(best[decision_key])   # 记录计过票的那个值
            out["_decision_tally"] = dtally
            out["_split"] = min(dtally.values()) > 0
        return out

    # -------- 批量并行
    def map(self, role: Role, items: list, build, key: str = "category",
            use_votes: bool = True, verbose: bool = True,
            decision_key: str | None = None, adaptive: bool | None = None,
            gate: float = ADAPTIVE_GATE, max_votes: int = ADAPTIVE_MAX_VOTES) -> list:
        """对一批条目并行跑同一个角色。build(item) -> (prompt, images)。
        返回和 items 等长的结果列表；失败或**平票**的位置为 None。

        每个非 None 结果带 _tally（key 的计数）、_n_votes（收到的有效回答数）、
        _n_asked（实际调用次数）；给了 decision_key 还带 _decision_tally
        （{"true": n, "false": m}）和 _split（去留多数不是全票时为 True）。
        adaptive=None 时沿用池的设置（votes="adaptive"）。
        """
        if not self.client.available:
            return [None] * len(items)
        if adaptive is None:
            adaptive = self.adaptive
        results: list = [None] * len(items)
        done = 0

        def one(idx_item):
            i, item = idx_item
            prompt, images = build(item)
            if use_votes and adaptive:
                w, tally, answers, asked = self._vote_adaptive(
                    role, prompt, images, key, decision_key, gate, max_votes)
            elif use_votes and self.votes > 1:
                w, tally, answers = self.vote(role, prompt, images, key=key,
                                              decision_key=decision_key)
                asked = self.votes
            else:
                a = self.ask(role, prompt, images)
                answers = [a] if _valid(a, key, decision_key) else []
                w, tally, _ = tally_answers(answers, key, decision_key)
                asked = 1
            if w is None:
                return i, None
            return i, self._decorate(w, tally, answers, asked, key, decision_key)

        # 池宽度按 workers 算；每个条目内部还会再开 votes 路，所以这里收着点
        inner = 2 if (use_votes and adaptive) else (self.votes if use_votes else 1)
        outer = max(1, self.workers // max(1, inner))
        with cf.ThreadPoolExecutor(max_workers=outer) as ex:
            for i, res in ex.map(one, list(enumerate(items))):
                results[i] = res
                done += 1
                if verbose and (done % 5 == 0 or done == len(items)):
                    print(f"    [{role.name}] {done}/{len(items)}", flush=True)
        return results
