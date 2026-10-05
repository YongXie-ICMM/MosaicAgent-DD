"""Agent contract and registry.

Design principle: **every stage is an Agent, but not every Agent uses a large model.**

What separates them is who makes the *judgement*, not whether we call it an Agent:
  - registration, blending, U-Net inference and pixel counting must be deterministic;
    handing them to an LLM would only make the results irreproducible;
  - but they must still declare their inputs and outputs, self-check what they
    produced, report a confidence, and escalate to a human when warranted.

One shared contract lets the orchestration layer dispatch, cache and log every stage
the same way, and makes "who decided this step, and on what grounds" answerable at a
glance from the provenance record -- exactly what the 2026-08-29 review found missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Kind = Literal["llm", "deterministic", "hybrid"]


@dataclass
class Result:
    ok: bool
    data: dict = field(default_factory=dict)
    confidence: float = 1.0
    evidence: str = ""                 # grounds for the self-check; goes into the manifest
    escalate: bool = False             # True = a human needs to look at this
    escalate_reason: str = ""


@dataclass
class Agent:
    name: str
    kind: Kind
    consumes: tuple
    produces: tuple
    purpose: str
    judgment: str                      # who makes the call: a named role, or "closed-form"
    run: Callable | None = None
    self_check: str = ""               # how it justifies claiming this step went fine


# ─────────────────────────────────────────────────────────── registry
# Implemented agents (run points at the implementation in stages.py / MosaicAgent)
REGISTRY: dict[str, Agent] = {}


def register(a: Agent) -> Agent:
    REGISTRY[a.name] = a
    return a


register(Agent(
    name="orchestrator", kind="deterministic",
    consumes=("config", "samples", "manifests", "human verdicts"),
    produces=("plan.json", "manifest.json per sample", "report.json", "review queue"),
    purpose="编排层：先出计划（缓存/输入/人工判决），按阶段表派发，"
            "对每个 Result 按策略核验；具名自检失败时用有限次的确定性工具修参重跑"
            "（换填充色、加大重叠、加票复投），仍不过的上报到人工复核队列；"
            "人的判决下次运行按几何身份匹配、覆盖并留档",
    judgment="closed-form policy (min_confidence, max_repairs)；本身不调模型",
    self_check="执行是否与计划一致；是否有推翻全票的人工覆盖；是否还有未处理的上报"))

register(Agent(
    name="planner", kind="llm",
    consumes=("raw file listing",), produces=("grid layout",),
    purpose="从文件清单反推网格布局，换个数据集不用改代码",
    judgment="Kimi ROLE_GRID_PLANNER",
    self_check="反推出的行列数与文件数是否自洽；异常项必须显式列出"))

register(Agent(
    name="tile_inspector", kind="llm",
    consumes=("raw tiles",), produces=("keep/drop per tile",),
    purpose="瓦片质检：空衬底和样品边缘要保留，虚焦和空拍要丢",
    judgment="Kimi ROLE_TILE_INSPECTOR, 3 votes",
    self_check="票数分歧 = 低置信，进 escalate 名单"))

register(Agent(
    name="conflict_adjudicator", kind="llm",
    consumes=("duplicate tiles at one grid slot",), produces=("winner",),
    purpose="同一网格位置多张补拍，选一张",
    judgment="Kimi ROLE_CONFLICT_ADJUDICATOR, 3 votes",
    self_check="与上下邻居的视野连续性"))

register(Agent(
    name="registrar", kind="deterministic",
    consumes=("kept tiles",), produces=("tile positions",),
    purpose="逐对 NCC 配准 + 全局最小二乘",
    judgment="closed-form",
    self_check="全局残差；覆盖率；有没有瓦片被解到画布外"))

register(Agent(
    name="compositor", kind="deterministic",
    consumes=("positions", "tiles"), produces=("stitched map",),
    purpose="平场校正 + 距离加权羽化融合",
    judgment="closed-form",
    self_check="接缝两侧亮度阶跃的统计量"))

register(Agent(
    name="seam_inspector", kind="llm",
    consumes=("stitched map crops",), produces=("defect verdict",),
    purpose="判定接缝缺陷还是样品本身的特征",
    judgment="Kimi ROLE_SEAM_INSPECTOR",
    self_check="确认缺陷则回炉重配准重融合"))

register(Agent(
    name="segmenter", kind="deterministic",
    consumes=("stitched map", "fixed weights"), produces=("layer map", "valid mask"),
    purpose="UNet 重叠切块推理；非扫描填充区用几何显式排除",
    judgment="closed-form (model_0409_all.pth, 不训练)",
    self_check="非扫描区占比、四类像素分布、切块边界带与内部的类别分布差异"))

register(Agent(
    name="segmentation_auditor", kind="hybrid",
    consumes=("layer map", "stitched map", "valid mask"),
    produces=("audit report (alarms + pilot verdicts)",),
    purpose="分割图与原图对照，抓系统性误判——整块无纹理区域被归到某一晶体类、"
            "某一类明显外溢。2026-08-29 那个'纯白非扫描区 99.9% 被判成厚层'的 bug "
            "属于这一类：先由免费的确定性检测器报警（局部光度方差≈0 却被标成晶体），"
            "再按需在少量确定性抽样的窗口上跑视觉模型试点；只上报，不改分割图。",
    judgment="deterministic low-variance detector + Kimi ROLE segmentation_auditor pilot, 3 votes",
    self_check="任一确定性报警、试点标错窗口比例超阈值、或票数分歧 → 上报；"
               "试点没有植入对照，只限定能力，不证明分割正确"))

register(Agent(
    name="region_adjudicator", kind="llm",
    consumes=("layer map", "stitched map"), produces=("exclusion mask", "decision log"),
    purpose="厚层连通域逐块判去留：熔池/连续膜/污染/伪影剔除，真晶体保留",
    judgment="Kimi ROLE_REGION_ADJUDICATOR, 3 votes",
    self_check="票数分歧记录在案；uncertain 一律保留并标人工复核"))

register(Agent(
    name="statistician", kind="deterministic",
    consumes=("layer map", "valid mask", "exclusion mask"), produces=("ratios", "pie"),
    purpose="面积占比，分母 1L+2L+TL（裸衬底不计入）",
    judgment="closed-form",
    self_check="同时给出剔除前后两套数字，差异过大自动 escalate"))

register(Agent(
    name="spectrum_fitter", kind="deterministic",
    consumes=("DR spectrum",), produces=("E_A", "E_B", "amplitudes", "SNR"),
    purpose="AutoSpectra 第一步：1.72–2.14 eV 窗口内双高斯 + 二次背景，"
            "与 Spectra/fit_ab.py 同一模型同一边界，能量就是论文里的能量",
    judgment="closed-form (autospectra.fit_ab)",
    self_check="幅度 ≥ 2.5× 残差噪声；峰位不贴边界；否则判 unfit 上报"))

register(Agent(
    name="spectrum_qc", kind="llm",
    consumes=("spectrum + fit overlay image",), produces=("fit_ok | bad_background | peak_missing | shifted | noisy",),
    purpose="AutoSpectra 第二步：看图判拟合可不可信——数字上收敛不等于抓对了峰",
    judgment="Kimi ROLE spectrum_qc, 3 votes",
    self_check="非 fit_ok 或票数分歧 → 上报复核队列"))

register(Agent(
    name="stacking_assigner", kind="hybrid",
    consumes=("E_A", "E_B", "amplitudes", "substrate", "calibrated reference"),
    produces=("2H | 3R | 1L | uncertain",),
    purpose="AutoSpectra 第三步：闭式规则先判（B−A 劈裂定 2H，A 幅度定 1L/2L），"
            "LLM 按同一参考表交叉判；规则分不开 3R/1L 时用模型的判定",
    judgment="rule (autospectra.rule_assign, --calibrate 定阈值) + Kimi ROLE stacking_assigner, 3 votes",
    self_check="模型与规则不一致、uncertain、票数分歧 → 上报；与标注的一致率写进结果"))

register(Agent(
    name="batch_reviewer", kind="llm",
    consumes=("all samples' stats + decision logs",), produces=("cross-sample review",),
    purpose="【本次新增】跨样品复核：各样品是不是用了同一套规则、"
            "某个样品的结论是不是高度依赖单独一块区域的去留、"
            "趋势在剔除前后是否都成立。b 面板那件事就是它该报警的。",
    judgment="Kimi ROLE_BATCH_REVIEWER  [待实现]",
    self_check="必须同时看剔除前和剔除后两套数字才能下结论"))


register(Agent(
    name="flake_geometer", kind="deterministic",
    consumes=("layer map", "raw full-resolution tiles", "tile positions"),
    produces=("per-flake orientation", "2L-in-1L twist angle", "size distribution"),
    purpose="【拟增】逐片几何：实例分割、边缘方向直方图、最小外接矩形、"
            "双层岛与其宿主单层配对后算相对转角（0 度附近 3R，60 度附近 2H）。"
            "必须在**原始瓦片**上做——拼接图是降采样约 7.5 倍来的，"
            "上面绝大多数双层岛只有几个像素宽，量不出方向。",
    judgment="closed-form  [待实现]",
    self_check="同一片晶体在相邻瓦片重叠区各量一次，两次角度之差就是误差棒"))

register(Agent(
    name="stacking_adjudicator", kind="llm",
    consumes=("flake crops", "geometric measurements"),
    produces=("measurable? 3R / 2H / twisted / unmeasurable",),
    purpose="【拟增】这片能不能量——完整三角，还是被截断、粘连、遮挡；"
            "几何上模棱两可时判堆垛构型。测量交给 flake_geometer，"
            "这里只做把关和判读。",
    judgment="Kimi ROLE_STACKING_ADJUDICATOR, 3 votes  [待实现]",
    self_check="判定结果与 DR 光谱那 90 片的独立标注做交叉验证"))


def summary():
    """Print the agent roster, grouped by kind."""
    for k, label in (("llm", "大模型判断"), ("deterministic", "确定性计算")):
        print(f"\n== {label} ==")
        for a in REGISTRY.values():
            if a.kind != k:
                continue
            todo = "  [待实现]" if "待实现" in a.judgment else ""
            print(f"  {a.name:<24}{a.judgment.replace('  [待实现]','')}{todo}")
            print(f"  {'':<24}自检: {a.self_check}")


if __name__ == "__main__":
    summary()
