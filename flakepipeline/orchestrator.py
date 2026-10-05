"""Deterministic orchestration for stitching, layer segmentation and area statistics.

For each configured sample, the pipeline plans work, dispatches tools, checks
results and records input/output hashes. Named failures may trigger a bounded
parameter repair; unresolved cases enter the human review queue. Segmentation
uses supplied pretrained weights and does not train a new model.

Layer-map auditing and region eligibility review are distinct from segmentation.
Saved human verdicts and model decisions can be replayed with their provenance.
Area statistics retain the valid support and exclusion mask. Running this software
does not establish that a region exclusion is physically correct or that model
supervision improves measurement reliability; those require separate evidence."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import stages
import review as RV
from agents import REGISTRY, Result   # the roster and the shared contract, actually used

# Stage name -> agent name in the roster. The stitch stage chains several agents
# internally (planner / tile_inspector / conflict_adjudicator / registrar / compositor /
# seam_inspector), so what is recorded here is its exit agent.
# stages whose Result carries no product later stages key on (escalate-only)
NO_PRODUCT = {"audit"}

STAGE_AGENT = {"stitch": "compositor", "segment": "segmenter",
               "audit": "segmentation_auditor",
               "regions": "region_adjudicator", "stats": "statistician"}


def _hash_file(p, chunk=1 << 20, *, full=False):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    return digest if full else digest[:16]


def _hash_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()[:16]


# ============================================================ tools
# A tool is a bounded reaction to a *named* self-check failure: it looks at the
# stage's Result and the context it ran with, and either returns the parameter
# overrides for one retry plus a one-line justification, or None when it does not
# apply. Tools never call a model and never touch data; they change parameters.
def tool_segment_retry_filler(ctx, res):
    if "填充色可能判错" in res.escalate_reason:
        new = "black" if ctx.get("filler", "white") == "white" else "white"
        return {"filler": new}, f"非扫描区几乎为零，换填充色 {ctx.get('filler','white')} -> {new} 重跑"
    return None


def tool_segment_more_overlap(ctx, res):
    if "切块边界伪影偏大" in res.escalate_reason:
        ov = int(ctx.get("overlap", 64) or 0)
        new = max(128, ov * 2)
        return {"overlap": new}, f"切块边界伪影偏大，重叠 {ov} -> {new} px 重跑"
    return None


def tool_regions_revote(ctx, res):
    """+2 votes when a fixed-vote run split on keep/exclude ("去留分歧", the text
    stage_regions emits since 2026-09-02; "票数分歧" kept for old manifests). In
    adaptive mode the pool already re-votes on a split, so there is nothing to repair."""
    if not any(t in res.escalate_reason for t in ("去留分歧", "票数分歧")):
        return None
    if ctx.get("no_ai") or ctx.get("replay"):
        return None
    adaptive, v = stages.votes_setting(ctx)
    if adaptive:
        return None
    return {"votes": v + 2}, f"去留分歧，投票 {v} -> {v + 2} 路重判"


TOOLS = {
    "segment": [("retry_filler", tool_segment_retry_filler),
                ("more_overlap", tool_segment_more_overlap)],
    "regions": [("revote", tool_regions_revote)],
}


@dataclass
class Attempt:
    params: dict
    ok: bool
    confidence: float
    escalate: bool
    reason: str
    evidence: str = ""
    tool: str = ""                  # the tool whose overrides produced this attempt


@dataclass
class Decision:
    action: str                     # accept | repair | escalate
    reason: str = ""
    tool: str = ""
    overrides: dict = field(default_factory=dict)


# ============================================================ orchestrator
class Orchestrator:
    """Runs the five stages per sample. One configuration applies to every sample in
    the batch."""

    STAGES = ["stitch", "segment", "audit", "regions", "stats"]

    def __init__(self, config: dict):
        self.cfg = dict(config)
        self.work_root = Path(self.cfg["work_root"])
        self.work_root.mkdir(parents=True, exist_ok=True)
        # config_hash covers what changes the science; run-control switches do not
        self.cfg_hash = _hash_obj({k: v for k, v in self.cfg.items()
                                   if k not in ("samples", "work_root", "force", "replay",
                                                "human_verdicts", "no_repair", "plan_only")})
        self.policy = {
            "min_confidence": float(self.cfg.get("min_confidence", 0.6)),
            "max_repairs": 0 if self.cfg.get("no_repair") else int(self.cfg.get("max_repairs", 1)),
        }
        hv = self.cfg.get("human_verdicts") or RV.verdicts_path(self.work_root)
        self.human = RV.load_json(Path(hv), []) if Path(hv).exists() else []
        self.plan_doc = None

    # ---------------------------------------------------------- stage table
    def _specs(self):
        """DISPATCH table. key(ctx, st) -> cache key; run(ctx, st) -> Result;
        outputs(ctx, res) -> files the cache must find."""
        def k_stitch(ctx, st):
            return {"raw": str(ctx.get("raw_dir", "")), "mosaic_in": str(ctx.get("mosaic_in", ""))}

        def k_segment(ctx, st):
            return {"mosaic": _hash_file(st["mosaic"]), "weights": _hash_file(ctx["weights"]),
                    "tile": ctx.get("tile"), "overlap": ctx.get("overlap"),
                    "trim": ctx.get("border_trim_px"), "filler": ctx.get("filler")}

        def k_audit(ctx, st):
            # Image texture and scanned support can change without changing the
            # semantic mask. All three files are direct audit inputs.
            P = stages.audit_params(ctx)
            return {"mask": _hash_file(st["mask"]),
                    "mosaic": _hash_file(st["mosaic"]),
                    "valid_mask": _hash_file(st["valid_mask"]), **P,
                    "votes": ctx.get("votes"), "seed": ctx.get("seed", 0),
                    "no_ai": bool(ctx.get("no_ai"))}

        def k_regions(ctx, st):
            return {"mask": _hash_file(st["mask"]), "min_pct": ctx.get("min_region_pct"),
                    "max": ctx.get("max_regions"), "votes": ctx.get("votes"),
                    "no_ai": bool(ctx.get("no_ai")), "replay": bool(ctx.get("replay")),
                    "human": _hash_obj(self.human) if self.human else ""}

        def k_stats(ctx, st):
            return {"mask": _hash_file(st["mask"]),
                    "valid_mask": _hash_file(st["valid_mask"]),
                    "excl": _hash_file(st["exclusion_mask"]) if st.get("exclusion_mask") else ""}

        return [
            dict(name="stitch", key=k_stitch,
                 run=lambda ctx, st: stages.stage_stitch(ctx),
                 outputs=lambda ctx, r: [r["mosaic"]]),
            dict(name="segment", key=k_segment,
                 run=lambda ctx, st: stages.stage_segment(ctx, st["mosaic"]),
                 outputs=lambda ctx, r: [r["mask"], r["valid_mask"]]),
            # escalate-only: its Result carries no product key, so _state_from
            # picks nothing up and the mask reaches regions/stats untouched
            dict(name="audit", key=k_audit,
                 run=lambda ctx, st: stages.stage_audit(ctx, st["mosaic"], st["mask"],
                                                        st["valid_mask"]),
                 outputs=lambda ctx, r: [str(ctx["work"] / "02_segment" /
                                             f"{ctx['sample']}_audit.json")]),
            dict(name="regions", key=k_regions,
                 run=lambda ctx, st: stages.stage_regions(ctx, st["mosaic"], st["mask"],
                                                          st["valid_mask"]),
                 outputs=lambda ctx, r: ([str(ctx["work"] / "03_regions" /
                                              f"{ctx['sample']}_exclusions.json")]
                                         + ([r["exclusion_mask"]] if r.get("exclusion_mask") else []))),
            dict(name="stats", key=k_stats,
                 run=lambda ctx, st: stages.stage_stats(ctx, st["mask"], st["valid_mask"],
                                                        st.get("exclusion_mask")),
                 outputs=lambda ctx, r: [str(ctx["work"] / "04_stats" /
                                             f"{ctx['sample']}_stats.json")]),
        ]

    # ---------------------------------------------------------- context
    def _ctx(self, sample: dict) -> dict:
        ctx = dict(self.cfg)
        ctx.update(sample)
        ctx["sample"] = sample["name"]
        ctx["work"] = self.work_root / sample["name"]
        ctx["human_verdicts"] = [h for h in self.human
                                 if h.get("region_key", {}).get("sample") == sample["name"]]
        return ctx

    @staticmethod
    def _load_manifest(ctx):
        man_path = ctx["work"] / "manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
        man.setdefault("sample", ctx["sample"])
        man.setdefault("stages", {})
        return man_path, man

    def _cached(self, man, stage, key):
        rec = man["stages"].get(stage)
        if not rec or self.cfg.get("force"):
            return None
        if rec.get("inputs_hash") != _hash_obj(key):
            return None
        if rec.get("config_hash") != self.cfg_hash:
            return None
        outputs, hashes = rec.get("outputs"), rec.get("output_sha256")
        # Legacy manifests without output hashes are deliberately a cache miss.
        # File existence alone cannot establish that the recorded result still
        # corresponds to its products (e.g. after an interrupted external edit).
        if (not isinstance(outputs, list) or not outputs
                or any(not isinstance(p, str) for p in outputs)
                or not isinstance(hashes, dict) or set(hashes) != set(outputs)):
            return None
        for p in outputs:
            if not isinstance(hashes[p], str) or not Path(p).is_file():
                return None
            try:
                if _hash_file(p, full=True) != hashes[p]:
                    return None
            except OSError:
                return None
        return rec

    # ---------------------------------------------------------- PLAN
    def plan(self, samples: list) -> dict:
        """Say what would run, before running it. For every stage: cache hit, run, or
        'after <stage>' when its cache key depends on an output that does not exist
        yet. Written to work_root/plan.json."""
        doc = {"config_hash": self.cfg_hash, "policy": self.policy,
               "human_verdicts": len(self.human), "samples": []}
        for s in samples:
            ctx = self._ctx(s)
            _, man = self._load_manifest(ctx)
            st = {}
            entry = {"sample": s["name"], "stages": [], "human_verdicts": len(ctx["human_verdicts"])}
            blocked = None
            for spec in self._specs():
                name = spec["name"]
                line = {"stage": name, "agent": STAGE_AGENT[name],
                        "tools": [t for t, _ in TOOLS.get(name, [])]}
                if name == "stitch":
                    src = ctx.get("mosaic_in") or ctx.get("raw_dir")
                    if not src or not Path(src).exists():
                        line.update(action="input missing", detail=f"找不到 {src or 'mosaic_in/raw_dir'}")
                        blocked = name
                        entry["stages"].append(line)
                        continue
                if blocked:
                    line.update(action=f"after {blocked}", detail="上游还没有产物")
                    entry["stages"].append(line)
                    continue
                try:
                    key = spec["key"](ctx, st)
                except (KeyError, FileNotFoundError, TypeError):
                    line.update(action=f"after {entry['stages'][-1]['stage']}",
                                detail="缓存键依赖上游产物")
                    if name not in NO_PRODUCT:
                        blocked = name
                    entry["stages"].append(line)
                    continue
                rec = self._cached(man, name, key)
                if rec:
                    line.update(action="cache", detail=f"命中 {rec.get('at', '')}")
                    st.update(self._state_from(rec["result"]))
                else:
                    why = "force" if self.cfg.get("force") else (
                        "无记录" if name not in man["stages"] else "输入或配置变了")
                    line.update(action="run", detail=why)
                    if name not in NO_PRODUCT:   # downstream keys need this stage's products
                        blocked = name
                entry["stages"].append(line)
            doc["samples"].append(entry)
        (self.work_root / "plan.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        self.plan_doc = doc
        return doc

    @staticmethod
    def _state_from(r: dict) -> dict:
        """The products later stages key on, pulled out of a stage's result dict."""
        st = {}
        for k in ("mosaic", "mask", "valid_mask", "exclusion_mask"):
            if r.get(k):
                st[k] = r[k]
        return st

    @staticmethod
    def print_plan(doc: dict):
        print(f"plan  config {doc['config_hash']}  策略 {doc['policy']}  "
              f"人工判决 {doc['human_verdicts']} 条")
        for s in doc["samples"]:
            print(f"  {s['sample']}")
            for l in s["stages"]:
                tools = f"  工具 {','.join(l['tools'])}" if l["tools"] else ""
                print(f"    {l['stage']:<8} {l['action']:<16} {l.get('detail','')}{tools}")

    # ---------------------------------------------------------- VERIFY
    def _verify(self, stage: str, ctx: dict, res: Result, attempts: list) -> Decision:
        if stage == "stitch":
            self._attach_stitch_findings(ctx, res)
        conf_ok = float(res.confidence) >= self.policy["min_confidence"]
        if res.ok and not res.escalate and conf_ok:
            return Decision("accept")
        reason = res.escalate_reason or (
            f"置信度 {res.confidence:.2f} 低于阈值 {self.policy['min_confidence']}" if not conf_ok
            else "自检未通过")
        if len(attempts) < self.policy["max_repairs"]:
            used = {a.tool for a in attempts}
            for tname, tool in TOOLS.get(stage, []):
                if tname in used:
                    continue
                hit = tool(ctx, res)
                if hit:
                    overrides, why = hit
                    return Decision("repair", reason=why, tool=tname, overrides=overrides)
        return Decision("escalate", reason=reason)

    @staticmethod
    def _attach_stitch_findings(ctx, res: Result):
        """The seam inspector inside run_stitch.py records its findings in the stitch
        work state; read them back so a major seam defect is a verified escalation of
        this stage rather than a line buried in stitch.log."""
        sp = ctx["work"] / "01_mosaic" / "_cache" / "state.json"
        if not sp.exists():
            return
        try:
            findings = json.loads(sp.read_text(encoding="utf-8")).get("inspect") or []
        except Exception:
            return
        major = [f for f in findings if f.get("severity") == "major"]
        res.data["seam_findings"] = findings
        if findings:
            res.evidence = (res.evidence + "; " if res.evidence else "") + \
                f"接缝抽查 {len(findings)} 处可疑（major {len(major)}）"
        if major:
            res.escalate = True
            res.ok = False
            res.confidence = min(float(res.confidence), 0.5)
            res.escalate_reason = (res.escalate_reason + "; " if res.escalate_reason else "") + \
                "接缝抽查发现 major 缺陷: " + "; ".join(
                    f"{f.get('xy')} {f.get('defects')}" for f in major[:3])

    # ---------------------------------------------------------- one sample
    def run_sample(self, sample: dict) -> dict:
        name = sample["name"]
        ctx = self._ctx(sample)
        ctx["work"].mkdir(parents=True, exist_ok=True)
        man_path, man = self._load_manifest(ctx)
        man["config_hash"] = self.cfg_hash
        escalations, actions = [], {}
        st = {}                              # products, keyed as later stages need them
        results = {}

        for spec in self._specs():
            stage = spec["name"]
            key = spec["key"](ctx, st)
            rec = self._cached(man, stage, key)
            if rec:
                print(f"  [{name}] {stage:<8} : 命中缓存")
                r = rec["result"]
                actions[stage] = "cache"
                results[stage] = r
                st.update(self._state_from(r))
                self._print_stage(stage, r)
                continue

            print(f"  [{name}] {stage:<8} ...")
            attempts, overrides, tool_used = [], {}, ""
            while True:
                ctx_a = {**ctx, **overrides}
                res = spec["run"](ctx_a, st)
                dec = self._verify(stage, ctx_a, res, attempts)
                attempts.append(Attempt(
                    params={k: ctx_a.get(k) for k in ("filler", "overlap", "tile", "votes",
                                                      "border_trim_px", "audit_var_thr",
                                                      "audit_min_px", "audit_vlm_windows")
                            if k in ctx_a},
                    ok=bool(res.ok), confidence=round(float(res.confidence), 3),
                    escalate=bool(res.escalate), reason=res.escalate_reason,
                    evidence=res.evidence, tool=tool_used))
                if dec.action == "repair":
                    print(f"      [TOOL {dec.tool}] {dec.reason}")
                    overrides.update(dec.overrides)
                    tool_used = dec.tool
                    continue
                break

            key = spec["key"](ctx_a, st)      # the key of the attempt that was kept
            r = res.data
            self._record(man, man_path, stage, key, res, spec["outputs"](ctx_a, r),
                         attempts, dec)
            actions[stage] = "repair" if len(attempts) > 1 else "run"
            results[stage] = r
            st.update(self._state_from(r))
            if dec.action != "escalate":
                RV.clear_queue(self.work_root, name, stage)   # a passed re-run closes its old line
            if dec.action == "escalate":
                e = {"stage": stage, "sample": name, "reason": dec.reason,
                     "confidence": round(float(res.confidence), 3)}
                if stage == "regions":
                    e["regions"] = [
                        {"rank": v.get("rank"), "category": v.get("category"),
                         "_tally": v.get("_tally"), "region_key": RV.region_key(name, v)}
                        for v in r.get("verdicts", [])
                        if v.get("category") == "uncertain"
                        or v.get("needs_human") or v.get("_split")
                        or (v.get("_tally") and max(v["_tally"].values()) < stages.votes_setting(ctx)[1])
                        or not v.get("_ai")]
                elif stage == "audit":
                    e["alarms"] = [{k: a.get(k) for k in ("rank", "bbox", "area_px",
                                                          "dominant_class", "pct_of_class")}
                                   for a in r.get("alarms", [])]
                escalations.append(e)
                RV.push_queue(self.work_root, e)
                print(f"      [ESCALATE] {stage}: {dec.reason}")
            elif res.evidence:
                print(f"      自检: {res.evidence}")
            self._print_stage(stage, r)

        r3, r4 = results["regions"], results["stats"]
        return {"sample": name, "manifest": str(man_path), **r4,
                "regions": r3.get("verdicts", []),
                "human_overrides": r3.get("human_overrides", []),
                "actions": actions, "escalations": escalations}

    def _record(self, man, man_path, stage, key, res, outputs, attempts, dec):
        """The contract fields are logged individually; data goes into result to stay
        backward compatible. attempts/decision are the new provenance."""
        outputs = [str(o) for o in outputs]
        hashes = {}
        for p in outputs:
            try:
                hashes[p] = _hash_file(p, full=True) if Path(p).is_file() else None
            except OSError:
                # Preserve the failed stage's record, but never reuse an output
                # whose bytes could not be recorded.
                hashes[p] = None
        man["stages"][stage] = {
            "inputs_hash": _hash_obj(key), "config_hash": self.cfg_hash,
            "outputs": outputs, "output_sha256": hashes,
            "ok": bool(res.ok), "confidence": round(float(res.confidence), 3),
            "evidence": res.evidence,
            "escalate": bool(res.escalate), "escalate_reason": res.escalate_reason,
            "agent": STAGE_AGENT.get(stage, stage),
            "agent_kind": (REGISTRY[STAGE_AGENT[stage]].kind
                           if stage in STAGE_AGENT else "unknown"),
            "agent_judgment": (REGISTRY[STAGE_AGENT[stage]].judgment
                               if stage in STAGE_AGENT else ""),
            "decision": dec.action, "decision_reason": dec.reason,
            "attempts": [asdict(a) for a in attempts],
            "result": res.data,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _print_stage(stage, r):
        if stage == "segment":
            print(f"      非扫描区域排除 {r.get('excluded_non_scan_pct')}%，"
                  f"切块 {r.get('tiles')} 块（{r.get('tile')} px，重叠 {r.get('overlap')}）")
        elif stage == "audit":
            p = r.get("pilot") or {}
            pilot = (f"VLM 试点 {p.get('flagged', 0)}/{p.get('answered', 0)} 窗口有误，"
                     f"分歧 {p.get('split', 0)}" if p.get("ran") else "VLM 试点未运行")
            print(f"      分割审计：确定性报警 {r.get('n_alarms', 0)} 处，"
                  f"无纹理晶体像素 {float(r.get('low_var_crystal_frac', 0) or 0) * 100:.2f}%；{pilot}")
            for a in r.get("alarms", [])[:5]:
                print(f"        报警 #{a['rank']} bbox {a['bbox']}  {a['area_px']:,} px  "
                      f"{a['dominant_class']} 占该类 {a['pct_of_class']}%  灰度 {a['mean_gray']}")
        elif stage == "regions":
            for v in r.get("verdicts", []):
                flag = "剔除" if v.get("exclude") else "保留"
                who = f"  [人工 {v['decided_by']}]" if v.get("decided_by") else ""
                print(f"      #{v['rank']} 占厚层 {v.get('pct_of_class', 0):>5}%  "
                      f"{v.get('category','?'):<20} {flag}  "
                      f"票 {v.get('_tally',{})}  {str(v.get('reason',''))[:40]}{who}")
        elif stage == "stats":
            rp = r.get("ratios_pct", {})
            print(f"      => 1L {rp.get('1L')}%  2L {rp.get('2L')}%  TL {rp.get('TL')}%")

    # ---------------------------------------------------------- batch
    def run(self, samples: list) -> dict:
        t0 = time.time()
        plan = self.plan(samples)
        self.print_plan(plan)
        results = []
        for s in samples:
            print(f"--- {s['name']} ---")
            results.append(self.run_sample(s))
        report = {"config_hash": self.cfg_hash, "policy": self.policy,
                  "config": {k: str(v) for k, v in self.cfg.items() if k != "samples"},
                  "plan": plan,
                  "samples": results, "seconds": round(time.time() - t0, 1),
                  "consistency": self._consistency(results, plan)}
        (self.work_root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _consistency(self, results, plan=None):
        """Cross-sample consistency assertions. Reads every sample's manifest.json."""
        checks = []

        # ---- 1. every sample actually used the same configuration ----
        seen, missing = {}, []
        for r in results:
            mp = Path(r.get("manifest", ""))
            if not mp.exists():
                missing.append(r["sample"]); continue
            man = json.loads(mp.read_text(encoding="utf-8"))
            hashes = {st: rec.get("config_hash") for st, rec in man.get("stages", {}).items()}
            seen[r["sample"]] = hashes
        all_h = {h for hs in seen.values() for h in hs.values() if h}
        ok = (len(all_h) == 1) and not missing
        detail = f"从 {len(seen)} 个样品的 manifest 里读出 {len(all_h)} 个不同的 config_hash"
        if all_h:
            detail += "：" + ", ".join(sorted(all_h))
        if missing:
            detail += f"；缺 manifest 的样品: {', '.join(missing)}"
        if not ok and len(all_h) > 1:
            detail += ("。**同一批里出现了不同的配置，这批结果不能互相比较。**"
                       " 逐样品: " + json.dumps(seen, ensure_ascii=False))
        checks.append({"check": "All samples actually used the same configuration (verified against each manifest)",
                       "pass": ok, "detail": detail})

        # ---- 2. exclusion behaviour ----
        cats = {r["sample"]: sorted({v.get("category") for v in r.get("regions", [])
                                     if v.get("exclude")}) for r in results}
        excl_n = {r["sample"]: sum(1 for v in r.get("regions", []) if v.get("exclude"))
                  for r in results}
        uneven = bool([s for s, n in excl_n.items() if n == 0]) and \
                 bool([s for s, n in excl_n.items() if n > 0])
        checks.append({
            "check": "Every sample triggered at least one region exclusion", "pass": not uneven,
            "detail": f"每个样品剔除的区域数 {excl_n}；剔除类别 {cats}。"
                      + ("有的样品剔除了、有的没有。这本身可以是对的（有的样品就是没有熔池），"
                         "但写进论文时必须逐样品说明。" if uneven else "")})

        # ---- 3. is the exclusion rule the same everywhere ----
        rules = {r["sample"]: self.cfg.get("exclusion_rule") for r in results
                 if Path(r.get("manifest", "")).exists()}
        checks.append({"check": "Exclusion rule is identical across samples",
                       "pass": len(set(rules.values())) <= 1,
                       "detail": f"exclusion_rule = {self.cfg.get('exclusion_rule')}"})

        # ---- 4. how far the conclusions depend on the region exclusion ----
        dep = {r["sample"]: r["delta_from_region_exclusion_pp"] for r in results
               if r.get("delta_from_region_exclusion_pp")}
        alarm = float(self.cfg.get("sensitivity_alarm_pct", 2.0))
        hot = {s: {k: v for k, v in d.items() if abs(v) > alarm} for s, d in dep.items()}
        hot = {s: v for s, v in hot.items() if v}
        checks.append({
            "check": f"No sample's proportions hinge on region exclusion (threshold {alarm} pp)",
            "pass": not hot,
            "detail": (f"剔除前后各类占比变化 {dep}。"
                       + (f"**以下超过阈值，写论文时必须交代: {hot}**" if hot else ""))})

        # ---- 5. any escalation still unhandled ----
        esc = [e for r in results for e in r.get("escalations", [])]
        checks.append({"check": "No escalations are awaiting a human", "pass": not esc,
                       "detail": (f"{len(esc)} 条需要人看，已写入 REVIEW/review_queue.json: "
                                  + "; ".join(f"[{e['sample']}/{e['stage']}] {e['reason']}"
                                              for e in esc)) if esc else "无"})

        # ---- 6. did the run follow its own plan ----
        if plan:
            planned = {s["sample"]: {l["stage"]: l["action"] for l in s["stages"]}
                       for s in plan["samples"]}
            dev = []
            for r in results:
                for stg, act in r.get("actions", {}).items():
                    p = planned.get(r["sample"], {}).get(stg, "")
                    if p == "cache" and act != "cache":
                        dev.append(f"{r['sample']}/{stg}: 计划命中缓存，实际 {act}")
                    if p == "run" and act == "cache":
                        dev.append(f"{r['sample']}/{stg}: 计划重跑，实际命中缓存")
            repaired = [f"{r['sample']}/{s}" for r in results
                        for s, a in r.get("actions", {}).items() if a == "repair"]
            checks.append({"check": "Execution matched the plan (cache hits as planned; repairs listed)",
                           "pass": not dev,
                           "detail": ("偏离: " + "; ".join(dev) if dev else "与计划一致")
                                     + (f"；用工具修复过的阶段: {repaired}" if repaired else "")})

        # ---- 7. human overrides, especially of unanimous votes ----
        ov = [(r["sample"], o) for r in results for o in r.get("human_overrides", [])]
        strong = [f"{s} #{o['rank']} {o['agent']['category']}->{o['human']['category']} "
                  f"by {o['human']['decided_by']}: {o['human']['note']}"
                  for s, o in ov if o.get("override_of_unanimous")]
        checks.append({"check": "Human overrides are recorded, and none reverses a unanimous vote silently",
                       "pass": not strong,
                       "detail": (f"人工判决 {len(ov)} 条"
                                  + (f"；**推翻全票一致的判决: {strong}**" if strong else "")
                                  if ov else "无人工判决")})
        return checks
