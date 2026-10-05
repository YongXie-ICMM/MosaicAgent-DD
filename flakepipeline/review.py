#!/usr/bin/env python3
"""Human-in-the-loop for the region adjudication (ROADMAP 近-7).

Three things live here:

  1. `region_key()` -- a region's **geometric identity**: sample name, full-resolution
     centroid and area. Human verdicts are matched to regions by this key with a
     tolerance, never by rank or by config_hash, so a re-run that renumbers the
     regions or changes a measurement slightly still finds the human's decision.
  2. `apply_overrides()` -- the human's verdict replaces the agent's; the agent's
     original category / vote tally stay in the record next to `decided_by`, and an
     override that reverses a unanimous vote is flagged (`override_of_unanimous`) so
     `report.json` can highlight it instead of silently absorbing it.
  3. `replay_verdicts()` -- rebuild the verdict list from a recorded exclusions.json
     without any model call, so "these files are enough to reproduce" is true.

CLI (writes work_root/REVIEW/human_verdicts.json; `--note` is mandatory):

    python3 review.py --work _work list
    python3 review.py --work _work decide --sample S15mg --rank 1 \
        --category precursor_melt --exclude --by "Yong Xie" --note "液滴状边界，内有未反应团块"
    python3 review.py --work _work decide --sample S05mg --rank 3 --keep --by ... --note ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

QUEUE = "review_queue.json"
VERDICTS = "human_verdicts.json"


# ------------------------------------------------------------------ identity
def region_key(sample: str, v: dict) -> dict:
    x0, x1, y0, y1 = v["bbox"]
    return {"sample": sample, "cx": round((x0 + x1) / 2.0, 1),
            "cy": round((y0 + y1) / 2.0, 1), "area_px": int(v["area_px"])}


def same_region(a: dict, b: dict, pos_frac=0.05, area_frac=0.15, pos_min_px=8) -> bool:
    """Tolerant match: centroid within 5 % of the region's linear size (at least 8 px)
    and area within 15 %. Loose enough to survive a re-run, tight enough that two
    different regions never collide (the candidates are the largest components, so
    they are far apart)."""
    if a["sample"] != b["sample"]:
        return False
    size = math.sqrt(max(a["area_px"], 1))
    tol = max(pos_min_px, pos_frac * size)
    if abs(a["cx"] - b["cx"]) > tol or abs(a["cy"] - b["cy"]) > tol:
        return False
    return abs(a["area_px"] - b["area_px"]) <= area_frac * max(a["area_px"], b["area_px"], 1)


def find_record(sample: str, v: dict, records: list) -> dict | None:
    key = region_key(sample, v)
    for r in records:
        rk = r.get("region_key") or {}
        if rk and same_region(key, rk):
            return r
    return None


# ------------------------------------------------------------------ overrides
def apply_overrides(sample: str, verdicts: list, human: list, votes: int) -> tuple:
    """Let recorded human verdicts overwrite the agent's. Returns (verdicts, log)."""
    log = []
    for v in verdicts:
        h = find_record(sample, v, human)
        if not h:
            continue
        tally = v.get("_tally") or {}
        unanimous = bool(tally) and max(tally.values()) >= votes
        changed = (bool(h["exclude"]) != bool(v.get("exclude"))
                   or (h.get("category") and h["category"] != v.get("category")))
        v["agent_category"] = v.get("category")
        v["agent_exclude"] = bool(v.get("exclude"))
        v["exclude"] = bool(h["exclude"])
        if h.get("category"):
            v["category"] = h["category"]
        v["decided_by"] = h.get("decided_by", "human")
        v["human_note"] = h.get("note", "")
        v["decided_at"] = h.get("at", "")
        v["override_of_unanimous"] = bool(unanimous and changed)
        log.append({"rank": v.get("rank"), "region_key": region_key(sample, v),
                    "agent": {"category": v["agent_category"], "exclude": v["agent_exclude"],
                              "tally": tally},
                    "human": {"category": v["category"], "exclude": v["exclude"],
                              "decided_by": v["decided_by"], "note": v["human_note"]},
                    "changed": changed, "override_of_unanimous": v["override_of_unanimous"]})
    return verdicts, log


# ------------------------------------------------------------------ replay
def replay_verdicts(sample: str, candidates: list, recorded: list) -> list:
    """Rebuild verdicts for freshly measured candidates from a recorded
    exclusions.json (no model call). A candidate with no recorded twin is kept and
    flagged `_replay_unmatched` so it shows up in the self-check."""
    out = []
    for c in candidates:
        cd = c.to_dict() if hasattr(c, "to_dict") else dict(c)
        rec = find_record(sample, cd, [{**r, "region_key": region_key(sample, r)}
                                       for r in recorded if "bbox" in r])
        if rec is None:
            out.append({**cd, "category": "uncertain", "exclude": False, "confidence": 0.0,
                        "reason": "回放时找不到对应的历史判决，按保留处理",
                        "_tally": {}, "_n_votes": 0, "_ai": False,
                        "_replay_unmatched": True})
            continue
        keep = {k: rec.get(k) for k in ("category", "exclude", "confidence", "reason",
                                        "_tally", "_n_votes", "_n_asked", "_decision_tally", "_split",
                                        "needs_human", "majority_exclude", "decided_by", "human_note",
                                        "agent_category", "agent_exclude",
                                        "override_of_unanimous") if k in rec}
        out.append({**cd, **keep, "_ai": bool(rec.get("_ai", False)), "_replayed": True,
                    "_replay_source_rank": rec.get("rank")})
    return out


# ------------------------------------------------------------------ queue
def load_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def queue_path(work_root: Path) -> Path:
    return Path(work_root) / "REVIEW" / QUEUE


def verdicts_path(work_root: Path) -> Path:
    return Path(work_root) / "REVIEW" / VERDICTS


def push_queue(work_root: Path, entry: dict) -> None:
    """Append one escalation. Entries are keyed on (sample, stage, reason) so a re-run
    updates the existing line instead of duplicating it."""
    p = queue_path(work_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = load_json(p, [])
    if entry.get("regions"):
        # region-level lines are identified by their reason text (region ranks)
        key = (entry["sample"], entry["stage"], entry["reason"])
        q = [e for e in q if (e["sample"], e["stage"], e["reason"]) != key]
    elif entry.get("spectrum"):
        # spectrum-level lines (autospectra, spectra_qc): one open line per spectrum.
        # Found on the 2026-09-05 real-data run: without this branch every spectrum
        # escalation fell into the stage-level rule below and overwrote the previous
        # one, so the queue kept only the last escalated spectrum per substrate.
        key = (entry["sample"], entry["stage"], entry["spectrum"])
        q = [e for e in q if (e["sample"], e["stage"], e.get("spectrum")) != key]
    else:
        # stage-level lines (audit, segment, stitch, stats ...): one open line per
        # (sample, stage); a re-run replaces it, a passed re-run clears it (clear_queue)
        q = [e for e in q if not (e["sample"] == entry["sample"] and e["stage"] == entry["stage"]
                                  and not e.get("regions") and e.get("status") == "open")]
    q.append({**entry, "status": entry.get("status", "open"),
              "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    p.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_queue(work_root: Path, sample: str, stage: str) -> None:
    """Drop the open stage-level line of (sample, stage): the stage passed on re-run."""
    p = queue_path(work_root)
    if not p.exists():
        return
    q = load_json(p, [])
    q2 = [e for e in q if not (e["sample"] == sample and e["stage"] == stage
                               and not e.get("regions") and e.get("status") == "open")]
    if len(q2) != len(q):
        p.write_text(json.dumps(q2, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_entry(work_root: Path, sample: str, stage: str, by: str, note: str) -> int:
    """A human closes a stage-level line (audit alarms, unfit spectra ...) with a note."""
    p = queue_path(work_root)
    q = load_json(p, [])
    n = 0
    for e in q:
        if e["sample"] == sample and e["stage"] == stage and e.get("status") == "open":
            e.update(status="resolved", resolved_by=by, resolve_note=note,
                     resolved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            n += 1
    p.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


# ------------------------------------------------------------------ CLI
def _cmd_list(work: Path):
    q = load_json(queue_path(work), [])
    if not q:
        print("复核队列为空"); return
    for e in q:
        print(f"[{e.get('status','open'):<4}] {e['sample']}/{e['stage']}  "
              f"conf {e.get('confidence')}  {e['reason']}")
        for r in e.get("regions", []):
            print(f"        #{r['rank']} {r.get('category')} 票 {r.get('_tally')}  "
                  f"key {r['region_key']}")
        for a in e.get("alarms", [])[:5]:
            print(f"        alarm {a}")
        if e.get("spectrum"):
            print(f"        spectrum {e['spectrum']}")
    hv = load_json(verdicts_path(work), [])
    print(f"\n已记录人工判决 {len(hv)} 条 -> {verdicts_path(work)}")


def _cmd_decide(work: Path, a):
    if not a.note or not a.note.strip():
        sys.exit("--note 必填：不写理由的覆盖就是不受审计的后门")
    if a.exclude == a.keep:
        sys.exit("--exclude 和 --keep 二选一")
    excl = work / a.sample / "03_regions" / f"{a.sample}_exclusions.json"
    rec = load_json(excl, {})
    v = next((x for x in rec.get("verdicts", []) if x.get("rank") == a.rank), None)
    if v is None:
        sys.exit(f"{excl} 里没有 rank={a.rank} 的区域（先跑一遍流水线）")
    entry = {"region_key": region_key(a.sample, v), "rank_at_decision": a.rank,
             "category": a.category, "exclude": bool(a.exclude),
             "decided_by": a.by, "note": a.note.strip(),
             "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "agent_at_decision": {"category": v.get("category"),
                                   "exclude": v.get("exclude"), "tally": v.get("_tally")}}
    p = verdicts_path(work)
    p.parent.mkdir(parents=True, exist_ok=True)
    hv = load_json(p, [])
    hv = [h for h in hv if not same_region(h["region_key"], entry["region_key"])]
    hv.append(entry)
    p.write_text(json.dumps(hv, ensure_ascii=False, indent=2), encoding="utf-8")
    # close matching queue lines
    qp = queue_path(work)
    q = load_json(qp, [])
    for e in q:
        if e["sample"] == a.sample and any(
                same_region(r["region_key"], entry["region_key"]) for r in e.get("regions", [])):
            e["status"] = "decided"
    qp.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"记录：{a.sample} #{a.rank} -> {'剔除' if a.exclude else '保留'} "
          f"({a.category or 'agent 类别'}) by {a.by}。下次 run.py 会应用并留档。")


def main():
    ap = argparse.ArgumentParser(description="人工复核队列")
    ap.add_argument("--work", required=True, help="work_root（和 config 里一致）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("decide")
    d.add_argument("--sample", required=True)
    d.add_argument("--rank", type=int, required=True)
    d.add_argument("--category", default="")
    d.add_argument("--exclude", action="store_true")
    d.add_argument("--keep", action="store_true")
    d.add_argument("--by", required=True, help="决定人")
    d.add_argument("--note", required=True, help="理由，必填")
    r = sub.add_parser("resolve", help="关闭一条阶段级上报（audit 报警、unfit 谱 …）")
    r.add_argument("--sample", required=True)
    r.add_argument("--stage", required=True)
    r.add_argument("--by", required=True)
    r.add_argument("--note", required=True, help="理由，必填")
    a = ap.parse_args()
    work = Path(a.work)
    if a.cmd == "list":
        _cmd_list(work)
    elif a.cmd == "resolve":
        if not a.note.strip():
            sys.exit("--note 必填")
        n = resolve_entry(work, a.sample, a.stage, a.by, a.note.strip())
        print(f"关闭 {n} 条 {a.sample}/{a.stage} 的上报")
    else:
        _cmd_decide(work, a)


if __name__ == "__main__":
    main()
