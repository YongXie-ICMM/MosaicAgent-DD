"""Vote counting in kimi_agents.AgentPool and its consumers (SYNTHESIS 2026-09-02 #6,
#13, #17, #10), without a key and without the network -- the client is a scripted
stand-in and urlopen is monkeypatched where the real client is under test.

  * a shared top count (1-1 after a failed vote, 1-1-1) is no majority: vote() returns
    None instead of the first-inserted key;
  * with decision_key the majority is taken on the boolean (keep/exclude), the label is
    the most frequent one on the winning side, and melt / film / contamination wobble
    inside the exclude classes is not a split;
  * vote_adaptive: 1 vote; +2 below the confidence gate or on an invalid answer; +2 when
    the 3 are split on the decision; capped at max_votes; voters 0.. in order;
  * stage_regions forces a split verdict to keep + needs_human and escalates with
    "去留分歧", never on label wobble; votes="adaptive" runs and records _tally/_n_votes;
  * KimiClient: 5 retries with the longer 429 backoff, the 16384 floor for kimi-*
    models applied after the cache key, truncation at the ceiling is a failure and is
    not cached;
  * replay of the adaptive_votes experiment (skipped when its votes.json is absent):
    the adaptive rule reproduces the recorded keep/exclude decision on all 23 regions
    in every ordering of the six fresh votes, at ~1.7 calls per region.

Run:  cd MosaicAgent && python3 -m pytest flakepipeline/tests -q
"""
import io
import itertools
import json
import re
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FP = Path(__file__).resolve().parents[1]
ROOT = FP.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FP))

import kimi_agents as KA          # noqa: E402

ROLE = KA.Role(name="t", system="s", schema_hint='{"category": "...", "exclude": true/false}')
SEED = re.compile(r"你是第 (\d+) 位独立评审")


def _ans(cat, exclude, conf=0.9):
    return {"category": cat, "exclude": exclude, "confidence": conf, "reason": "r"}


class ScriptedClient:
    """Duck-types KimiClient. answers maps voter index (None for the unseeded prompt)
    to the reply; a callable is invoked; missing voters get None (a failed call)."""
    available = True

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.usage = KA.Usage()

    def chat(self, text, images=None, system=None, max_tokens=2048, want_json=True):
        m = SEED.search(text)
        voter = int(m.group(1)) - 1 if m else None
        self.calls.append(voter)
        a = self.answers.get(voter)
        a = a(voter) if callable(a) else a
        return dict(a) if isinstance(a, dict) else a


def _pool(answers, votes=3, workers=6):
    return KA.AgentPool(ScriptedClient(answers), workers=workers, votes=votes)


def _build(item):
    return f"region {item}", None


# ------------------------------------------------------------------ vote / tally
def test_shared_top_count_is_no_majority():
    w, tally, answers = _pool({0: _ans("a", True), 1: _ans("b", True), 2: _ans("c", True)}).vote(
        ROLE, "p")
    assert w is None and tally == {"a": 1, "b": 1, "c": 1} and len(answers) == 3
    # 1-1 after one failed vote
    w, tally, _ = _pool({0: _ans("a", True), 1: _ans("b", True)}).vote(ROLE, "p")
    assert w is None and tally == {"a": 1, "b": 1}
    # a real majority still wins
    w, tally, _ = _pool({0: _ans("a", True), 1: _ans("b", True), 2: _ans("a", True)}).vote(
        ROLE, "p")
    assert w == "a" and tally == {"a": 2, "b": 1}
    # no valid answer at all
    assert _pool({}).vote(ROLE, "p") == (None, {}, [])


def test_decision_key_majority_and_label_from_winning_side():
    # S05mg_r03 in the adaptive_votes experiment: three labels, 2/3 votes exclude. The
    # old tally (first key reaching the max) kept the region; the decision must win.
    votes = {0: _ans("multilayer_crystals", False, 0.6),
             1: _ans("substrate_damage", True, 0.82),
             2: _ans("contamination", True, 0.6)}
    w, tally, answers = _pool(votes).vote(ROLE, "p", decision_key="exclude")
    assert w == "substrate_damage"            # tie on the exclude side -> higher confidence
    assert tally == {"multilayer_crystals": 1, "substrate_damage": 1, "contamination": 1}
    assert len(answers) == 3
    r = _pool(votes).map(ROLE, [1], _build, decision_key="exclude", verbose=False)[0]
    assert r["exclude"] is True and r["category"] == "substrate_damage"
    assert r["_decision_tally"] == {"true": 2, "false": 1} and r["_split"] is True
    assert r["_tally"] == tally and r["_n_votes"] == 3 and r["_n_asked"] == 3


def test_label_wobble_inside_exclude_is_not_a_split():
    votes = {0: _ans("precursor_melt", True, 0.78), 1: _ans("continuous_film", True, 0.66),
             2: _ans("precursor_melt", True, 0.8)}
    r = _pool(votes).map(ROLE, [1], _build, decision_key="exclude", verbose=False)[0]
    assert r["category"] == "precursor_melt" and r["exclude"] is True
    assert r["_split"] is False and r["_decision_tally"] == {"true": 3, "false": 0}
    assert r["confidence"] == 0.8               # the representative is the best on the winning side
    # without decision_key the same wobble is a plain 2-1 label majority
    w, _, _ = _pool(votes).vote(ROLE, "p")
    assert w == "precursor_melt"


def test_decision_tie_and_string_booleans():
    # one vote lost -> 1-1 on exclude -> no majority -> None from map
    votes = {0: _ans("precursor_melt", True), 1: _ans("multilayer_crystals", False)}
    assert _pool(votes).map(ROLE, [1], _build, decision_key="exclude", verbose=False) == [None]
    w, tally, _ = _pool(votes).vote(ROLE, "p", decision_key="exclude")
    assert w is None and tally == {"precursor_melt": 1, "multilayer_crystals": 1}
    # a model that writes "false" as a string must not be counted as True
    votes = {0: _ans("multilayer_crystals", "false"), 1: _ans("multilayer_crystals", "false"),
             2: _ans("precursor_melt", "true")}
    r = _pool(votes).map(ROLE, [1], _build, decision_key="exclude", verbose=False)[0]
    assert r["exclude"] is False and r["_decision_tally"] == {"true": 1, "false": 2}
    # an answer without the decision field cannot be counted
    votes = {0: {"category": "precursor_melt", "confidence": 0.9}, 1: _ans("thick_flake", False),
             2: _ans("thick_flake", False)}
    r = _pool(votes).map(ROLE, [1], _build, decision_key="exclude", verbose=False)[0]
    assert r["_n_votes"] == 2 and r["_tally"] == {"thick_flake": 2} and r["_n_asked"] == 3


def test_single_vote_path_is_uniform():
    p = _pool({None: _ans("thick_flake", False, 0.7)}, votes=1)
    r = p.map(ROLE, [1], _build, decision_key="exclude", verbose=False)[0]
    assert r["_tally"] == {"thick_flake": 1} and r["_n_votes"] == 1 and r["_split"] is False
    assert p.map(ROLE, [1], _build, verbose=False)[0]["_n_asked"] == 1
    # an invalid single answer is None, not a dict without the key
    assert _pool({None: {"_unparsed": "..."}}, votes=1).map(ROLE, [1], _build, verbose=False) == [None]
    # no key: nothing is asked
    c = ScriptedClient({}); c.available = False
    assert KA.AgentPool(c).map(ROLE, [1, 2], _build, verbose=False) == [None, None]
    assert KA.AgentPool(c).vote_adaptive(ROLE, "p") == (None, {}, [])


# ------------------------------------------------------------------ adaptive
def test_vote_adaptive_gate_split_and_cap():
    # confident first vote: one call, done
    c = ScriptedClient({0: _ans("precursor_melt", True, 0.9)})
    pool = KA.AgentPool(c, votes="adaptive")
    assert pool.adaptive and pool.votes == 3          # vote() stays the fixed default
    w, tally, answers = pool.vote_adaptive(ROLE, "p", decision_key="exclude")
    assert w == "precursor_melt" and tally == {"precursor_melt": 1} and len(answers) == 1
    assert c.calls == [0]

    # low confidence: two more, unanimous, stop at 3
    c = ScriptedClient({0: _ans("precursor_melt", True, 0.7), 1: _ans("precursor_melt", True),
                        2: _ans("continuous_film", True)})
    r = KA.AgentPool(c, votes="adaptive").map(ROLE, [1], _build, decision_key="exclude",
                                              verbose=False)[0]
    assert sorted(c.calls) == [0, 1, 2] and r["_n_asked"] == 3 and r["_split"] is False
    assert r["category"] == "precursor_melt" and r["exclude"] is True

    # split on exclude after 3: two more, capped at 5, majority decides, still split
    c = ScriptedClient({0: _ans("multilayer_crystals", False, 0.7),
                        1: _ans("precursor_melt", True, 0.72), 2: _ans("precursor_melt", True, 0.65),
                        3: _ans("multilayer_crystals", False, 0.6),
                        4: _ans("multilayer_crystals", False, 0.75)})
    r = KA.AgentPool(c, votes="adaptive").map(ROLE, [1], _build, decision_key="exclude",
                                              verbose=False)[0]
    assert sorted(c.calls) == [0, 1, 2, 3, 4] and r["_n_asked"] == 5 and r["_n_votes"] == 5
    assert r["exclude"] is False and r["_decision_tally"] == {"true": 2, "false": 3}
    assert r["_split"] is True and r["category"] == "multilayer_crystals"

    # max_votes=3 caps the escalation; the explicit kwarg overrides the pool's default
    c = ScriptedClient({0: _ans("multilayer_crystals", False, 0.7),
                        1: _ans("precursor_melt", True), 2: _ans("precursor_melt", True)})
    r = KA.AgentPool(c, votes=3).map(ROLE, [1], _build, decision_key="exclude", adaptive=True,
                                     max_votes=3, verbose=False)[0]
    assert sorted(c.calls) == [0, 1, 2] and r["exclude"] is True and r["_split"] is True

    # invalid first answer (failed call): treated like a low-confidence one
    c = ScriptedClient({1: _ans("thick_flake", False), 2: _ans("thick_flake", False)})
    w, tally, answers = KA.AgentPool(c).vote_adaptive(ROLE, "p", decision_key="exclude")
    assert w == "thick_flake" and tally == {"thick_flake": 2} and sorted(c.calls) == [0, 1, 2]

    # without decision_key a label disagreement is the split
    c = ScriptedClient({0: _ans("a", True, 0.5), 1: _ans("b", True), 2: _ans("a", True),
                        3: _ans("a", True), 4: _ans("a", True)})
    w, tally, _ = KA.AgentPool(c).vote_adaptive(ROLE, "p")
    assert w == "a" and tally == {"a": 4, "b": 1} and len(c.calls) == 5

    # an exact tie after the cap is None
    c = ScriptedClient({0: _ans("a", True, 0.5), 1: _ans("b", False), 2: None, 3: _ans("a", True),
                        4: _ans("b", False)})
    assert KA.AgentPool(c).map(ROLE, [1], _build, decision_key="exclude", adaptive=True,
                               verbose=False) == [None]


# ------------------------------------------------------------------ stage_regions
def _synthetic(tmp_path):
    H, W = 400, 400
    mask = np.full((H, W), 2, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    pool = (yy - 200) ** 2 + (xx - 220) ** 2 < 60 ** 2
    mask[pool] = 3
    mask[50:70, 50:70] = 3
    photo = np.full((H, W, 3), 200, np.uint8)
    photo[pool] = (90, 110, 170)
    mp = tmp_path / "mosaic.png"
    Image.fromarray(photo).save(mp)
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "valid.npy", np.ones((H, W), bool))
    return mp, tmp_path / "mask.npy", tmp_path / "valid.npy"


def _run_regions(tmp_path, monkeypatch, answers, votes):
    import stages
    client = ScriptedClient(answers)
    monkeypatch.setattr(KA, "KimiClient", lambda *a, **k: client)
    ctx = {"sample": "T", "work": tmp_path / "_work", "votes": votes,
           "min_region_pct": 1.0, "max_regions": 8, "region_scale_div": 1,
           "no_refine": True}
    ctx["work"].mkdir(exist_ok=True)
    return stages.stage_regions(ctx, *_synthetic(tmp_path)), client


def test_stage_regions_forces_keep_on_exclude_split(tmp_path, monkeypatch):
    res, client = _run_regions(tmp_path, monkeypatch,
                               {0: _ans("precursor_melt", True, 0.9),
                                1: _ans("multilayer_crystals", False, 0.7),
                                2: _ans("precursor_melt", True, 0.85)}, votes=3)
    v = res.data["verdicts"]
    assert len(v) == 2 and all(x["_n_votes"] == 3 for x in v)
    assert all(x["exclude"] is False and x["needs_human"] is True
               and x["majority_exclude"] is True and x["_split"] for x in v)
    assert res.data["excluded"] == 0 and res.data["needs_human"] == 2
    assert res.escalate and "去留分歧" in res.escalate_reason
    assert "票数分歧" not in res.escalate_reason
    assert res.data["kimi_calls"] == 6 and len(client.calls) == 6
    rec = json.loads((tmp_path / "_work" / "03_regions" / "T_exclusions.json").read_text())
    assert rec["votes"] == 3 and rec["kimi_calls"] == 6
    assert rec["verdicts"][0]["_decision_tally"] == {"true": 2, "false": 1}
    assert res.data["kimi_usage"]["calls"] == 0        # the scripted client never hit the API


def test_stage_regions_label_wobble_is_not_escalated(tmp_path, monkeypatch):
    res, _ = _run_regions(tmp_path, monkeypatch,
                          {0: _ans("precursor_melt", True, 0.78),
                           1: _ans("continuous_film", True, 0.66),
                           2: _ans("precursor_melt", True, 0.8)}, votes=3)
    assert not res.escalate and res.ok
    assert res.data["excluded"] == 2 and res.data["needs_human"] == 0
    assert all(x["category"] == "precursor_melt" and "majority_exclude" not in x
               for x in res.data["verdicts"])
    assert "去留全票一致 2/2" in res.evidence


def test_stage_regions_adaptive_votes(tmp_path, monkeypatch):
    res, client = _run_regions(tmp_path, monkeypatch,
                               {0: _ans("precursor_melt", True, 0.9)}, votes="adaptive")
    v = res.data["verdicts"]
    assert not res.escalate and res.data["excluded"] == 2
    assert all(x["_n_votes"] == 1 and x["_n_asked"] == 1 and x["_tally"] == {"precursor_melt": 1}
               for x in v)
    assert res.data["kimi_calls"] == 2 and client.calls == [0, 0]
    rec = json.loads((tmp_path / "_work" / "03_regions" / "T_exclusions.json").read_text())
    assert rec["votes"] == "adaptive"
    # a tie after the cap is uncertain / keep and escalated
    res, client = _run_regions(tmp_path, monkeypatch,
                               {0: _ans("precursor_melt", True, 0.5), 1: _ans("thick_flake", False),
                                2: None, 3: _ans("precursor_melt", True), 4: _ans("thick_flake", False)},
                               votes="adaptive")
    assert res.escalate and "uncertain" in res.escalate_reason
    assert all(x["category"] == "uncertain" and x["exclude"] is False for x in res.data["verdicts"])
    assert sorted(client.calls) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_votes_setting_and_other_stages_keep_fixed_three():
    import stages
    assert stages.votes_setting({}) == (False, 3)
    assert stages.votes_setting({"votes": 5}) == (False, 5)
    assert stages.votes_setting({"votes": "adaptive"}) == (True, 3)
    assert stages.votes_setting({"votes": " Adaptive "}) == (True, 3)


# ------------------------------------------------------------------ KimiClient
class _Resp(io.BytesIO):
    pass


def _ok_body(content, finish="stop"):
    return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()


def test_client_retries_backoff_and_reasoning_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(KA, "load_env", lambda: {"KIMI_API_KEY": "k"})
    sleeps, payloads = [], []
    monkeypatch.setattr(KA.time, "sleep", lambda s: sleeps.append(s))

    def limited(req, timeout=0):
        payloads.append(json.loads(req.data))
        raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b"slow down"))
    monkeypatch.setattr(KA.urllib.request, "urlopen", limited)
    c = KA.KimiClient(model="kimi-k3", cache_dir=str(tmp_path / "c"))
    assert c.max_retries == 5
    assert c.chat("q", max_tokens=8192) is None
    assert len(payloads) == 5 and sleeps == [5.0, 10.0, 20.0, 40.0]
    assert all(p["max_tokens"] == 16384 for p in payloads)      # reasoning floor
    assert c.usage.failed == 1 and not list((tmp_path / "c").glob("*.json"))

    # the cache key is computed on the caller's budget, so a cache written before the
    # floor existed (max_tokens 8192 in the payload) is still a hit
    old_payload = {"model": "kimi-k3", "max_tokens": 8192, "temperature": 1.0,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]}
    (tmp_path / "c" / f"{c._cache_key(old_payload)}.json").write_text(
        json.dumps({"raw": '{"category": "cached"}'}), encoding="utf-8")
    assert c.chat("q", max_tokens=8192) == {"category": "cached"} and c.usage.cached == 1

    # a non-reasoning model keeps its budget and the short 5xx backoff
    payloads.clear(); sleeps.clear()
    calls = {"n": 0}

    def flaky(req, timeout=0):
        payloads.append(json.loads(req.data))
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "down", {}, io.BytesIO(b""))
        return _Resp(_ok_body('{"a": 1}'))
    monkeypatch.setattr(KA.urllib.request, "urlopen", flaky)
    m = KA.KimiClient(model="moonshot-v1-128k", cache_dir=str(tmp_path / "m"))
    assert m.chat("q", max_tokens=2048) == {"a": 1}
    assert [p["max_tokens"] for p in payloads] == [2048, 2048] and sleeps == [3]
    assert m.usage.calls == 1 and m.usage.prompt_tokens == 10

    # truncation at the ceiling is a failure, not a cached empty reply
    payloads.clear()
    monkeypatch.setattr(KA.urllib.request, "urlopen",
                        lambda req, timeout=0: (payloads.append(json.loads(req.data)),
                                                _Resp(_ok_body("", "length")))[1])
    k = KA.KimiClient(model="kimi-k3", cache_dir=str(tmp_path / "k"))
    assert k.chat("q2", max_tokens=8192) is None
    assert len(payloads) == 1 and k.usage.truncated == 1 and k.usage.failed == 1
    assert not list((tmp_path / "k").glob("*.json"))
    # ...while a non-reasoning model still gets the one bump-and-retry
    payloads.clear()
    m2 = KA.KimiClient(model="moonshot-v1-128k")
    assert m2.chat("q2", max_tokens=2048) is None
    assert [p["max_tokens"] for p in payloads] == [2048, 16384] and m2.usage.truncated == 2


# ------------------------------------------------------------------ experiment replay
VOTES_JSON = FP / "_work_exp" / "adaptive_votes" / "votes.json"
RESULTS_JSON = FP / "docs" / "experiments_20260902" / "adaptive_votes" / "results.json"


@pytest.mark.skipif(not (VOTES_JSON.exists() and RESULTS_JSON.exists()),
                    reason="adaptive_votes experiment votes.json not on this machine")
def test_replay_adaptive_votes_experiment():
    """SYNTHESIS §3 guardrail: on the 23 regions of the experiment the adaptive rule must
    reproduce the recorded keep/exclude decision in every ordering of the six fresh
    votes (the experiment reports 100% over 720 orderings at 1.73 calls/region)."""
    votes = json.load(open(VOTES_JSON, encoding="utf-8"))["votes"]
    recorded = {r["id"]: r["recorded"] for r in
                json.load(open(RESULTS_JSON, encoding="utf-8"))["regions"]}
    voters = ["none", "0", "1", "2", "3", "4"]
    disagreements, calls, n = [], 0, 0
    for rid, rec in recorded.items():
        fresh = [votes[f"{rid}|{v}"]["answer"] for v in voters]
        for order in itertools.permutations(range(6)):
            c = ScriptedClient({k: fresh[order[k]] for k in range(6)})
            r = KA.AgentPool(c, votes="adaptive").map(ROLE, [rid], _build, decision_key="exclude",
                                                      verbose=False)[0]
            n += 1
            calls += len(c.calls)
            if r is None or bool(r["exclude"]) != bool(rec["exclude"]):
                disagreements.append((rid, order, None if r is None else r["_decision_tally"]))
    assert n == 23 * 720
    assert disagreements == []
    assert 1.5 <= calls / n <= 2.0, calls / n
