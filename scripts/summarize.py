"""Accuracy tables and paired tests from the episode / baseline outputs under $CE_WORK/runs.

  python scripts/summarize.py --qids-file splits/lite903.json --models qwen3-vl-32b-local glm-5.3-flash \
      --conds direct bench_agent single_call_no_graph single_call no_navigator clinicaleye \
      --pairs clinicaleye:single_call clinicaleye:no_navigator single_call:single_call_no_graph

Conditions: any condition of experiments/run_agent.py, plus `direct` (scripts/dtvqa_direct.py) and `bench_agent`
(scripts/dtvqa_agent.py, predicted-tool mode). Accuracy with a 95% case-clustered bootstrap interval (questions of the same
CT resampled together); pairs: paired exact McNemar test on the questions both conditions answered."""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from ctgraph import paths as P  # noqa: E402


def load(model: str, cond: str) -> dict:
    """qid -> correct (bool)"""
    out = {}
    if cond == "direct":
        for f in (P.RUNS / "dtvqa_direct" / model).glob("*.json"):
            r = json.load(open(f))
            if "correct" in r:
                out[r["qid"]] = bool(r["correct"])
    elif cond == "bench_agent":
        sys.path.insert(0, str(P.DTVQA_SRC))
        from deeptumorvqa.eval.metrics import extract_letter
        for f in (P.RUNS / "dtvqa_agent" / model / "predicted").glob("*.json"):
            r = json.load(open(f))
            out[r["qid"]] = extract_letter(str(r.get("raw_output") or r.get("pred_answer") or "")) == r["correct_option"]
    else:
        for f in (P.RUNS / "agent" / "eps" / model / cond).glob("*.json"):
            r = json.load(open(f))["result"]
            out[r["qid"]] = bool(r["correct"])
    return out


def mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))


def bootstrap_ci(qids, res, case_of, n=2000, seed=0):
    cases = sorted({case_of[q] for q in qids})
    by = {c: [res[q] for q in qids if case_of[q] == c] for c in cases}
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n):
        pick = rng.choice(len(cases), len(cases))
        v = [x for i in pick for x in by[cases[i]]]
        accs.append(np.mean(v))
    return np.percentile(accs, [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids-file", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b pairs for the paired McNemar test")
    a = ap.parse_args()
    qids = json.load(open(a.qids_file))["qids"]
    case_of = {}
    for l in open(P.TEST_QA):
        r = json.loads(l)
        case_of[r["qid"]] = r["image_id"]
    for m in a.models:
        print(f"== {m}  ({len(qids)} questions)")
        res = {c: load(m, c) for c in set(a.conds) | {x for p in a.pairs for x in p.split(":")}}
        for c in a.conds:
            have = [q for q in qids if q in res[c]]
            if not have:
                print(f"  {c:28s} no results")
                continue
            lo, hi = bootstrap_ci(have, res[c], case_of)
            print(f"  {c:28s} n={len(have):4d}  acc={np.mean([res[c][q] for q in have]):.3f}  [{lo:.3f}, {hi:.3f}]")
        for p in a.pairs:
            x, y = p.split(":")
            both = [q for q in qids if q in res[x] and q in res[y]]
            b = sum(res[x][q] and not res[y][q] for q in both)
            c = sum(res[y][q] and not res[x][q] for q in both)
            if both:
                print(f"  {x} vs {y}: n={len(both)}  {np.mean([res[x][q] for q in both]):.3f} vs "
                      f"{np.mean([res[y][q] for q in both]):.3f}  +{b}/-{c}  McNemar p={mcnemar(b, c):.3g}")
