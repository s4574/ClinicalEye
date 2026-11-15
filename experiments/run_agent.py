"""Run the agent / single-call conditions of the paper on a question set; one JSON per episode.

  python experiments/run_agent.py --models qwen3-vl-32b-local --conds clinicaleye no_navigator single_call \
      --qids-file splits/lite903.json --workers 4

Episodes go to $CE_WORK/runs/agent/eps/<model>/<cond>/<qid>.json (skipped when present, so a crashed run resumes); every LLM
call is cached ($CE_LLM_CACHE). Results: python scripts/summarize.py.

Inputs of every condition: the question text and the CT with TotalSegmentator outputs. Benchmark metadata (organ, lesion,
question type / subtype) never reaches the agent; the organ field is used only to CONSTRUCT the masked-organ experiment
(which organ to remove from the segmentation).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctgraph import paths as P  # noqa: E402
from ctgraph import slicepct  # noqa: E402
from ctgraph.agent import Agent, CTEnv, parse_options, question_stem  # noqa: E402
from ctgraph.graph import build_graph, load_graph, save_graph  # noqa: E402
from ctgraph.knowledge import knowledge_text  # noqa: E402
from ctgraph.llm import LLM, extract_json, pmap, spent, user_msg  # noqa: E402
from ctgraph.navi5 import Navi5Agent  # noqa: E402
from ctgraph.views import organ_structures  # noqa: E402
from ctgraph.volume import TS_DIR, Case, load_ts_case  # noqa: E402

OUT = P.RUNS / "agent"
GRAPHS_TS = P.CACHE / "graphs_ts"
TS_TASKS = ["total", "liver_vessels", "liver_segments"]

_NAV = dict(navi=5, views="vision", seg="ts", max_renders=2, anatomy=True, v3=True, v3b=True, nv5_slice_pct=True,
            llm_sample=1, temperature=1.0)
_B0 = dict(baseline=True, seg="ts", slice_pct=True, kb=True, infer_organ=True, llm_sample=1, temperature=1.0)
# condition -> configuration. Internal ids of the development code in brackets.
CONDS = {
    # full method: reader VLM + navigator [NV5v3b_r1]
    "clinicaleye": dict(_NAV),
    # no navigator: the reader calls the same rendering rules itself [NV5v3b_self_r1]
    "no_navigator": dict(_NAV, views="self"),
    # single call: organ inferred from the question, overview images of it + anatomical-graph summary + criteria
    # [B0s_ts_kb_inf_r1]; the same without the anatomical-graph summary [B0s_ts_kb_inf_nograph_r1]
    "single_call": dict(_B0),
    "single_call_no_graph": dict(_B0, graph_mode="none"),
    # ablations
    "clinicaleye_no_overlay": dict(_NAV, no_overlay=True),                     # [NV5v3b_noov_r1]
    "no_navigator_no_overlay": dict(_NAV, views="self", no_overlay=True),       # [NV5v3b_self_noov_r1]
    "no_navigator_memory": dict(_NAV, nomem_nav=True),                         # [NV5v3b_navnomem_r1]
    "no_reader_memory": dict(_NAV, nomem_vlm=True),                            # [NV5v3b_vlmnomem_r1]
    "grounded_navigator": dict(_NAV, v3c=True),                                # [NV5v3c_r1]
    "render_cap20": dict(_NAV, max_renders=20),                                # [NV5v3b_mr20_r1]
    # second independent samples (same prompts, different sampling draw)
    "clinicaleye_sample2": dict(_NAV, llm_sample=2),                           # [NV5v3b_r2]
    "no_navigator_sample2": dict(_NAV, views="self", llm_sample=2),            # [NV5v3b_self_r2]
    # missing-tool simulation: the question's organ removed from the segmentation (MASK_MAP)
    "clinicaleye_masked_organ": dict(_NAV, mask_org=True),                     # [NV5v3b_mask_r1]
    "no_navigator_masked_organ": dict(_NAV, views="self", mask_org=True),       # [NV5v3b_self_mask_r1]
    "single_call_masked_organ": dict(_B0, mask_org=True),                      # [B0s_ts_kb_inf_mask_r1]
}

MASK_MAP = {"pancreas": ["pancreas"], "pancreatic": ["pancreas"],
            "kidney": ["kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right"]}
MASK_MAP["left kidney"] = MASK_MAP["right kidney"] = MASK_MAP["kidney"]      # both kidneys (the other side would be a cue)
# masking the liver also drops the liver-segment and liver-vessel (incl. liver_tumor) layers, which would reveal it
MASK_MAP.update({"liver": ["liver"], "spleen": ["spleen"], "colon": ["colon"]})
MASK_LAYERS = {"liver": ["liver_segments", "liver_vessels"]}


def ts_ready(cid: str) -> bool:
    return all((TS_DIR / cid / f"{t}.nii.gz").exists() for t in TS_TASKS)


def mask_names(organ_field: str) -> list[str]:
    return sorted({n for o in str(organ_field).split(",") for n in MASK_MAP.get(o.strip(), [])})


def _graph(case, gp: Path):
    """anatomical graph of a case, built on first use and written atomically"""
    if gp.exists():
        return load_graph(gp)
    g = build_graph(case)
    gp.parent.mkdir(parents=True, exist_ok=True)
    tmp = gp.with_suffix(f".tmp{threading.get_ident()}")
    save_graph(g, tmp)
    os.replace(tmp, gp)
    return g


def load_ts(cid: str):
    case = load_ts_case(cid)
    return case, _graph(case, GRAPHS_TS / f"{cid}.json")


def load_ts_masked(cid: str, drop: list[str]):
    case = load_ts_case(cid)
    keep = [n for n in case.names if n not in drop]
    lut = np.zeros(len(case.names) + 1, np.uint8)
    for i, n in enumerate(case.names, start=1):
        if n in keep:
            lut[i] = keep.index(n) + 1
    meta = dict(case.meta)
    drop_layers = {l for n in drop for l in MASK_LAYERS.get(n, [])}
    if drop_layers:
        meta["layers"] = {k: v for k, v in case.layers.items() if k not in drop_layers}
    mcase = Case(case.case_id, case.ct, lut[case.labels], keep, case.spacing, meta)
    return mcase, _graph(mcase, GRAPHS_TS.parent / "graphs_ts_masked" / f"{cid}__{'-'.join(drop)}.json")


def items(qids: set) -> list[dict]:
    return [it for it in map(json.loads, open(P.TEST_QA)) if it["qid"] in qids]


ORGAN_CHOICES = ["liver", "spleen", "pancreas", "kidney", "left kidney", "right kidney", "colon"]
ORGAN_PROMPT = """You will answer a multiple-choice question about an abdominal CT scan. Before looking at the scan, decide which
organ(s) must be examined to answer it. Choose ONE or TWO from this list (exact spelling): liver, spleen, pancreas, kidney
(= both kidneys), left kidney, right kidney, colon.

QUESTION: {question}
OPTIONS: {options}

Reply with JSON only: {{"organs": ["<organ>", ...], "reason": "<few words>"}}"""


def infer_organ(model: str, cond: str, it: dict, llm_kw: dict) -> tuple[tuple[str, str], float]:
    """single-call conditions: the target organ inferred by the same VLM from the question text (one text-only call)"""
    llm = LLM(model, tag=f"agent/{cond}/organ", **llm_kw)
    msg = [user_msg(ORGAN_PROMPT.format(question=question_stem(it["mc_question"]), options=json.dumps(parse_options(it["mc_question"]))))]
    r = llm.chat(msg, max_tokens=200)
    j = extract_json(r["text"]) or {}
    org = [o.strip().lower() for o in (j.get("organs") or []) if isinstance(o, str) and o.strip().lower() in ORGAN_CHOICES][:2]
    if not org:     # unusable reply: one resend as a new sample, then a fixed fallback (counted in the results)
        r = llm.chat(msg, max_tokens=200, cache_salt="retry1")
        j = extract_json(r["text"]) or {}
        org = [o.strip().lower() for o in (j.get("organs") or []) if isinstance(o, str) and o.strip().lower() in ORGAN_CHOICES][:2]
    return (",".join(org), "ok") if org else ("liver", "fallback"), r["cost"]


def _progress(msg: str):
    with open(OUT / "progress.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def run_one(model, cond, it, max_steps):
    path = OUT / "eps" / model / cond / f"{it['qid']}.json"
    if path.exists():
        return json.load(open(path))["result"]
    res = _run_one(model, cond, it, max_steps)
    _progress(f"{model} {cond} {it['qid']} " + (f"ERROR {res['error'][:120]}" if "error" in res else
              f"correct={res['correct']} steps={res['n_steps']} cost={res['cost']}"))
    return res


def _run_one(model, cond, it, max_steps):
    path = OUT / "eps" / model / cond / f"{it['qid']}.json"
    cfg = dict(CONDS[cond])
    baseline = cfg.pop("baseline", False)
    cfg.pop("seg")
    slice_pct = cfg.pop("slice_pct", False)
    infer = cfg.pop("infer_organ", False)
    graph_mode = cfg.pop("graph_mode", "full")
    kb = cfg.pop("kb", False)
    mask_org = cfg.pop("mask_org", False)
    llm_kw = {k: cfg.pop(k2) for k, k2 in (("sample", "llm_sample"), ("temperature", "temperature")) if k2 in cfg}
    navi = cfg.pop("navi", False)
    if navi:
        nav, per = LLM(model, tag=f"agent/{cond}/nav", **llm_kw), LLM(model, tag=f"agent/{cond}/per", **llm_kw)
        agent = Navi5Agent(per, nav, navi_mode=cfg.pop("views"), max_renders=cfg.pop("max_renders"), anatomy=cfg.pop("anatomy"),
                           v3=cfg.pop("v3"), v3b=cfg.pop("v3b"), slice_pct=cfg.pop("nv5_slice_pct"), v3c=cfg.pop("v3c", False),
                           nomem_nav=cfg.pop("nomem_nav", False), nomem_vlm=cfg.pop("nomem_vlm", False),
                           no_overlay=cfg.pop("no_overlay", False))
    else:
        agent = Agent(LLM(model, tag=f"agent/{cond}", **llm_kw), max_steps=max_steps, **cfg)
    try:
        if not ts_ready(it["image_id"]):
            return dict(model=model, cond=cond, qid=it["qid"], error="SKIP totalsegmentator outputs not ready")
        case, g = load_ts_masked(it["image_id"], mask_names(it["organ"])) if mask_org else load_ts(it["image_id"])
        if navi:     # no benchmark metadata reaches the agent
            ep = agent.run(case, g, it)
        else:
            (organ_field, organ_status), organ_cost = infer_organ(model, cond, it, llm_kw)
            env = CTEnv(case, g, organ_structures(organ_field))
            if mask_org and not env.targets:      # the inferred organ was masked: one overview of the upper abdomen instead
                env.targets = [n for n in ("liver", "spleen", "stomach") if n in g["nodes"]]
                env.union_overview = True
            env.graph_mode = graph_mode
            if slice_pct and re.search(r"\bslice", it["mc_question"], re.I):
                env.extra_text = slicepct.describe(it["image_id"], case.shape[2] - 1, source="ts")
            if kb and knowledge_text(it["mc_question"]):
                env.extra_text = (env.extra_text + "\n" if env.extra_text else "") + knowledge_text(it["mc_question"])
            ep = agent.baseline(env, it)
            ep.cost += organ_cost
            ep.final["organ_inferred"], ep.final["organ_status"] = organ_field, organ_status
    except Exception as e:
        return dict(model=model, cond=cond, qid=it["qid"], error=f"{type(e).__name__}: {str(e)[:300]}")
    pred = str(ep.final.get("answer", "")).strip().upper()[:1]
    steps = [s for s in ep.steps if "reply" in s]
    res = dict(model=model, cond=cond, qid=it["qid"], subtype=it["question_subtype"], pred=pred,
               correct=pred == it["correct_option"], conf=ep.final.get("confidence"), n_steps=len(steps),
               n_evidence=len(ep.evidence), cost=round(ep.cost, 4),
               step_conf=[(s["reply"].get("best_option"), s["reply"].get("confidence")) if isinstance(s["reply"], dict) else None
                          for s in steps])
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"result": res, "item": it, "episode": asdict(ep)}, open(path, "w"), default=str, indent=1)
    return res


def run(models, conds, workers, max_steps, qids):
    its = items(qids)
    jobs = [(m, c, it) for m in models for c in conds for it in its]
    print(f"{len(its)} items x {len(models)} models x {len(conds)} conds = {len(jobs)} episodes; ledger total ${spent():.2f}", flush=True)
    res = pmap(lambda j: run_one(*j, max_steps), jobs, workers)
    by = defaultdict(list)
    for r in res:
        if "correct" in r:
            by[(r["model"], r["cond"])].append(r["correct"])
    for (m, c), v in sorted(by.items()):
        print(f"{m:24s} {c:28s} n={len(v):4d} acc={np.mean(v):.4f}")
    print(f"errors / skipped: {sum('error' in r for r in res)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", required=True, choices=sorted(CONDS))
    ap.add_argument("--qids-file", required=True, help="json with a 'qids' list (e.g. splits/lite903.json)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run(a.models, a.conds, a.workers, a.max_steps, set(json.load(open(a.qids_file))["qids"]))
