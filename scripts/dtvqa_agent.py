"""DeepTumorVQA's own tool agent (released code, $CE_DTVQA_CODE) with our models on our question sets.

Faithful to `deeptumorvqa.eval.agent_evaluator.AgentEvaluator.run`: their system prompt (SYSTEM_PROMPT_AGENT), user prompt
(`build_prompt(q, "2d_image", "mc")`, text only), toolkit (`build_toolkit(mode, cache)`: segment_organ / measure /
lookup_medical_knowledge, answered from the released TotalSegmentator cache in "predicted" mode), max_steps 5 with the
tools dropped on the last step, image_id injected into every tool call, and their scoring (`eval.metrics.aggregate`).
Model call as in their reference vLLM agent (`reference/qwen3vl_agent.py`): OpenAI function calling, temperature 0,
max_tokens 512. Differences (documented): our loop runs a question subset in parallel and keeps one JSON per question
(resumable); assistant / tool messages carry OpenAI tool_call ids (required by the APIs); `--adapter prompt` (models
without a function-calling parser, e.g. MedGemma) describes the same tools in the prompt and parses a JSON tool call.
Calls go through ctgraph.llm.LLM (cache + token ledger).
Usage: python3 scripts/dtvqa_agent.py --model glm-5.3-flash --qids-file splits/lite903.json [--mode predicted]
       [--adapter native|prompt] [--workers 3]"""
import argparse, json, math, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from ctgraph import paths as P  # noqa: E402
sys.path.insert(0, str(P.DTVQA_SRC))
from ctgraph.llm import LLM, extract_json  # noqa: E402
from deeptumorvqa.data.prompt import SYSTEM_PROMPT_AGENT, build_prompt  # noqa: E402
from deeptumorvqa.eval.tools import build_toolkit  # noqa: E402
from deeptumorvqa.eval.metrics import aggregate  # noqa: E402

ROOT = P.WORK / "dtvqa_root"


def prepare_root():
    """benchmark/test_qa.json (all 10K, NaN -> None as in the released JSON) + the released tool caches."""
    qa = ROOT / "benchmark/test_qa.json"
    if not qa.exists():
        qa.parent.mkdir(parents=True, exist_ok=True)
        qs = []
        for l in open(P.TEST_QA):
            q = json.loads(l)
            qs.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in q.items()})
        json.dump({"questions": qs}, open(qa, "w"))
    (ROOT / "tool_cache").mkdir(exist_ok=True)
    for f in ("benchmark_totalsegmentator_cache.json", "benchmark_oracle_tool_cache.json"):
        if not (ROOT / "tool_cache" / f).exists():
            os.symlink(P.TOOL_CACHE / f, ROOT / "tool_cache" / f)
    return {q["qid"]: q for q in json.load(open(qa))["questions"]}


PROMPT_TOOLS = """

TOOLS (function calling): you may call these tools. To call tools, reply with ONLY a JSON object
{{"tool_calls": [{{"name": "<tool name>", "arguments": {{...}}}}]}} and nothing else; the results come back in the next
message. When you have enough evidence, reply with your final answer instead (no JSON tool call).
{schemas}"""


class Agent:
    def __init__(self, model, mode, adapter, max_steps=5):
        self.llm = LLM(model, tag=f"dtvqa_agent/{mode}")
        self.mode, self.adapter, self.max_steps = mode, adapter, max_steps
        cache = ROOT / "tool_cache" / ("benchmark_totalsegmentator_cache.json" if mode == "predicted" else "benchmark_oracle_tool_cache.json")
        self.tools = build_toolkit(mode, cache)
        self.schemas = [t.to_openai_tool() for t in self.tools]
        self.by_name = {t.name: t for t in self.tools}

    def chat(self, messages, tools):
        if self.adapter == "native":
            kw = dict(tools=tools, tool_choice="auto") if tools else {}
            r = self.llm.chat(messages, max_tokens=512, temperature=0.0, **kw)
            return r["text"], [{"id": t["id"], "name": t["name"], "arguments": t["arguments"]} for t in r.get("tool_calls") or []]
        # prompt adapter: same tools described in the system prompt; tool turns rendered as plain text
        msgs = []
        for m in messages:
            if m["role"] == "system":
                msgs.append({"role": "system", "content": m["content"] + (PROMPT_TOOLS.format(
                    schemas=json.dumps([s["function"] for s in tools], indent=1)) if tools else
                    "\n\nNo more tool calls are possible: give your final answer now.")})
            elif m["role"] == "assistant":
                tc = [{"name": t["function"]["name"], "arguments": json.loads(t["function"]["arguments"])} for t in m.get("tool_calls") or []]
                msgs.append({"role": "assistant", "content": json.dumps({"tool_calls": tc}) if tc else (m.get("content") or "")})
            elif m["role"] == "tool":
                msgs.append({"role": "user", "content": f"TOOL RESULT ({m['name']}): {m['content']}"})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        # merge consecutive user messages (some chat templates require alternating roles)
        merged = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"] == "user":
                merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n" + m["content"]}
            else:
                merged.append(m)
        r = self.llm.chat(merged, max_tokens=512, temperature=0.0)
        j = extract_json(r["text"]) if tools and "tool_calls" in (r["text"] or "") else {}
        calls = j.get("tool_calls") if isinstance(j, dict) and isinstance(j.get("tool_calls"), list) else []
        out = [{"id": f"call_{i}", "name": c.get("name"), "arguments": json.dumps(c.get("arguments") or {})}
               for i, c in enumerate(calls) if isinstance(c, dict)]
        return ("" if out else r["text"]), out

    def run_one(self, q):
        """= AgentEvaluator.run's per-question loop (their code), with tool_call ids."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT},
                    {"role": "user", "content": build_prompt(q, "2d_image", "mc")}]
        traj, final, content = [], "", ""
        for step in range(self.max_steps):
            offered = self.schemas if step < self.max_steps - 1 else []
            try:
                content, calls = self.chat(messages, offered)
            except Exception as e:
                final = f"[ERROR] {type(e).__name__}: {e}"
                traj.append({"step": step + 1, "error": final})
                break
            content = content or ""
            if not calls:
                final = content
                traj.append({"step": step + 1, "final_answer": content})
                break
            messages.append({"role": "assistant", "content": content, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else dict(c["arguments"] or {})
                except Exception:
                    args = {}
                args = dict(args) if isinstance(args, dict) else {}
                args["image_id"] = q["image_id"]
                tool = self.by_name.get(c["name"])
                if tool is None:
                    res = {"error": f"Unknown tool '{c['name']}'."}
                else:
                    try:
                        res = tool.execute(**args)
                    except Exception as e:
                        res = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": json.dumps(res)})
                traj.append({"step": step + 1, "tool": c["name"], "arguments": args, "result": res})
        else:
            final = content
        return {"qid": q["qid"], "image_id": q["image_id"], "question_subtype": q.get("question_subtype"),
                "question_type": q.get("question_type"), "raw_output": final, "pred_answer": final,
                "correct_option": q.get("correct_option"), "answer": q.get("answer"), "trajectory": traj}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--qids-file", required=True)
    ap.add_argument("--mode", default="predicted"); ap.add_argument("--adapter", default="native")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    Q = prepare_root()
    qids = json.load(open(a.qids_file))["qids"]
    out = P.RUNS / "dtvqa_agent" / a.model / a.mode; out.mkdir(parents=True, exist_ok=True)
    if not a.score_only:
        ag = Agent(a.model, a.mode, a.adapter)
        todo = [q for q in qids if not (out / f"{q}.json").exists()]
        print(f"{len(todo)} to run ({len(qids) - len(todo)} done)", flush=True)

        def job(qid):
            r = ag.run_one(Q[qid])
            if not str(r["pred_answer"]).startswith("[ERROR]"):
                json.dump(r, open(out / f"{qid}.json", "w"))
            return r
        with ThreadPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(job, todo)):
                if i % 25 == 0:
                    print(f"{i + 1}/{len(todo)} {r['qid']} pred {str(r['pred_answer'])[:40]!r} gt {r['correct_option']}", flush=True)
    res = [json.load(open(out / f"{q}.json")) for q in qids if (out / f"{q}.json").exists()]
    m = aggregate(res, Q, fmt="mc")
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict) or len(v) < 8}, indent=1)[:1500])
    json.dump({"model": a.model, "mode": a.mode, "adapter": a.adapter, "qids_file": a.qids_file, "n": len(res), "metrics": m},
              open(out.parent / f"metrics_{a.mode}_{Path(a.qids_file).stem}.json", "w"), indent=1)
