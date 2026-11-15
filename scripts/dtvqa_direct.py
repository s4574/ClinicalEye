"""DeepTumorVQA's own direct (non-agent) VQA baseline with our models on our question sets.
Faithful to `deeptumorvqa.eval.vqa_evaluator.HFVLMEvaluator.run` + `reference/qwen3vl_vqa.py`: input mode "2d_image"
(the released whole-volume PNG `benchmark/images_2d/whole/<image_id>.png`, one image per case), their system prompt
SYSTEM_PROMPT_VQA, their user prompt `build_prompt(q, "2d_image", "mc")`, temperature 0, max 256 new tokens, their
scoring (`extract_letter`). Calls go through ctgraph.llm.LLM (cache + ledger); GLM with thinking disabled (MODEL_KW).
One JSON per question under runs/dtvqa_direct/<model>/. Usage:
  python3 scripts/dtvqa_direct.py --model qwen3-vl-32b-local --qids-file splits/lite903.json [--workers 4]"""
import argparse, base64, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from ctgraph import paths as P  # noqa: E402
sys.path.insert(0, str(P.DTVQA_SRC)); sys.path.insert(0, str(R / "scripts"))
from ctgraph.llm import LLM, spent  # noqa: E402
from deeptumorvqa.data.prompt import SYSTEM_PROMPT_VQA, build_prompt  # noqa: E402
from deeptumorvqa.eval.metrics import extract_letter  # noqa: E402
from dtvqa_agent import prepare_root, ROOT  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--qids-file", required=True)
    ap.add_argument("--workers", type=int, default=4)
    # optional per-run cap in USD on top of what the ledger already holds
    ap.add_argument("--run-budget", type=float, default=None)
    a = ap.parse_args()
    Q = prepare_root()
    qids = json.load(open(a.qids_file))["qids"]
    out = P.RUNS / "dtvqa_direct" / a.model; out.mkdir(parents=True, exist_ok=True)
    llm = LLM(a.model, tag="dtvqa_direct", **({"budget_usd": spent() + a.run_budget} if a.run_budget is not None else {}))

    def job(qid):
        f = out / f"{qid}.json"
        if f.exists():
            return json.load(open(f))
        q = Q[qid]
        img = P.DTVQA_DATA / "benchmark/images_2d/whole" / f"{q['image_id']}.png"
        if not img.exists():
            return {"qid": qid, "error": f"missing {img.name}"}
        b64 = base64.b64encode(img.read_bytes()).decode()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_VQA},
                {"role": "user", "content": [{"type": "text", "text": build_prompt(q, "2d_image", "mc")},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        try:
            r = llm.chat(msgs, max_tokens=256, temperature=0.0)
        except Exception as e:
            return {"qid": qid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        rec = {"qid": qid, "image_id": q["image_id"], "question_subtype": q["question_subtype"], "raw_output": r["text"],
               "pred_letter": extract_letter(r["text"]), "correct_option": q["correct_option"]}
        rec["correct"] = rec["pred_letter"] == q["correct_option"]
        json.dump(rec, open(f, "w"))
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, qids))
    ok = [r for r in res if "correct" in r]
    errs = [r for r in res if "error" in r]
    print(f"{a.model}: {len(ok)}/{len(qids)} answered, accuracy {sum(r['correct'] for r in ok) / max(len(ok), 1):.3f}, "
          f"unparsed {sum(not r['pred_letter'] for r in ok)}, errors {len(errs)} {errs[:2]}")
