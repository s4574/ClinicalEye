#!/usr/bin/env python3
"""Score 3DMedAgent's raw outputs against our question set (HANDOFF section 6).

Reads the per-image JSON records 3DMedAgent writes under --save-dir, joins them
back to our `qid` through `case_idx` (= row index of the VQA csv, which
select_target_rows assigns with reset_index BEFORE any filtering), and writes
predictions.jsonl + summary.json.

Questions whose preprocessing failed keep their SKIP marker and are listed
separately instead of being silently dropped.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import csv

LETTER_RE = re.compile(r"\b([A-F])\b")


def first_letter(text):
    """First A-F option letter in their GPT_summarized_result, else None.

    Their pipeline_utils.normalized_final_answer() already collapses the model's
    final_answer to one character when it can (fullmatch [A-Z], else the first
    standalone \\b[A-Z]\\b, else text[:1]), so this field is normally a single
    letter. We accept a bare letter or a standalone letter, and deliberately do
    NOT fall back to "any A-F character anywhere": that would read the 'A' out
    of a refusal like "I cannot determine this" and score it as an answer.
    Unparsable stays None and is counted wrong but kept (HANDOFF 6).
    """
    s = str(text or "").strip().upper()
    if len(s) == 1 and "A" <= s <= "F":
        return s
    m = LETTER_RE.search(s)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-csv", required=True)
    ap.add_argument("--save-dir", required=True, help="3DMedAgent --save-dir (per-image .json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    with open(args.vqa_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_case_idx = {i: r for i, r in enumerate(rows)}

    # ---- load their records -------------------------------------------------
    recs = {}
    save_dir = Path(args.save_dir)
    n_files = 0
    for path in sorted(save_dir.glob("*.json")):
        n_files += 1
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN unreadable {path.name}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            ci = rec.get("case_idx")
            if ci is None:
                continue
            recs[int(ci)] = rec

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, skips = [], []
    for ci, row in by_case_idx.items():
        qid = row["qid"]
        gt = (row.get("correct option") or "").strip().upper()
        rec = recs.get(ci)
        if rec is None:
            skips.append({"qid": qid, "image_id": row["Image ID"], "reason": "no_record"})
            continue
        raw_result = str(rec.get("GPT_raw_result") or "")
        summarized = rec.get("GPT_summarized_result")
        if raw_result.startswith("SKIP:") or raw_result.startswith("ERROR:"):
            skips.append({
                "qid": qid,
                "image_id": row["Image ID"],
                "question_type": row.get("question type", ""),
                "reason": raw_result.split("\n")[0][:200],
            })
            continue
        pred = first_letter(summarized)
        preds.append({
            "qid": qid,
            "image_id": row["Image ID"],
            "question_type": row.get("question type", ""),
            "question_subtype": row.get("question subtype", ""),
            "pred_letter": pred,
            "correct_option": gt,
            # unparsable prediction counts as wrong and is kept (HANDOFF 6)
            "correct": int(pred is not None and pred == gt),
            "raw": str(summarized or "")[:300],
            "n_llm_calls": rec.get("n_llm_calls"),
            "error": None,
        })

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out_dir / "skipped.jsonl", "w", encoding="utf-8") as f:
        for s in skips:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- summary ------------------------------------------------------------
    def acc(items):
        return (sum(p["correct"] for p in items) / len(items)) if items else None

    by_type = defaultdict(list)
    for p in preds:
        by_type[p["question_type"]].append(p)

    summary = {
        "run_label": args.run_label,
        "vqa_csv": args.vqa_csv,
        "save_dir": args.save_dir,
        "n_questions_in_set": len(rows),
        "n_record_files": n_files,
        "n_scored": len(preds),
        "n_skipped": len(skips),
        "skip_reasons": dict(Counter(s["reason"] for s in skips)),
        "n_unparsable_pred": sum(1 for p in preds if p["pred_letter"] is None),
        "accuracy_overall": acc(preds),
        "accuracy_by_question_type": {k: acc(v) for k, v in sorted(by_type.items())},
        "n_by_question_type": {k: len(v) for k, v in sorted(by_type.items())},
    }
    usage_path = os.environ.get("FINAL_TEST_USAGE_LOG")
    if usage_path and os.path.exists(usage_path):
        # NOTE: each question-type run is a separate process and each one
        # OVERWRITES this file, so it only holds the last type's numbers. The
        # per-type totals are printed into the run logs; sum those instead and
        # keep the partial file under a clearly-named key.
        summary["api_usage_last_question_type_only"] = json.load(open(usage_path))

    log_glob = os.environ.get("FINAL_TEST_LOG_GLOB")
    if log_glob:
        import glob as _glob
        import re as _re

        totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        per_type = {}
        for path in sorted(_glob.glob(log_glob)):
            text = open(path, encoding="utf-8", errors="ignore").read()
            hits = _re.findall(r"API usage: (\{.*?\})", text)
            if not hits:
                continue
            try:
                d = json.loads(hits[-1])
            except Exception:  # noqa: BLE001
                continue
            per_type[os.path.basename(path)] = d
            for k in totals:
                totals[k] += int(d.get(k, 0))
        if per_type:
            summary["api_usage_total"] = totals
            summary["api_usage_by_log"] = per_type

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
