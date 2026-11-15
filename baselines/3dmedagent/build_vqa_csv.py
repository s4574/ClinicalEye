#!/usr/bin/env python3
"""Build 3DMedAgent's VQA CSV from our lite903 / lite1k question sets.

Column names mirror the repo's own sample table (DeepChestVQA_v1.csv) and the
keys the code reads (`Image ID`, `dataset`, `multiple-choice question`,
`question type`, `question subtype`, `correct option`, `organ`, `lesion`,
`answer`, plus case_context fields shape/spacing/sex/age/contrast/scanner).
`qid` is kept as an extra column so results join back to our tables.

`case_idx` is NOT written: their select_target_rows() adds it as the row index.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ctgraph import paths as P  # noqa: E402

COLUMNS = [
    "qid", "Question ID", "Image ID", "spacing", "shape", "sex", "age", "scanner",
    "contrast", "question", "answer", "multiple-choice question", "correct option",
    "organ", "lesion", "question type", "question subtype", "split", "dataset",
    "volume_path",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="lite903", choices=["lite903", "lite1k"])
    ap.add_argument("--dataset-name", default="DeepTumorVQA")
    ap.add_argument("--out", required=True, help="e.g. <3DMedAgent>/VQA/DeepTumorVQA_lite903.csv")
    ap.add_argument("--mask-metadata", action="store_true", help="blank the organ / lesion columns (variant without metadata)")
    args = ap.parse_args()

    qids = json.load(open(P.SPLITS / f"{args.set}.json"))["qids"]
    order = {q: i for i, q in enumerate(qids)}
    want = set(qids)

    rows = []
    with open(P.TEST_QA) as f:
        for line in f:
            d = json.loads(line)
            if d["qid"] in want:
                rows.append(d)
    assert len(rows) == len(want), f"matched {len(rows)} of {len(want)}"
    rows.sort(key=lambda d: order[d["qid"]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def clean(v):
        # pandas writes float('nan') for missing; keep empty string instead
        if v is None:
            return ""
        s = str(v)
        return "" if s.lower() in ("nan", "none") else s

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, d in enumerate(rows):
            iid = d["image_id"]
            w.writerow({
                "qid": d["qid"],
                "Question ID": i + 1,
                "Image ID": iid,
                "spacing": clean(d.get("spacing")),
                "shape": clean(d.get("shape")),
                "sex": clean(d.get("sex")),
                "age": clean(d.get("age")),
                "scanner": clean(d.get("scanner")),
                "contrast": clean(d.get("contrast")),
                "question": clean(d.get("question")),
                "answer": clean(d.get("answer")),
                "multiple-choice question": clean(d.get("mc_question")),
                "correct option": clean(d.get("correct_option")),
                "organ": "" if args.mask_metadata else clean(d.get("organ")),
                "lesion": "" if args.mask_metadata else clean(d.get("lesion")),
                "question type": clean(d.get("question_type")),
                "question subtype": clean(d.get("question_subtype")),
                "split": "test",
                "dataset": args.dataset_name,
                "volume_path": f"data/{args.dataset_name}/img/{iid}.nii.gz",
            })
    print(f"wrote {out}  rows={len(rows)}  cases={len({d['image_id'] for d in rows})}")


if __name__ == "__main__":
    main()
