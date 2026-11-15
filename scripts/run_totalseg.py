"""Run TotalSegmentator on DeepTumorVQA benchmark CTs (realistic, predicted segmentation).

Run in a separate environment with TotalSegmentator installed (see README):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_totalseg.py --ids-file splits/cases_all.json --shard 0/2
Resumable: a (case, task) whose output file exists is skipped. Progress: one timestamped, flushed line
per (case, task) in $CE_TOTALSEG_OUT/run_<shard>.log.

Tasks
  total          117 classes (incl. kidney_cyst_left/right)          -> <cid>/total.nii.gz (multi-label)
  liver_vessels  liver_vessels, liver_tumor                           -> <cid>/liver_vessels.nii.gz
  liver_segments Couinaud segments 1-8                                -> <cid>/liver_segments.nii.gz
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctgraph import paths as P  # noqa: E402

ATLAS = P.ATLAS_CT
OUT = P.TOTALSEG_OUT
QA = P.TEST_QA
TASKS = ["total", "liver_vessels", "liver_segments"]


def case_list(which: str, ids_file: str | None = None) -> list[str]:
    if ids_file:        # e.g. splits/cases_all.json {"ids": [...]}
        return list(json.load(open(ids_file))["ids"])
    return sorted({json.loads(l)["image_id"] for l in open(QA)})


def log(f, msg):
    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all", choices=["all"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--ids-file", default=None, help='json with an "ids" list of BDMAP ids (overrides --cases)')
    ap.add_argument("--log-name", default=None, help="log file name in $CE_TOTALSEG_OUT (default run_<shard>.log)")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    from totalsegmentator.python_api import totalsegmentator

    OUT.mkdir(parents=True, exist_ok=True)
    ids = case_list(a.cases, a.ids_file)[i::n]
    with open(OUT / (a.log_name or f"run_{i}.log"), "a") as f:
        log(f, f"START shard {a.shard} cases={a.cases} n={len(ids)} tasks={a.tasks}")
        for k, cid in enumerate(ids):
            for task in a.tasks:
                out = OUT / cid / f"{task}.nii.gz"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t = time.time()
                try:
                    totalsegmentator(str(ATLAS / cid / "ct.nii.gz"), str(out), ml=True, task=task, device="gpu",
                                     nr_thr_resamp=2, nr_thr_saving=2, quiet=True)
                    log(f, f"OK   {k + 1}/{len(ids)} {cid} {task} {time.time() - t:.0f}s")
                except Exception as e:
                    log(f, f"FAIL {k + 1}/{len(ids)} {cid} {task} {type(e).__name__}: {str(e)[:200]}")
                    log(f, traceback.format_exc().replace("\n", " | ")[-800:])
        log(f, "DONE")


if __name__ == "__main__":
    main()
