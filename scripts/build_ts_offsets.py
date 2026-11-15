"""GT-free crop offset (raw z, y, x) of our AbdomenAtlas copy within the original benchmark CT, for ctgraph/slicepct.py
(source="ts"): median difference between organ bounding boxes in the benchmark's own TotalSegmentator cache
(tool_cache/benchmark_totalsegmentator_cache.json, original coordinates, axes = raw array axes) and in OUR
TotalSegmentator output ($CE_TOTALSEG_OUT/<case>/total.nii.gz, raw grid of the Atlas crop). Same estimator as
ctgraph/oracle.raw_offset, with both sides from TotalSegmentator (no ground truth).
--check: recompute the existing entries and report agreement; otherwise only cases missing from the file are added.
Usage: python3 scripts/build_ts_offsets.py --ids-file splits/cases_all.json [--check]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from ctgraph import paths as P  # noqa: E402
from ctgraph.oracle import REF_ORGANS  # noqa: E402

OUT = P.CACHE / "ts_raw_offset.json"
BENCH = json.load(open(P.TOOL_CACHE / "benchmark_totalsegmentator_cache.json"))["cache"]
CLS = {v: int(k) for k, v in json.load(open(R / "ctgraph/totalseg_classes.json"))["total"].items()}


def offset(cid):
    f = P.TOTALSEG_OUT / cid / "total.nii.gz"
    if cid not in BENCH or not f.exists():
        return cid, None
    lab = np.asanyarray(nib.load(str(f)).dataobj)
    offs = []
    for o in REF_ORGANS:
        b = BENCH[cid]["segment"].get(o)
        if not b or not b.get("mask_found") or "bounding_box" not in b or o not in CLS:
            continue
        idx = np.argwhere(lab == CLS[o])
        if not len(idx):
            continue
        bb = b["bounding_box"]
        offs += [np.array([bb["z_min"], bb["y_min"], bb["x_min"]]) - idx.min(0),
                 np.array([bb["z_max"], bb["y_max"], bb["x_max"]]) - idx.max(0)]
    return cid, (np.median(np.array(offs), 0).round().astype(int).tolist() if offs else None)


if __name__ == "__main__":
    cur = json.load(open(OUT)) if OUT.exists() else {}
    if "--check" in sys.argv:
        ids = sorted(cur)[:40]
    else:
        ids = [c for c in json.load(open(sys.argv[sys.argv.index("--ids-file") + 1]))["ids"] if c not in cur]
    with Pool(4) as p:
        res = dict(p.map(offset, ids))
    if "--check" in sys.argv:
        d = [abs(np.array(res[c]) - np.array(cur[c])).max() for c in ids if res[c] is not None]
        print(f"check on {len(ids)} existing cases: recomputed {len(d)}; identical {sum(x == 0 for x in d)}, "
              f"within 1 {sum(x <= 1 for x in d)}, max diff {max(d)}")
    else:
        new = {c: o for c, o in res.items() if o is not None}
        print(f"{len(ids)} cases: {len(new)} offsets, {len(ids) - len(new)} without estimate")
        cur.update(new)
        json.dump(cur, open(OUT, "w"))
