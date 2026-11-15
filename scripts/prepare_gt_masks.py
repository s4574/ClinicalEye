"""Download and extract the AbdomenAtlas3.0Mini masks of the benchmark cases into $CE_GT_MASKS/bench/<BDMAP>/segmentations/.

Used ONLY by the structure probes (scripts/structure_probe.py, scripts/zprobe.py: building / scoring items) and GT-based
analyses; the agent never reads them. ~40 tarballs of ~300 MB (resumable: finished downloads / tarballs are skipped).
Usage: python scripts/prepare_gt_masks.py [--workers 6]"""
import argparse
import json
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctgraph import paths as P  # noqa: E402

REPO_ID = "AbdomenAtlas/AbdomenAtlas3.0Mini"
ROOT = P.GT_MASKS
IDS = {json.loads(l)["image_id"] for l in open(P.TEST_QA)}


def one(tar: str) -> int:
    done = ROOT / "bench" / f".done_{Path(tar).name}"
    if done.exists():
        return 0
    n = 0
    with tarfile.open(tar) as tf:
        for m in tf:
            parts = Path(m.name).parts
            cid = next((p for p in parts if p.startswith("BDMAP_")), None)
            if cid in IDS and m.isfile():
                m.name = str(Path(cid) / Path(*parts[parts.index(cid) + 1:]))
                tf.extract(m, ROOT / "bench")
                n += 1
    done.touch()
    print(f"{Path(tar).name}: {n} files", flush=True)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO_ID, path_in_repo="mask_only", repo_type="dataset")
                   if f.path.endswith(".tar.gz"))
    tars = [hf_hub_download(REPO_ID, p, repo_type="dataset", local_dir=ROOT) for p in files]
    (ROOT / "bench").mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p:
        print(f"extracted {sum(p.map(one, tars))} files")
