"""DeepTumorVQA oracle tool cache -> our canonical voxel frame (evaluation / oracle ablations only).

The cache bboxes are on the AbdomenAtlas grid with axes (z, y, x) = raw array axes (0, 1, 2) of the
AbdomenAtlas1.1Mini NIfTI, but with a case-specific translation (AbdomenAtlas3 crops differently).
The translation is estimated per case as the median offset of organ bboxes present in both.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

from . import paths as P
from .volume import ATLAS_ROOT

ORACLE = P.TOOL_CACHE / "benchmark_oracle_tool_cache.json"
LESION_KEYS = ["liver_tumor", "kidney_tumor", "kidney_cyst", "pancreas_tumor", "colon_tumor",
               "hepatic_vessel_tumor", "lung_tumor"]
REF_ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach", "aorta"]


@lru_cache(1)
def _cache() -> dict:
    return json.load(open(ORACLE))["cache"]


def oracle(case_id: str) -> dict:
    return _cache()[case_id]


def _raw_bbox(case_id: str, organ: str):
    f = ATLAS_ROOT / case_id / "segmentations" / f"{organ}.nii.gz"
    m = np.asanyarray(nib.load(str(f)).dataobj) > 0
    if not m.any():
        return None
    idx = np.argwhere(m)
    return idx.min(0), idx.max(0)


def raw_offset(case_id: str) -> np.ndarray:
    seg = oracle(case_id)["segment"]
    offs = []
    for o in REF_ORGANS:
        b = seg.get(o)
        if not b or not b.get("mask_found"):
            continue
        rb = _raw_bbox(case_id, o)
        if rb is None:
            continue
        bb = b["bounding_box"]
        omin = np.array([bb["z_min"], bb["y_min"], bb["x_min"]])
        omax = np.array([bb["z_max"], bb["y_max"], bb["x_max"]])
        offs += [omin - rb[0], omax - rb[1]]
    return np.median(np.array(offs), 0).round().astype(int)


def lesion_boxes_canonical(case_id: str) -> dict[str, dict]:
    """{lesion_key: {"box": [x0,x1,y0,y1,z0,z1] canonical voxels, "voxel_count": n}} (union bbox of all
    instances of that lesion class)."""
    img = nib.load(str(ATLAS_ROOT / case_id / "ct.nii.gz"))
    shape = np.array(img.shape[:3])
    ornt = nib.orientations.io_orientation(img.affine)
    off = raw_offset(case_id)
    out = {}
    for k in LESION_KEYS:
        b = oracle(case_id)["segment"].get(k)
        if not b or not b.get("mask_found"):
            continue
        bb = b["bounding_box"]
        lo = np.array([bb["z_min"], bb["y_min"], bb["x_min"]]) - off
        hi = np.array([bb["z_max"], bb["y_max"], bb["x_max"]]) - off
        can = np.zeros((3, 2), int)
        for i in range(3):
            ax, flip = int(ornt[i, 0]), ornt[i, 1]
            a, b_ = (lo[i], hi[i]) if flip > 0 else (shape[i] - 1 - hi[i], shape[i] - 1 - lo[i])
            can[ax] = [a, b_]
        out[k] = {"box": can.ravel().tolist(), "voxel_count": b["voxel_count"]}
    return out
