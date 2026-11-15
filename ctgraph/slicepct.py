"""Slice-position percentages as used by DeepTumorVQA ("At which CT image slice (%) ...").

Verified on the benchmark (2026-10-21): answer % = index along the 3rd raw array axis of the ORIGINAL
benchmark CT / its size * 100 (no flip); 101/115 "largest lesion slice" answers fall inside the oracle
lesion bbox under this definition (the rest are oracle-mask vs report mismatches).
Our volumes are AbdomenAtlas1.1Mini crops of the same scans (canonical RAS after loading), so
  canonical z -> raw Atlas index along raw axis 2 -> + crop offset (oracle.raw_offset) -> / original size.
Only defined when raw axis 2 is the superior-inferior axis (~96% of cases).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib

from . import paths as P
from .oracle import raw_offset
from .volume import ATLAS_ROOT

QA = P.TEST_QA


@lru_cache(1)
def _bench_shapes() -> dict:
    out = {}
    for l in open(QA):
        r = json.loads(l)
        out[r["image_id"]] = [int(x) for x in r["shape"].strip("()").split(",")]
    return out


TS_OFFSETS = P.CACHE / "ts_raw_offset.json"


@lru_cache(1)
def _ts_offsets() -> dict:
    """Crop offset (raw z, y, x) of our AbdomenAtlas copy within the original benchmark CT, estimated WITHOUT ground
    truth: median difference between organ bounding boxes in the benchmark's own TotalSegmentator cache (original
    coordinates) and in our TotalSegmentator outputs (Atlas crop). Agrees with the oracle-based estimate within +-1
    slice in 99% of the 279 cases where both exist (max 2 slices); 4 cases have no estimate."""
    return json.load(open(TS_OFFSETS)) if TS_OFFSETS.exists() else {}


@lru_cache(2048)
def z_to_pct_params(case_id: str, source: str = "oracle"):
    """(sign, offset, size): pct(z) = (offset + sign * z) / size * 100, or None if undefined.
    source: "oracle" (organ boxes of the benchmark's oracle cache; used by the NV3 family) or "ts" (no ground truth)."""
    img = nib.load(str(ATLAS_ROOT / case_id / "ct.nii.gz"))
    ornt = nib.orientations.io_orientation(img.affine)
    ax, flip = int(ornt[2, 0]), ornt[2, 1]          # raw axis 2 -> canonical axis ax
    if ax != 2:
        return None
    n_raw = img.shape[2]
    if source == "ts":
        o = _ts_offsets().get(case_id)
        if o is None:
            return None
        off = int(o[2])
    else:
        off = int(raw_offset(case_id)[2])
    size = _bench_shapes()[case_id][2]
    # canonical z == raw z if flip > 0 else n_raw - 1 - raw z
    return (1, off, size) if flip > 0 else (-1, off + n_raw - 1, size)


def pct(case_id: str, z: float, source: str = "oracle") -> float | None:
    p = z_to_pct_params(case_id, source)
    if p is None:
        return None
    s, off, size = p
    return (off + s * z) / size * 100


def describe(case_id: str, z_max: int, source: str = "oracle") -> str:
    p = z_to_pct_params(case_id, source)
    if p is None:
        return ""
    a, b = pct(case_id, 0, source), pct(case_id, z_max, source)
    return (f"Slice position as a percentage of the ORIGINAL scan (the unit used by questions about 'CT image slice (%)'): "
            f"z=0 -> {a:.1f}%, z={z_max} -> {b:.1f}%, linear in between ({(b - a) / max(z_max, 1):+.3f}% per slice).")
