"""CT volume + multi-structure label volume in a canonical RAS+ voxel frame.

Array axes after canonicalisation: a[x, y, z] with
  +x -> patient Right, +y -> patient Anterior, +z -> patient Superior.
All physical quantities are in mm, computed with the voxel spacing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import paths as P

TOTALSEG_ROOT = Path(os.environ.get("CE_TOTALSEG_DATASET", P.REPO / "data/totalsegmentator"))   # unused by the paper
CACHE_DIR = P.CACHE / "cases"


@dataclass
class Case:
    case_id: str
    ct: np.ndarray            # int16 HU, (X, Y, Z)
    labels: np.ndarray        # uint8, 0 = background, k -> names[k-1]
    names: list[str]
    spacing: tuple[float, float, float]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.ct.shape

    def name_to_id(self, name: str) -> int:
        return self.names.index(name) + 1

    @property
    def layers(self) -> dict:
        """Extra label sets that may overlap the main one: {layer: (uint8 array, names)}."""
        return self.meta.get("layers", {})

    def has(self, name: str) -> bool:
        return name in self.names or any(name in ns for _, ns in self.layers.values())

    def mask(self, name: str) -> np.ndarray:
        if name in self.names:
            return self.labels == self.name_to_id(name)
        for arr, ns in self.layers.values():
            if name in ns:
                return arr == ns.index(name) + 1
        raise KeyError(name)


def _canon(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj), tuple(float(s) for s in img.header.get_zooms()[:3])


def load_totalseg_case(case_id: str, root: Path = TOTALSEG_ROOT, use_cache: bool = True) -> Case:
    """Load a TotalSegmentator-dataset case; GT masks are merged into one label volume.

    The 117 TotalSegmentator classes are mutually exclusive, so a single uint8 volume
    is lossless. Overlapping label sets (e.g. lesions, liver segments) need extra volumes.
    """
    cache = CACHE_DIR / f"{case_id}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return Case(case_id, d["ct"], d["labels"], list(d["names"]), tuple(d["spacing"]))

    case_dir = root / case_id
    ct, spacing = _canon(case_dir / "ct.nii.gz")
    seg_dir = case_dir / "segmentations"
    files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".nii.gz"))
    labels = np.zeros(ct.shape, np.uint8)
    names = []
    for f in files:
        m, _ = _canon(seg_dir / f)
        m = m > 0
        if not m.any():
            continue
        names.append(f[: -len(".nii.gz")])
        labels[m & (labels == 0)] = len(names)
    case = Case(case_id, ct.astype(np.int16), labels, names, spacing)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ct=case.ct, labels=labels, names=np.array(names), spacing=np.array(spacing))
    return case


ATLAS_ROOT = P.ATLAS_CT


def load_atlas_case(case_id: str, root: Path = ATLAS_ROOT, use_cache: bool = True) -> Case:
    """AbdomenAtlas 1.1 (25 classes). Masks can overlap (hepatic_vessel inside liver), so
    larger structures are painted first and smaller ones overwrite them; the overwritten
    voxels are lost from the larger structure (a few % for liver)."""
    cache = CACHE_DIR / f"{case_id}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return Case(case_id, d["ct"], d["labels"], list(d["names"]), tuple(d["spacing"]))
    case_dir = root / case_id
    ct, spacing = _canon(case_dir / "ct.nii.gz")
    seg_dir = case_dir / "segmentations"
    masks = {}
    for f in sorted(os.listdir(seg_dir)):
        if f.endswith(".nii.gz"):
            m, _ = _canon(seg_dir / f)
            m = m > 0
            if m.any():
                masks[f[: -len(".nii.gz")]] = m
    names = sorted(masks, key=lambda n: -masks[n].sum())
    labels = np.zeros(ct.shape, np.uint8)
    for i, n in enumerate(names, start=1):
        labels[masks[n]] = i
    case = Case(case_id, ct.astype(np.int16), labels, names, spacing)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ct=case.ct, labels=labels, names=np.array(names), spacing=np.array(spacing))
    return case


TS_DIR = P.TOTALSEG_OUT
TS_CLASSES = json.load(open(Path(__file__).with_name("totalseg_classes.json")))
TS_LAYERS = ["liver_vessels", "liver_segments"]


def _compact(arr: np.ndarray, class_map: dict) -> tuple[np.ndarray, list[str]]:
    """Relabel to 1..K over the classes present; returns (uint8 array, names)."""
    present = [int(v) for v in np.unique(arr) if v != 0]
    lut = np.zeros(max(present + [0]) + 1, np.uint8)
    names = []
    for k in present:
        names.append(class_map[str(k)])
        lut[k] = len(names)
    return lut[arr], names


def load_ts_case(case_id: str, use_cache: bool = True) -> Case:
    """Benchmark CT + TotalSegmentator predictions (realistic setting).

    Main label set = 'total' (117 mutually exclusive classes); overlapping label sets in
    case.layers: liver_vessels (liver_vessels, liver_tumor) and liver_segments (Couinaud 1-8).
    The CT itself is shared with the AbdomenAtlas cache (same scan, same grid)."""
    ct_case = load_atlas_case(case_id, use_cache=use_cache)
    cache = CACHE_DIR / f"{case_id}__ts.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        layers = {l: (d[f"L_{l}"], list(d[f"N_{l}"])) for l in TS_LAYERS if f"L_{l}" in d}
        return Case(case_id, ct_case.ct, d["labels"], list(d["names"]), ct_case.spacing, {"layers": layers, "seg": "totalseg"})
    labels, names = _compact(_canon(TS_DIR / case_id / "total.nii.gz")[0], TS_CLASSES["total"])
    if labels.shape != ct_case.ct.shape:
        raise ValueError(f"{case_id}: TotalSegmentator grid {labels.shape} != CT {ct_case.ct.shape}")
    layers = {}
    for l in TS_LAYERS:
        f = TS_DIR / case_id / f"{l}.nii.gz"
        if f.exists():
            layers[l] = _compact(_canon(f)[0], TS_CLASSES[l])
    if use_cache:
        np.savez_compressed(cache, labels=labels, names=np.array(names),
                            **{f"L_{l}": a for l, (a, _) in layers.items()},
                            **{f"N_{l}": np.array(n) for l, (_, n) in layers.items()})
    return Case(case_id, ct_case.ct, labels, names, ct_case.spacing, {"layers": layers, "seg": "totalseg"})


GT_DIR = P.GT_MASKS / "bench"
GT_LAYERS = ["liver_segments", "pancreas_parts", "lesions"]
GT_PARTS = {"liver_segments": [f"liver_segment_{k}" for k in range(1, 9)],
            "pancreas_parts": ["pancreas_head", "pancreas_body", "pancreas_tail"]}
GT_LESIONS = ["liver_lesion", "kidney_lesion", "pancreatic_lesion", "colon_lesion"]


def _split_kidney_lesion(m: np.ndarray, kl: np.ndarray, kr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each connected kidney-lesion component to the kidney it touches most (dilated masks; else nearer centroid)."""
    from scipy import ndimage
    left, right = np.zeros_like(m), np.zeros_like(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return left, right
    dl, dr = ndimage.binary_dilation(kl, iterations=3), ndimage.binary_dilation(kr, iterations=3)
    cl = np.argwhere(kl).mean(0) if kl.any() else None
    cr = np.argwhere(kr).mean(0) if kr.any() else None
    for j, sl in enumerate(ndimage.find_objects(lab), start=1):
        c = lab[sl] == j
        a, b = int((c & dl[sl]).sum()), int((c & dr[sl]).sum())
        if a == b:
            cen = np.argwhere(c).mean(0) + [s.start for s in sl]
            a = -np.linalg.norm(cen - cl) if cl is not None else -np.inf
            b = -np.linalg.norm(cen - cr) if cr is not None else -np.inf
        (left if a > b else right)[sl][c] = True
    return left, right


def load_gt_case(case_id: str, use_cache: bool = True) -> Case:
    """Oracle ablation (2026-11-07): benchmark CT + AbdomenAtlas3.0Mini GT masks as the 'segmentation tool'.

    Main label set = organs / vessels (exclusive; larger painted first, as in load_atlas_case). Overlapping layers:
    liver_segments (Couinaud 1-8, same names as TotalSegmentator's task), pancreas_parts (head / body / tail) and
    lesions (liver_lesion, kidney_lesion_left / right, pancreatic_lesion, colon_lesion; binary GT masks, tumors and
    cysts not distinguished). Same grid as the AbdomenAtlas1.1Mini CT. Never exposes the reports."""
    ct_case = load_atlas_case(case_id, use_cache=use_cache)
    cache = CACHE_DIR / f"{case_id}__gt.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        layers = {l: (d[f"L_{l}"], list(d[f"N_{l}"])) for l in GT_LAYERS if f"L_{l}" in d}
        return Case(case_id, ct_case.ct, d["labels"], list(d["names"]), ct_case.spacing, {"layers": layers, "seg": "gt"})
    seg_dir = GT_DIR / case_id / "segmentations"
    part_names = {n for v in GT_PARTS.values() for n in v}
    masks, extra = {}, {}
    for f in sorted(os.listdir(seg_dir)):
        if not f.endswith(".nii.gz"):
            continue
        n = f[: -len(".nii.gz")]
        m, _ = _canon(seg_dir / f)
        m = m > 0.5
        if m.shape != ct_case.ct.shape:
            raise ValueError(f"{case_id}: GT mask {n} grid {m.shape} != CT {ct_case.ct.shape}")
        if n in part_names or n in GT_LESIONS:
            extra[n] = m
        elif m.any():
            masks[n] = m
    names = sorted(masks, key=lambda n: -masks[n].sum())
    labels = np.zeros(ct_case.ct.shape, np.uint8)
    for i, n in enumerate(names, start=1):
        labels[masks[n]] = i
    layers = {}
    for layer, parts in GT_PARTS.items():
        arr, ns = np.zeros(labels.shape, np.uint8), []
        for n in parts:
            if n in extra and extra[n].any():
                ns.append(n)
                arr[extra[n]] = len(ns)
        if ns:
            layers[layer] = (arr, ns)
    les = {}
    if "kidney_lesion" in extra and extra["kidney_lesion"].any():
        les["kidney_lesion_left"], les["kidney_lesion_right"] = _split_kidney_lesion(
            extra["kidney_lesion"], masks.get("kidney_left", np.zeros_like(labels, bool)),
            masks.get("kidney_right", np.zeros_like(labels, bool)))
    for n in ("liver_lesion", "pancreatic_lesion", "colon_lesion"):
        if n in extra:
            les[n] = extra[n]
    arr, ns = np.zeros(labels.shape, np.uint8), []
    for n, m in les.items():
        if m.any():
            ns.append(n)
            arr[m] = len(ns)
    if ns:
        layers["lesions"] = (arr, ns)
    if use_cache:
        np.savez_compressed(cache, labels=labels, names=np.array(names),
                            **{f"L_{l}": a for l, (a, _) in layers.items()},
                            **{f"N_{l}": np.array(n) for l, (_, n) in layers.items()})
    return Case(case_id, ct_case.ct, labels, names, ct_case.spacing, {"layers": layers, "seg": "gt"})
