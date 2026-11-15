"""Anatomical semantic graph: nodes = segmented structures, edges = spatial relations.

Coordinates are in mm in the canonical RAS+ voxel frame (x->Right, y->Anterior, z->Superior),
i.e. voxel_index * spacing. Only relative geometry is used, so the origin does not matter.

Edge types
  adjacent  : min surface distance <= adj_thresh_mm (within the main, mutually exclusive label set)
  contains  : between different label sets (case.layers: lesions, liver segments, ...): >= CONTAIN_FRAC
              of b's voxels lie inside a  (stored with a = container, b = contained)
  overlaps  : between different label sets, partial overlap (frac_a / frac_b = share of each inside the other)
Directions are rendered to text in code (see describe_relation) so the VLM never has to
convert coordinate conventions itself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from .volume import Case

AXES = (("right", "left"), ("anterior", "posterior"), ("superior", "inferior"))


@dataclass
class Node:
    name: str
    n_vox: int
    volume_ml: float
    bbox_vox: list[int]        # x0, x1, y0, y1, z0, z1 (inclusive)
    centroid_mm: list[float]
    extent_mm: list[float]
    max_diameter_mm: float
    hu_mean: float
    hu_std: float
    n_components: int


@dataclass
class Edge:
    a: str
    b: str
    min_dist_mm: float          # centre-to-centre distance of the closest voxel pair (0 if overlapping)
    contact_area_mm2: float     # approx. area of face-adjacent voxels (0 if not touching)
    closest_a_mm: list[float]
    closest_b_mm: list[float]
    type: str = "adjacent"
    frac_a: float | None = None  # share of a's voxels inside b (contains / overlaps)
    frac_b: float | None = None  # share of b's voxels inside a


CONTAIN_FRAC = 0.9

# lesion labels -> organ they belong to; each connected component becomes its own node "<label>#k"
LESION_ORGAN_TS = {"liver_tumor": "liver", "kidney_cyst_left": "kidney_left", "kidney_cyst_right": "kidney_right"}
# GT (AbdomenAtlas3.0Mini, oracle ablation 2026-11-07): binary lesion masks, tumors and cysts not distinguished;
# kidney_lesion is split by side in volume.load_gt_case
LESION_ORGAN_GT = {"liver_lesion": "liver", "kidney_lesion_left": "kidney_left", "kidney_lesion_right": "kidney_right",
                   "pancreatic_lesion": "pancreas", "colon_lesion": "colon"}
LESION_ORGAN = {**LESION_ORGAN_TS, **LESION_ORGAN_GT}    # lookups only; a case has the labels of one source


def lesion_organ(seg: str | None) -> dict:
    """Lesion label -> organ map of one segmentation source ('gt' or anything else = TotalSegmentator)."""
    return LESION_ORGAN_GT if seg == "gt" else LESION_ORGAN_TS
EMBED_CONTACT, EMBED_ENCLOSED = 0.2, 0.3


def _filled(m: np.ndarray) -> np.ndarray:
    """Organ mask with enclosed holes filled (per axial slice, after closing small boundary gaps)."""
    m = ndimage.binary_closing(m, structure=np.ones((3, 3, 1)), iterations=2)
    return np.stack([ndimage.binary_fill_holes(m[:, :, z]) for z in range(m.shape[2])], -1)


def _lesion_instances(case: Case, ct, sp, min_vol_ml: float = 0.02) -> tuple[dict, list]:
    """One node per connected lesion component + its topological relation to the parent organ:
    contains (>= CONTAIN_FRAC of the lesion lies inside the hole-filled organ), embedded (partially
    enclosed / surface contact), else adjacent. Also overlaps with liver segments (which segment)."""
    nodes, edges = {}, []
    seg = case.layers.get("liver_segments")
    for les, organ in lesion_organ(case.meta.get("seg")).items():
        if not case.has(les):
            continue
        lab, _ = ndimage.label(case.mask(les))
        O = case.mask(organ) if case.has(organ) else None
        k = 0
        for j, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab[sl] == j
            if m.sum() * sp.prod() / 1000 < min_vol_ml:
                continue
            k += 1
            name = f"{les}#{k}"
            nodes[name] = _node(name, m, sl, ct, sp, 0.01)
            if O is None:
                continue
            pad = tuple(slice(max(s.start - 6, 0), s.stop + 6) for s in sl)
            L = np.zeros(O[pad].shape, bool)
            L[tuple(slice(s.start - p.start, s.stop - p.start) for s, p in zip(sl, pad))] = m
            enclosed = float((L & _filled(O[pad])).sum() / L.sum())
            ring = ndimage.binary_dilation(L) & ~L
            contact = float((ring & O[pad]).sum() / max(ring.sum(), 1))
            typ = "contains" if enclosed >= CONTAIN_FRAC else (
                "embedded" if contact >= EMBED_CONTACT or enclosed >= EMBED_ENCLOSED else "adjacent")
            edges.append(Edge(organ, name, 0.0, 0.0, [], [], typ, round(contact, 3), round(enclosed, 3)))
            if seg is not None and organ == "liver":
                arr, names = seg
                v = arr[sl][m]
                cnt = np.bincount(v[v > 0], minlength=len(names) + 1)
                for si in np.nonzero(cnt)[0]:
                    f = cnt[si] / m.sum()
                    if f >= 0.1:
                        edges.append(Edge(names[si - 1], name, 0.0, 0.0, [], [], "contains" if f >= CONTAIN_FRAC else "overlaps",
                                          None, round(float(f), 3)))
    return nodes, edges


def _max_diameter(mask: np.ndarray, spacing) -> float:
    surf = mask & ~ndimage.binary_erosion(mask)
    pts = np.argwhere(surf) * np.asarray(spacing)
    if len(pts) < 5:
        return float(np.ptp(pts, 0).max()) if len(pts) else 0.0
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    try:
        pts = pts[ConvexHull(pts).vertices]
    except Exception:
        pass
    return float(pdist(pts).max())


def _node(name, m, sl, ct, sp, min_comp_frac) -> Node:
    """Node statistics from a boolean mask m cropped to slices sl."""
    n = int(m.sum())
    idx = np.argwhere(m) + np.array([s.start for s in sl])
    hu = ct[sl][m]
    comp, _ = ndimage.label(m)
    sizes = np.bincount(comp.ravel())[1:]
    return Node(
        name=name,
        n_vox=n,
        volume_ml=float(n * sp.prod() / 1000),
        bbox_vox=[int(v) for s in sl for v in (s.start, s.stop - 1)],
        centroid_mm=[float(v) for v in idx.mean(0) * sp],
        extent_mm=[float((s.stop - s.start) * d) for s, d in zip(sl, sp)],
        max_diameter_mm=_max_diameter(m, sp),
        hu_mean=float(hu.mean()),
        hu_std=float(hu.std()),
        n_components=int((sizes >= max(min_comp_frac * n, 1)).sum()),
    )


def _layer_edges(sets: list[tuple[np.ndarray, list[str]]], sp) -> list[Edge]:
    """contains / overlaps edges between every pair of label sets (joint histogram of label pairs)."""
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            (A, na), (B, nb) = sets[i], sets[j]
            sel = (A > 0) & (B > 0)
            if not sel.any():
                continue
            KB = len(nb) + 1
            joint = np.bincount(A[sel].astype(np.int64) * KB + B[sel], minlength=(len(na) + 1) * KB).reshape(-1, KB)
            ca = np.bincount(A.ravel(), minlength=len(na) + 1)
            cb = np.bincount(B.ravel(), minlength=KB)
            for ia, ib in zip(*np.nonzero(joint)):
                inter = joint[ia, ib]
                fa, fb = inter / ca[ia], inter / cb[ib]
                a, b = na[ia - 1], nb[ib - 1]
                if fb >= CONTAIN_FRAC:
                    e = Edge(a, b, 0.0, 0.0, [], [], "contains", round(float(fa), 3), round(float(fb), 3))
                elif fa >= CONTAIN_FRAC:
                    e = Edge(b, a, 0.0, 0.0, [], [], "contains", round(float(fb), 3), round(float(fa), 3))
                elif max(fa, fb) >= 0.1 or inter * sp.prod() >= 100:  # ignore boundary noise
                    e = Edge(a, b, 0.0, 0.0, [], [], "overlaps", round(float(fa), 3), round(float(fb), 3))
                else:
                    continue
                out.append(e)
    return out


def build_graph(case: Case, adj_thresh_mm: float = 10.0, min_comp_frac: float = 0.01) -> dict:
    sp = np.asarray(case.spacing)
    lab, ct = case.labels, case.ct
    objs = ndimage.find_objects(lab)
    nodes: dict[str, Node] = {}
    for k, sl in enumerate(objs, start=1):
        if sl is not None:
            name = case.names[k - 1]
            nodes[name] = _node(name, lab[sl] == k, sl, ct, sp, min_comp_frac)

    pad = np.ceil(adj_thresh_mm / sp).astype(int) + 1
    face = sp.max() * 1.01
    edges: dict[tuple[str, str], Edge] = {}
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        a = case.names[k - 1]
        lo = [max(s.start - p, 0) for s, p in zip(sl, pad)]
        hi = [min(s.stop + p, d) for s, p, d in zip(sl, pad, lab.shape)]
        crop = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dist, inds = ndimage.distance_transform_edt(crop != k, sampling=sp, return_indices=True)
        others = [int(i) for i in np.unique(crop) if i not in (0, k)]
        if not others:
            continue
        mins = ndimage.minimum(dist, crop, others)
        poss = ndimage.minimum_position(dist, crop, others)
        for j, d, pos in zip(others, mins, poss):
            if d > adj_thresh_mm:
                continue
            b = case.names[j - 1]
            key = tuple(sorted((a, b)))
            if key in edges:
                continue
            pa = np.array([inds[ax][pos] for ax in range(3)]) + lo
            pb = np.array(pos) + lo
            contact = float(((crop == j) & (dist <= face)).sum() * sp.prod() ** (2 / 3))
            e = Edge(a, b, float(d), contact, list(map(float, pa * sp)), list(map(float, pb * sp)))
            if key[0] != a:  # store with a == key[0]
                e = Edge(b, a, e.min_dist_mm, contact, e.closest_b_mm, e.closest_a_mm)
            edges[key] = e
    # overlapping label sets (lesions, liver segments): nodes + contains / overlaps edges
    sets = [(lab, case.names)]
    for layer, (arr, names) in case.layers.items():
        for k, sl in enumerate(ndimage.find_objects(arr), start=1):
            if sl is not None and names[k - 1] not in nodes:
                nodes[names[k - 1]] = _node(names[k - 1], arr[sl] == k, sl, ct, sp, min_comp_frac)
        sets.append((arr, names))
    layer_edges = _layer_edges(sets, sp) if len(sets) > 1 else []
    inst_nodes, inst_edges = _lesion_instances(case, ct, sp)
    nodes.update(inst_nodes)
    layer_edges += inst_edges
    return {
        "case_id": case.case_id,
        "seg": case.meta.get("seg", "gt"),
        "spacing": list(map(float, sp)),
        "shape": list(lab.shape),
        "adj_thresh_mm": adj_thresh_mm,
        "nodes": {k: asdict(v) for k, v in nodes.items()},
        "edges": [asdict(e) for e in edges.values()] + [asdict(e) for e in layer_edges],
    }


def save_graph(g: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(g, f)


def load_graph(path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- text rendering

def _axis_phrases(delta_mm, min_mm: float) -> list[str]:
    out = []
    for ax, (pos, neg) in enumerate(AXES):
        d = delta_mm[ax]
        if abs(d) >= min_mm:
            out.append(f"{pos if d > 0 else neg} ({abs(d):.0f} mm)")
    return out


def _interval_relation(a0, a1, b0, b1, pos: str, neg: str) -> str | None:
    if a0 > b1:
        return f"entirely {pos}"
    if a1 < b0:
        return f"entirely {neg}"
    return None


def describe_relation(g: dict, a: str, b: str, min_mm: float = 5.0) -> str:
    """Patient-frame description of where structure a lies relative to structure b."""
    na, nb = g["nodes"][a], g["nodes"][b]
    sp = g["spacing"]
    delta = np.array(na["centroid_mm"]) - np.array(nb["centroid_mm"])
    parts = _axis_phrases(delta, min_mm)
    s = f"{a} is " + (", ".join(parts) if parts else "roughly co-centred") + f" relative to {b} (centroid offsets, patient frame)"
    rels = []
    for ax, (pos, neg) in enumerate(AXES):
        a0, a1 = (v * sp[ax] for v in na["bbox_vox"][2 * ax : 2 * ax + 2])
        b0, b1 = (v * sp[ax] for v in nb["bbox_vox"][2 * ax : 2 * ax + 2])
        r = _interval_relation(a0, a1, b0, b1, pos, neg)
        if r:
            rels.append(f"{r} of it")
    if rels:
        s += "; extent-wise " + " and ".join(rels)
    e = find_edge(g, a, b)
    if e is not None and e.get("type") == "contains":
        inner, outer = (e["b"], e["a"])
        return f"{inner} lies inside {outer} ({e['frac_b'] * 100:.0f}% of {inner} is within {outer})."
    if e is not None and e.get("type") == "embedded":
        return (f"{e['b']} is embedded in / bulges from {e['a']} ({e['frac_a'] * 100:.0f}% of its surface touches {e['a']}, "
                f"{e['frac_b'] * 100:.0f}% of it lies within {e['a']}'s outline).")
    if e is not None and e.get("type") == "overlaps":
        return (f"{e['a']} and {e['b']} partially overlap ({e['frac_a'] * 100:.0f}% of {e['a']} lies in {e['b']}; "
                f"{e['frac_b'] * 100:.0f}% of {e['b']} lies in {e['a']}).")
    if e is not None:
        touch = "touching" if e["contact_area_mm2"] > 0 else f"gap {e['min_dist_mm']:.0f} mm"
        s += f"; closest surfaces: {touch}"
    return s + "."


def find_edge(g: dict, a: str, b: str) -> dict | None:
    for e in g["edges"]:
        if {e["a"], e["b"]} == {a, b}:
            return e
    return None


def neighbors(g: dict, name: str) -> list[dict]:
    out = []
    for e in g["edges"]:
        if name in (e["a"], e["b"]):
            out.append(e)
    return sorted(out, key=lambda e: e["min_dist_mm"])


def graph_to_text(g: dict, relations: bool = True) -> str:
    """Compact text serialisation of the whole graph for an LLM prompt."""
    sp = g["spacing"]
    lines = [f"Volume: {g['shape']} voxels, spacing {[round(s, 2) for s in sp]} mm. "
             "Frame: x->patient right, y->anterior, z->superior.",
             "NODES (name: volume, max diameter, mean HU +- std, z-extent superior-inferior, connected components):"]
    for n, d in sorted(g["nodes"].items()):
        z0, z1 = d["bbox_vox"][4] * sp[2], d["bbox_vox"][5] * sp[2]
        lines.append(f"- {n}: {d['volume_ml']:.1f} mL, {d['max_diameter_mm']:.0f} mm, HU {d['hu_mean']:.0f}+-{d['hu_std']:.0f}, "
                     f"z {z0:.0f}-{z1:.0f} mm, {d['n_components']} comp")
    lines.append(f"EDGES (structure pairs within {g['adj_thresh_mm']:.0f} mm):")
    for e in sorted(g["edges"], key=lambda e: (e["a"], e["b"])):
        if relations:
            lines.append("- " + describe_relation(g, e["a"], e["b"]))
        else:
            touch = "touching" if e["contact_area_mm2"] > 0 else f"gap {e['min_dist_mm']:.0f} mm"
            lines.append(f"- {e['a']} -- {e['b']}: {touch}")
    return "\n".join(lines)
