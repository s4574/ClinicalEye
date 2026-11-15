"""Lesion-candidate proposals inside an organ (the CT analogue of HEE's in-region entity detection).

Training-free: voxels of the (eroded) organ mask whose smoothed HU deviates from the organ median
by more than max(min_delta, k * robust std) form connected components; each component is a
candidate entity that the VLM must visually confirm or reject. Measurements (volume, diameter,
HU, slice) are computed here so the VLM never has to measure pixels itself.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .graph import _max_diameter
from .volume import Case


def _pancreas_part(case: Case, c_vox) -> str:
    """head / body / tail by position along the pancreas principal axis; head = patient-right end."""
    idx = np.argwhere(case.mask("pancreas")).astype(np.float32)
    sp = np.asarray(case.spacing, np.float32)
    pts = idx * sp
    mu = pts.mean(0)
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
    v = np.linalg.svd(sub - mu, full_matrices=False)[2][0]
    if v[0] < 0:  # orient towards patient right (+x) = head
        v = -v
    proj = (pts - mu) @ v
    p = (np.asarray(c_vox) * sp - mu) @ v
    q1, q2 = np.quantile(proj, [1 / 3, 2 / 3])
    return "head" if p >= q2 else ("body" if p >= q1 else "tail")


def propose(case: Case, organs: list[str], max_n: int = 8, min_vol_ml: float = 0.1,
            k: float = 2.5, min_delta: float = 20.0, min_compact: float = 0.03) -> list[dict]:
    sp = np.asarray(case.spacing)
    vox_ml = sp.prod() / 1000
    out = []
    for organ in organs:
        if organ not in case.names:
            continue
        m = case.mask(organ)
        sl = ndimage.find_objects(m.astype(np.uint8))[0]
        m = m[sl]
        core = ndimage.binary_erosion(m, iterations=1)
        if core.sum() < 100:
            continue
        hu = ndimage.gaussian_filter(case.ct[sl].astype(np.float32), 1.0)
        v = hu[core]
        med = float(np.median(v))
        rstd = float(np.median(np.abs(v - med)) * 1.4826)
        thr = max(min_delta, k * rstd)
        off = np.array([s.start for s in sl])
        for kind, sel in (("hypo", core & (hu < med - thr)), ("hyper", core & (hu > med + thr))):
            sel = ndimage.binary_opening(sel)  # cut thin vessel-like / rim structures
            lab, n = ndimage.label(sel)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            for j in np.nonzero(sizes * vox_ml >= min_vol_ml)[0] + 1:
                cm = lab == j
                bb = ndimage.find_objects(cm.astype(np.uint8))[0]
                cmb = cm[bb]
                idx = np.argwhere(cmb) + np.array([s.start for s in bb])
                zc = np.bincount(idx[:, 2])
                z_peak = int(zc.argmax()) + off[2]
                hu_raw = case.ct[sl][cm]
                c_vox = idx.mean(0) + off
                diam = _max_diameter(cmb, sp)
                vol = float(cm.sum() * vox_ml)
                compact = vol * 1000 / (np.pi / 6 * max(diam, 1.0) ** 3)  # 1 = sphere
                if compact < min_compact:
                    continue
                cand = dict(
                    organ=organ, kind=kind, vol_ml=round(vol, 2), max_diam_mm=round(diam, 1),
                    compactness=round(compact, 2),
                    hu_mean=round(float(hu_raw.mean()), 1), organ_hu_median=round(med, 1),
                    z_range=[int(idx[:, 2].min() + off[2]), int(idx[:, 2].max() + off[2])], z_peak=z_peak,
                    box_xy=[int(idx[:, 0].min() + off[0]), int(idx[:, 0].max() + off[0]),
                            int(idx[:, 1].min() + off[1]), int(idx[:, 1].max() + off[1])],
                    centroid_vox=[round(float(c), 1) for c in c_vox],
                )
                if organ == "pancreas":
                    cand["part"] = _pancreas_part(case, c_vox)
                cand["score"] = cand["vol_ml"] * abs(cand["hu_mean"] - med)
                out.append(cand)
    out.sort(key=lambda c: -c["score"])
    out = out[:max_n]
    for i, c in enumerate(out, 1):
        c["id"] = f"C{i}"
    return out


def describe(c: dict) -> str:
    loc = f", pancreas {c['part']}" if "part" in c else ""
    return (f"{c['id']}: in {c['organ']}{loc}, {c['kind']}attenuating vs organ "
            f"(mean {c['hu_mean']:.0f} HU vs organ median {c['organ_hu_median']:.0f}), "
            f"volume {c['vol_ml']:.2f} mL, max diameter {c['max_diam_mm']:.0f} mm, "
            f"slices z={c['z_range'][0]}-{c['z_range'][1]} (largest at z={c['z_peak']})")


# lesion label produced by the segmentation tool for each organ (TotalSegmentator: liver tumors via the
# liver_vessels task, kidney cysts via the total task; it has NO pancreas or kidney tumor classes)
SEG_LESIONS = {"liver": ["liver_tumor"], "kidney_left": ["kidney_cyst_left"], "kidney_right": ["kidney_cyst_right"]}
SEG_LESIONS_GT = {"liver": ["liver_lesion"], "kidney_left": ["kidney_lesion_left"], "kidney_right": ["kidney_lesion_right"],
                  "pancreas": ["pancreatic_lesion"], "colon": ["colon_lesion"]}


def seg_lesions(case: Case) -> dict:
    """Organ -> lesion labels of the case's segmentation source (GT masks for the oracle ablation)."""
    return SEG_LESIONS_GT if case.meta.get("seg") == "gt" else SEG_LESIONS


def from_segmentation(case: Case, organs: list[str], max_n: int = 12, min_vol_ml: float = 0.02) -> list[dict]:
    """Lesion entities = connected components of the segmentation tool's lesion labels inside the organs."""
    sp = np.asarray(case.spacing)
    vox_ml = sp.prod() / 1000
    seg_layer = case.layers.get("liver_segments")
    out = []
    for organ in organs:
        for les in seg_lesions(case).get(organ, []):
            if not case.has(les):
                continue
            lab, n = ndimage.label(case.mask(les))
            for j, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                cm = lab[sl] == j
                vol = float(cm.sum() * vox_ml)
                if vol < min_vol_ml:
                    continue
                idx = np.argwhere(cm) + np.array([s.start for s in sl])
                zc = np.bincount(idx[:, 2] - sl[2].start)
                hu = case.ct[sl][cm]
                omask = case.mask(organ)[sl] & ~cm
                ohu = case.ct[sl][omask] if omask.any() else case.ct[case.mask(organ)]
                cand = dict(organ=organ, source=les, kind="segmented", vol_ml=round(vol, 2),
                            max_diam_mm=round(_max_diameter(cm, sp), 1), hu_mean=round(float(hu.mean()), 1),
                            organ_hu_median=round(float(np.median(ohu)), 1),
                            z_range=[int(idx[:, 2].min()), int(idx[:, 2].max())],
                            z_peak=int(zc.argmax()) + sl[2].start,
                            box_xy=[int(idx[:, 0].min()), int(idx[:, 0].max()), int(idx[:, 1].min()), int(idx[:, 1].max())],
                            centroid_vox=[round(float(c), 1) for c in idx.mean(0)])
                if seg_layer is not None and organ == "liver":
                    arr, names = seg_layer
                    v = arr[sl][cm]
                    v = v[v > 0]
                    if len(v):
                        cand["liver_segment"] = names[int(np.bincount(v).argmax()) - 1].replace("liver_segment_", "")
                out.append(cand)
    out.sort(key=lambda c: -c["vol_ml"])
    out = out[:max_n]
    for i, c in enumerate(out, 1):
        c["id"] = f"L{i}"
    return out


def describe_seg(c: dict) -> str:
    seg = f", Couinaud segment {c['liver_segment']}" if "liver_segment" in c else ""
    rel = "hypo" if c["hu_mean"] < c["organ_hu_median"] - 10 else ("hyper" if c["hu_mean"] > c["organ_hu_median"] + 10 else "iso")
    return (f"{c['id']}: {c['source']} component in {c['organ']}{seg}; volume {c['vol_ml']:.2f} mL, "
            f"max diameter {c['max_diam_mm']:.0f} mm, mean {c['hu_mean']:.0f} HU vs surrounding organ "
            f"{c['organ_hu_median']:.0f} HU ({rel}attenuating), slices z={c['z_range'][0]}-{c['z_range'][1]} "
            f"(largest at z={c['z_peak']})")


# ---------------------------------------------------------------------- measurement helpers (round 4)

def contacts(case: Case, m: np.ndarray, ring_vox: int = 2, exclude: tuple = ()) -> dict[str, float]:
    """Structures touching mask m: share of the ring (m dilated by ring_vox voxels, minus m) per label."""
    sl = ndimage.find_objects(m.astype(np.uint8))[0]
    pad = tuple(slice(max(s.start - ring_vox - 1, 0), s.stop + ring_vox + 1) for s in sl)
    mm = m[pad]
    ring = ndimage.binary_dilation(mm, iterations=ring_vox) & ~mm
    lab = case.labels[pad][ring]
    tot = max(int(ring.sum()), 1)
    out = {}
    for k, n in zip(*np.unique(lab, return_counts=True)):
        if k == 0:
            continue
        name = case.names[k - 1]
        if name in exclude:
            continue
        f = n / tot
        if f >= 0.02:
            out[name] = round(float(f), 2)
    return dict(sorted(out.items(), key=lambda t: -t[1]))


def measure_local(case: Case, z: int, box_xy, max_extent_mm: float = 120.0, return_mask: bool = False) -> dict:
    """Approximate lesion measurement from a VLM evidence box on axial slice z (training-free region growing).

    Seed = inner half of the box; voxels within a two-sided HU band around the seed median (half-width = half the
    seed-vs-surrounding-soft-tissue contrast, 15-60 HU); connected component through the seed inside an ROI (box x2 in-plane,
    +-max(box size) in z). Flags 'unreliable' when contrast is weak or the component leaks to the ROI border."""
    sp = np.asarray(case.spacing)
    X, Y, Z = case.shape
    x0, x1, y0, y1 = [int(v) for v in box_xy]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = max((x1 - x0) / 2, 2), max((y1 - y0) / 2, 2)
    size_mm = max(2 * hx * sp[0], 2 * hy * sp[1])
    hz = int(np.ceil(min(max(size_mm, 10), max_extent_mm) / sp[2]))
    rx0, rx1 = int(max(cx - 2 * hx - 3, 0)), int(min(cx + 2 * hx + 3, X - 1))
    ry0, ry1 = int(max(cy - 2 * hy - 3, 0)), int(min(cy + 2 * hy + 3, Y - 1))
    rz0, rz1 = max(z - hz, 0), min(z + hz, Z - 1)
    roi = ndimage.gaussian_filter(case.ct[rx0:rx1 + 1, ry0:ry1 + 1, rz0:rz1 + 1].astype(np.float32), 0.8)
    zc = z - rz0
    sx0, sx1 = int(cx - hx / 2) - rx0, int(cx + hx / 2) - rx0
    sy0, sy1 = int(cy - hy / 2) - ry0, int(cy + hy / 2) - ry0
    seed = roi[max(sx0, 0):sx1 + 1, max(sy0, 0):sy1 + 1, zc]
    inbox = np.zeros(roi.shape[:2], bool)
    inbox[x0 - rx0:x1 - rx0 + 1, y0 - ry0:y1 - ry0 + 1] = True
    # reference = parenchyma of the organ under the box (outside the box) on this slice; fallback: soft tissue around it
    lab2 = case.labels[rx0:rx1 + 1, ry0:ry1 + 1, z]
    under = lab2[inbox]
    under = under[under > 0]
    ring = np.array([])
    org = None
    if len(under):
        org = np.bincount(under).argmax()
        ring = roi[:, :, zc][(lab2 == org) & ~inbox]
    if len(ring) < 20:
        ring = roi[:, :, zc][~inbox]
        soft = ring[ring > -100]
        ring = soft if len(soft) >= 20 else ring
    s_med, r_med = float(np.median(seed)), float(np.median(ring))
    out = {"seed_hu": round(s_med, 1), "surround_hu": round(r_med, 1)}
    if abs(s_med - r_med) < 15:
        out["reliable"] = False
        out["reason"] = "weak contrast between box centre and surroundings"
        return out
    tol = float(np.clip(abs(s_med - r_med) / 2, 15, 30))
    sel = np.abs(roi - s_med) < tol          # two-sided band around the seed (excludes fat / air / bone)
    if org is not None:                      # stay within the organ under the box (+3 voxels for exophytic lesions)
        om = case.labels[rx0:rx1 + 1, ry0:ry1 + 1, rz0:rz1 + 1] == org
        sel &= ndimage.binary_dilation(om, iterations=3)
    # slice-by-slice 2D tracking (avoids 3D leaks through water-like neighbours such as the renal pelvis)
    comp = np.zeros(sel.shape, bool)
    lab0, _ = ndimage.label(sel[:, :, zc])
    ids = lab0[max(sx0, 0):sx1 + 1, max(sy0, 0):sy1 + 1]
    ids = ids[ids > 0]
    if not len(ids):
        out["reliable"] = False
        out["reason"] = "no component through the box centre"
        return out
    comp[:, :, zc] = lab0 == np.bincount(ids).argmax()
    a0 = comp[:, :, zc].sum()
    leak = bool(comp[:, :, zc][[0, -1], :].any() or comp[:, :, zc][:, [0, -1]].any())
    for step in (1, -1):
        prev, amax = comp[:, :, zc], a0
        k = zc + step
        while 0 <= k < sel.shape[2]:
            labk, _ = ndimage.label(sel[:, :, k])
            ov = labk[prev]
            ov = ov[ov > 0]
            if not len(ov):
                break
            cur = labk == np.bincount(ov).argmax()
            a = cur.sum()
            if a < 4 or a > 2 * amax or a < 0.15 * amax:   # vanished / leaked / tapered into a thin structure
                break
            if cur[[0, -1], :].any() or cur[:, [0, -1]].any():
                leak = True
                break
            comp[:, :, k] = cur
            prev, amax = cur, max(amax, a)
            k += step
    full = np.zeros(case.shape, bool)
    full[rx0:rx1 + 1, ry0:ry1 + 1, rz0:rz1 + 1] = comp
    idx = np.argwhere(comp)
    area = np.bincount(idx[:, 2])
    zmax = int(area.argmax()) + rz0
    out.update(reliable=not leak, kind="hypo" if s_med < r_med else "hyper",
               vol_ml=round(float(comp.sum() * sp.prod() / 1000), 2),
               max_diam_mm=round(_max_diameter(comp, sp), 1),
               z_range=[int(idx[:, 2].min() + rz0), int(idx[:, 2].max() + rz0)], z_max_area=zmax,
               mean_hu=round(float(case.ct[full].mean()), 1))
    if leak:
        out["reason"] = "region reached the ROI border (boundary unclear)"
    under = case.labels[full]
    under = under[under > 0]
    if len(under):
        out["inside"] = case.names[int(np.bincount(under).argmax()) - 1]
    if "liver_segments" in case.layers:
        arr, names = case.layers["liver_segments"]
        v = arr[full]
        v = v[v > 0]
        if len(v):
            out["liver_segment"] = names[int(np.bincount(v).argmax()) - 1].replace("liver_segment_", "")
    out["contacts"] = contacts(case, full, exclude=(out.get("inside"),))
    if return_mask:                          # offline evaluation only
        out["_mask"] = full
    return out


def describe_measure(m: dict, pct=None) -> str:
    if not m.get("vol_ml"):
        return f"code measurement failed ({m.get('reason')}; box centre {m['seed_hu']} HU vs surroundings {m['surround_hu']} HU)"
    s = (f"code measurement ({'approximate' if m['reliable'] else 'UNRELIABLE: ' + m['reason']}): {m['kind']}attenuating "
         f"(mean {m['mean_hu']:.0f} HU vs surroundings {m['surround_hu']:.0f}), volume {m['vol_ml']:.2f} mL, max diameter "
         f"{m['max_diam_mm']:.0f} mm, z {m['z_range'][0]}-{m['z_range'][1]}, largest cross-section at z={m['z_max_area']}")
    if pct is not None:
        s += f" ({pct:.1f}% slice position)"
    if m.get("inside"):
        s += f", mostly inside {m['inside']}"
    if m.get("liver_segment"):
        s += f", Couinaud segment {m['liver_segment']}"
    if m.get("contacts"):
        s += ", touches " + ", ".join(f"{k} ({v * 100:.0f}% of its border)" for k, v in m["contacts"].items())
    return s
