"""Client of the nnInteractive service (ctgraph/nni_server.py) + code-side reliability check of the result.

measure_nni(case, z, box) -> a measurement dict in the format of candidates.measure_local (so describe_measure and the
F# bookkeeping work unchanged) plus "source": "nnInteractive" and the reliability features:
  contrast    |mean HU of the result on slice z - organ parenchyma median| / robust organ SD (organ = TotalSeg organ
              under the box; parenchyma = that organ on slice z minus the result, eroded 2 voxels)
  organ_share share of that organ's area on slice z covered by the result
  box_ratio   box area / result area on slice z
reliable = contrast >= 1.0 and organ_share < 0.5 and box_ratio <= 4 (fixed rule, chosen a priori; offline the contrast
of results seeded with tight GT-lesion boxes had median 2.2 vs 0.4 on lesion-free boxes, runs/nni_nolesion_score.txt).
nnInteractive returns a region for ANY box, so the check is what tells "probably no distinct lesion" apart."""
from __future__ import annotations

import base64
import json
import urllib.request

import numpy as np
from scipy import ndimage

from ctgraph.candidates import _max_diameter, contacts
from ctgraph.volume import Case

URL = "http://127.0.0.1:8765/segment"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # local service: never via http_proxy
ORGANS = ("kidney_left", "kidney_right", "pancreas", "liver", "spleen")
NOT_LESION = ("aorta", "inferior_vena_cava", "portal_vein", "superior_mesenteric", "iliac", "splenic_vein", "vertebra",
              "rib", "pulmonary", "colon", "small_bowel", "duodenum", "stomach", "esophagus", "costal", "sacrum", "hip", "femur")


def _call(case_id: str, z: int, box) -> dict:
    req = urllib.request.Request(URL, data=json.dumps({"case_id": case_id, "z": int(z), "box": [int(v) for v in box]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=300) as r:
        return json.loads(r.read())


def _organ_under(case: Case, box, z):
    x0, x1, y0, y1 = box
    sub = case.labels[max(x0, 0):x1 + 1, max(y0, 0):y1 + 1, z]
    best, n = None, 0
    for o in ORGANS:
        if o in case.names:
            k = int((sub == case.name_to_id(o)).sum())
            if k > n:
                best, n = o, k
    return best


def measure_nni(case: Case, z: int, box, return_mask: bool = False) -> dict:
    x0, x1, y0, y1 = [int(v) for v in box]
    try:
        r = _call(case.case_id, z, (x0, x1, y0, y1))
    except Exception as e:
        return {"source": "nnInteractive", "reliable": False, "reason": f"segmentation service unavailable ({type(e).__name__})",
                "seed_hu": float("nan"), "surround_hu": float("nan")}
    if not r.get("ok") or r.get("empty"):
        return {"source": "nnInteractive", "reliable": False, "reason": r.get("error", "empty segmentation"),
                "seed_hu": float("nan"), "surround_hu": float("nan")}
    sh = r["shape"]
    crop = np.unpackbits(np.frombuffer(base64.b64decode(r["bits"]), np.uint8))[:int(np.prod(sh))].reshape(sh).astype(bool)
    full = np.zeros(case.shape, bool)
    o = r["offset"]
    full[o[0]:o[0] + sh[0], o[1]:o[1] + sh[1], o[2]:o[2] + sh[2]] = crop
    sp = np.asarray(case.spacing)
    sl = case.ct[:, :, z].astype(np.float32)
    pz = full[:, :, z]
    org_name = _organ_under(case, (x0, x1, y0, y1), z)
    out = {"source": "nnInteractive", "seconds": round(r.get("seconds", 0), 1)}
    barea = (x1 - x0 + 1) * (y1 - y0 + 1)
    contrast, organ_share, ref, rsd = None, None, None, None
    if pz.any() and org_name:
        org = case.labels[:, :, z] == case.name_to_id(org_name)
        organ_share = float((pz & org).sum() / max(org.sum(), 1))
        par = sl[ndimage.binary_erosion(org & ~pz, iterations=2)]
        if len(par) >= 30:
            ref = float(np.median(par))
            rsd = max(float(1.4826 * np.median(np.abs(par - ref))), 5.0)
            contrast = abs(float(sl[pz].mean()) - ref) / rsd
    box_ratio = barea / max(int(pz.sum()), 1)
    out.update(organ_checked=org_name, contrast=None if contrast is None else round(contrast, 2),
               organ_share=None if organ_share is None else round(organ_share, 2), box_ratio=round(box_ratio, 1))
    reasons = []
    if not pz.any():
        reasons.append("the segmented region does not lie on the boxed slice")
    if contrast is not None and contrast < 1.0:
        reasons.append(f"the segmented region is not distinct from the surrounding {org_name} "
                       f"(contrast {contrast:.1f} SD) - the box may contain no separate lesion")
    if organ_share is not None and organ_share >= 0.5:
        reasons.append(f"the result covers {organ_share * 100:.0f}% of the {org_name} on this slice - it probably "
                       f"segmented the organ, not a lesion")
    if box_ratio > 4:
        reasons.append(f"your box is {box_ratio:.0f}x larger than the segmented region - redraw a box that touches the "
                       f"object's borders")
    if org_name is None:
        reasons.append("the box is not on a segmented abdominal organ (kidney, pancreas, liver, spleen); reliability "
                       "could not be checked")
    elif contrast is None and (organ_share is None or organ_share < 0.5):
        reasons.append(f"too little {org_name} tissue left around the region to check its contrast")
    hu = float(case.ct[full].mean())
    if hu < -20:
        reasons.append(f"the region has fat / air density (mean {hu:.0f} HU)")
    lz = case.labels[:, :, z][pz]
    lz = lz[lz > 0]
    if len(lz):
        main = case.names[int(np.bincount(lz).argmax()) - 1]
        if main.startswith(NOT_LESION):
            reasons.append(f"the region lies mostly in the {main} (a vessel / bowel / bone structure)")
    out["reliable"] = not reasons
    if reasons:
        out["reason"] = "; ".join(reasons)
    out.update(seed_hu=round(float(sl[pz].mean()) if pz.any() else hu, 1),
               surround_hu=round(ref if ref is not None else float("nan"), 1),
               kind="hypo" if ref is not None and hu < ref else "hyper",
               vol_ml=round(float(full.sum() * sp.prod() / 1000), 2), max_diam_mm=round(_max_diameter(full, sp), 1))
    idx = np.argwhere(full)
    area = np.bincount(idx[:, 2])
    out.update(z_range=[int(idx[:, 2].min()), int(idx[:, 2].max())], z_max_area=int(area.argmax()), mean_hu=round(hu, 1))
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
    if return_mask:
        out["_mask"] = full
    return out
