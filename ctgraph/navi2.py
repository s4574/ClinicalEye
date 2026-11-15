"""Two-agent reading, v2: a text-only NAVIGATOR writes its own questions to a VLM PERCEIVER and chooses what is rendered.

Navigator (LLM, no images) sees: question + options + patient + benchmark definitions + segmented structures + the
segmentation tool's lesion list + its own MEMORY (rewritten every turn) + the history of actions and replies.
Each turn ONE action:
  {"action": "measure", "targets": [...]}   numbers computed by code from the segmentation / anatomical graph
  {"action": "ask", "view": {...}, "question": "..."}   code renders the requested view, the perceiver answers in free
        text; findings it boxes (BOX lines) are measured by code (candidates.measure_local) and become F# findings
  {"action": "answer", "answer": "<letter>", ...}
Every rendered view is described back to the navigator in a text RECEIPT (what is on which panel), because it cannot
see the images. v0 / v1 (fixed verify / search templates) stay in navi.py.
"""
from __future__ import annotations

import json
import re

import numpy as np
from PIL import Image, ImageDraw

from . import render as R
from . import views as V
from . import candidates as CAND
from .agent import CONV, Episode, parse_options, question_stem
from .graph import LESION_ORGAN
from .hee import Tree
from .knowledge import knowledge_text
from .llm import LLM, extract_json, user_msg
from .reference import flags
from .volume import Case

MAX_TURNS = 6
MAX_PANELS = 12
YEL, RED, CYAN = (255, 230, 0), (255, 60, 60), (60, 220, 255)
OL_COLORS = [(255, 230, 0), (60, 220, 255), (80, 255, 80), (255, 80, 255), (255, 150, 0)]

NAV2 = """You are the NAVIGATOR of a two-agent CT reading system. You CANNOT see images. A vision model (the PERCEIVER, an
expert radiologist) looks at CT views that code renders for you and answers the questions YOU write. Your job: work out
which facts the question needs, collect them with the tools below, and decide the answer using the benchmark definitions.

TOOLS (one per turn):
1. measure - numbers computed by code from automatic segmentation (TotalSegmentator: organs and large vessels; its lesion
   classes are ONLY liver tumors and kidney cysts, it can miss lesions or over-call them; no pancreatic / renal solid tumor
   class). Targets: structure names below, tool lesion ids L#, perceiver finding ids F#.
   {{"action": "measure", "targets": ["<name or id>", ...], "why": "<short>", "memory": "<see below>"}}
2. ask - code renders a view and the perceiver answers your question in free text.
   {{"action": "ask", "view": {{"target": "<structure name | L# | F# | full>", "plane": "axial|coronal|sagittal",
     "slices": "center|range|thick", "z": <int, optional centre>, "z_from": <int>, "z_to": <int>, "thick_mm": <number, for thick>,
     "margin_mm": <number, default 20>, "overlay": [<any of "tool_lesions", "outline", "anatomy_map">], "locator": true|false,
     "window": "soft_tissue|narrow|vessel|bone"}},
    "question": "<your question to the perceiver>", "why": "<short>", "memory": "<see below>"}}
   - target: what the view is cropped to (a margin is added). "full" = whole axial slice.
   - slices: "center" = 3 adjacent thin slices around z (default: the target's largest cross-section); "range" = up to 12
     evenly spaced thin slices z_from..z_to; "thick" = consecutive slabs of thick_mm over z_from..z_to (good for scanning a
     whole organ, can hide small lesions). Coronal / sagittal: 3 planes through the target.
   - overlay: "tool_lesions" = tool lesion contours (red, with ids); "outline" = target structure outline (yellow);
     "anatomy_map" = extra image with all segmented structures / Couinaud segments outlined and labelled (axial only).
     "locator" = extra image: full axial slice with the crop boxed. "narrow" window = W200 around the target organ's mean HU.
   - Write NEUTRAL questions: say where to look and what to describe or decide (presence, number, size, density vs the organ,
     margins, internal structure, enhancement, duct dilatation, relation to vessels / organs, ...). Do NOT state the answer
     you expect or the tool's claims as facts, and never tell the perceiver what it should find, which finding to prefer or
     what to ignore. When replies or measurements conflict, show the conflicting findings (e.g. a wider view or another plane)
     and ask neutrally which is correct. The perceiver can box findings; boxed findings are measured by code (F# ids).
3. answer - {{"action": "answer", "answer": "<option letter>", "confidence": <0-1>, "derivation": "<facts -> definition -> answer>"}}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.
{kb}
SEGMENTED STRUCTURES: {structures}
TOOL LESIONS (segmentation tool; unverified):
{tool}
PERCEIVER FINDINGS (boxed by the perceiver, measured by code):
{findings}

YOUR MEMORY (written by you last turn): {memory}

HISTORY:
{history}

Turns left (including this one): {left}. When none are left after this one you must answer.
Every non-answer reply must include "memory": your updated notes for the next turn (established facts with their source,
open hypotheses, what is still to check). Reply with STRICT JSON only."""

PER2 = """You are the PERCEIVER, an expert abdominal radiologist. A navigator who cannot see images asks you about the CT
views below. {conv}
IMAGES: {receipt}
NAVIGATOR'S QUESTION: {question}
Answer from what you SEE in these images: be specific (which panel, location, approximate size, density relative to the
organ, margins, internal structure, relation to neighbouring structures) and say plainly when something cannot be judged
from these views. Do not assume the question's premises are true. For each focal finding you describe, you may add a line
BOX <panel id> [left, top, right, bottom]   (fractions 0-1 of that panel; axial panels only) so code can measure it.
Keep the answer under 150 words."""


class Navi2Agent:
    def __init__(self, nav: LLM, per: LLM, max_turns: int = MAX_TURNS):
        self.nav, self.per, self.max_turns = nav, per, max_turns

    # ------------------------------------------------------------------ text context
    def _tool_lesions(self, tree: Tree) -> dict:
        out = {}
        for t in tree.targets:
            for n in tree.lesions_of(t):
                out[f"L{len(out) + 1}"] = n
        return out

    def _segments_of(self, g, n):
        return sorted([(e["frac_b"], e["a"].split("_")[-1]) for e in g["edges"]
                       if e["b"] == n and e["a"].startswith("liver_segment")], reverse=True)

    def _lesion_line(self, tree: Tree, lid: str, n: str) -> str:
        nd = tree.g["nodes"][n]
        segs = self._segments_of(tree.g, n)
        org = LESION_ORGAN.get(n.split("#")[0], "?")
        return (f"{lid}: {n.split('#')[0]} in {org}" + (f", Couinaud segment {segs[0][1]}" if segs else "")
                + f", {nd['volume_ml']:.2f} mL, max diameter {nd['max_diameter_mm']:.0f} mm, z {nd['bbox_vox'][4]}-{nd['bbox_vox'][5]}")

    def _structures(self, tree: Tree) -> str:
        g, c = tree.g, tree.case
        parts = [f"{c.shape[2]} axial slices (z=0 inferior, spacing {c.spacing[2]:.1f} mm)"]
        for t in tree.targets:
            parts.append(f"TARGET {t} z {g['nodes'][t]['bbox_vox'][4]}-{g['nodes'][t]['bbox_vox'][5]}")
        if "liver" in tree.targets:
            parts.append("liver_segment_1..8")
        others = [n for n in c.names if n not in tree.targets and not n.startswith(("rib_", "vertebrae_T", "costal"))
                  and n not in LESION_ORGAN and n in g["nodes"]]
        parts.append("others: " + ", ".join(others))
        return "; ".join(parts)

    # ------------------------------------------------------------------ measure
    def _measure(self, tree: Tree, st: dict, targets) -> str:
        g, out = tree.g, []
        names = [str(t) for t in (targets if isinstance(targets, list) else [targets])][:8]
        for t in names:
            if t in st["L"]:
                n = st["L"][t]
                nd, info = g["nodes"][n], tree.lesion_info(n)
                org = LESION_ORGAN.get(n.split("#")[0])
                share = (f", {100 * nd['volume_ml'] / g['nodes'][org]['volume_ml']:.1f}% of the {org} volume"
                         if org in g["nodes"] else "")
                out.append(f"{self._lesion_line(tree, t, n)}, mean HU {nd['hu_mean']:.0f} (std {nd['hu_std']:.0f}){share}"
                           + (", touches " + ", ".join(f"{k} ({v * 100:.0f}% of its border)" for k, v in info["contacts"].items())
                              if info.get("contacts") else ""))
            elif t in st["F"]:
                out.append(f"{t}: " + st["F"][t]["text"])
            elif t in g["nodes"] and "#" not in t:
                nd = g["nodes"][t]
                nb = sorted([e for e in g["edges"] if t in (e["a"], e["b"]) and e.get("type") == "adjacent"
                             and e["min_dist_mm"] <= 3 and "#" not in e["a"] + e["b"]], key=lambda e: -e["contact_area_mm2"])[:6]
                touch = ", ".join(f"{e['b'] if e['a'] == t else e['a']} ({e['contact_area_mm2']:.0f} mm2)" for e in nb)
                fl = flags(g, t)
                out.append(f"{t}: volume {nd['volume_ml']:.1f} mL, max diameter {nd['max_diameter_mm']:.0f} mm, mean HU "
                           f"{nd['hu_mean']:.1f} (std {nd['hu_std']:.0f}), z {nd['bbox_vox'][4]}-{nd['bbox_vox'][5]}"
                           + (f"; reference: {fl}" if fl else "") + (f"; touches {touch}" if touch else ""))
            else:
                out.append(f"{t}: unknown (use a structure name from SEGMENTED STRUCTURES, L# or F#)")
        if {"liver", "spleen"} <= set(names) and {"liver", "spleen"} <= set(g["nodes"]):
            out.append(f"liver/spleen HU ratio {g['nodes']['liver']['hu_mean'] / g['nodes']['spleen']['hu_mean']:.2f}")
        return "measure: " + " | ".join(out)

    # ------------------------------------------------------------------ rendering
    def _target(self, tree: Tree, st: dict, name: str):
        """-> (mask or None, in-plane box, z_ref, organ for the window, description)"""
        c = tree.case
        if name in st["L"]:
            m = tree._lesion_mask(st["L"][name])
            zr = int(m.sum((0, 1)).argmax())
            idx = np.argwhere(m[:, :, zr])
            return m, [int(idx[:, 0].min()), int(idx[:, 0].max()), int(idx[:, 1].min()), int(idx[:, 1].max())], zr, \
                LESION_ORGAN.get(st["L"][name].split("#")[0]), f"tool lesion {name}"
        if name in st["F"]:
            f = st["F"][name]
            return None, f["box"], f["z"], f.get("inside"), f"perceiver finding {name}"
        m = None
        if name.startswith("liver_segment_") and "liver_segments" in c.layers:
            arr, names = c.layers["liver_segments"]
            if name in names:
                m = arr == names.index(name) + 1
        elif c.has(name):
            m = c.mask(name)
        if m is None or not m.any():
            return None, None, None, None, None
        idx = np.argwhere(m)
        zr = int(m.sum((0, 1)).argmax())
        return m, [int(idx[:, 0].min()), int(idx[:, 0].max()), int(idx[:, 1].min()), int(idx[:, 1].max())], zr, \
            ("liver" if name.startswith("liver") else name), name

    def _win(self, tree: Tree, organ, w):
        if w == "narrow" and organ in tree.g["nodes"]:
            return (tree.g["nodes"][organ]["hu_mean"], 200), f"narrow (W200 around {organ} mean HU)"
        w = w if w in ("soft_tissue", "vessel", "bone") else "soft_tissue"
        return w, w

    def _contour(self, img, rc, color, label=None):
        if not rc.any():
            return img
        img = R.overlay_mask(img, rc, color=color, alpha=0.0)
        if label:
            s = img.size[0] / rc.shape[1]
            rr, cc = np.nonzero(rc)
            x, y = cc.max() * s + 3, max(rr.min() * s - 4, 0)
            d = ImageDraw.Draw(img)
            d.rectangle([x - 1, y, x + 9 * len(label) + 2, y + 15], fill=(0, 0, 0))
            d.text((x, y), label, fill=color, font=R._font(13))
        return img

    def _render(self, tree: Tree, st: dict, view: dict):
        """-> (images, receipt text, panels {id: (z, crop)}) or (None, error text, None)"""
        c, sp = tree.case, tree.sp
        view = view if isinstance(view, dict) else {}
        name = str(view.get("target") or "full")
        ov = set(view.get("overlay") or [])
        if name == "full":
            m, box, zr, organ, desc = None, [0, c.shape[0] - 1, 0, c.shape[1] - 1], int(c.shape[2] // 2), (tree.targets or [None])[0], "full slice"
            crop = box
        else:
            m, box, zr, organ, desc = self._target(tree, st, name)
            if box is None:
                return None, f"unknown target '{name}' (use a structure name, L#, F# or full)", None
            mm = float(view.get("margin_mm") or 20)
            crop = V.pad_box(box, c.shape, mm, sp, 60 if name in st["L"] or name in st["F"] else 70)
            fov = view.get("fov_mm")
            if isinstance(fov, (int, float)) and fov > 0:   # gated3: in-plane zoom of the given field of view around the target
                zf = int(np.clip(view.get("z") if isinstance(view.get("z"), int) else zr, 0, c.shape[2] - 1))
                if m is not None and m[:, :, zf].any() and name not in st["L"]:
                    cx, cy = np.argwhere(m[:, :, zf]).mean(0)
                else:
                    cx, cy = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
                hx, hy = max(float(fov), 30.0) / 2 / sp[0], max(float(fov), 30.0) / 2 / sp[1]
                crop = [int(max(cx - hx, 0)), int(min(cx + hx, c.shape[0] - 1)), int(max(cy - hy, 0)), int(min(cy + hy, c.shape[1] - 1))]
        cb = view.get("crop_box")
        if isinstance(cb, (list, tuple)) and len(cb) == 4:      # NV5 rules: explicit in-plane crop (voxels)
            crop = [int(np.clip(cb[0], 0, c.shape[0] - 1)), int(np.clip(cb[1], 0, c.shape[0] - 1)),
                    int(np.clip(cb[2], 0, c.shape[1] - 1)), int(np.clip(cb[3], 0, c.shape[1] - 1))]
        pfx = str(view.get("panel_prefix") or "")
        extra_ol = []                                            # NV5: several structures outlined in distinct colours
        for k, nm in enumerate(view.get("outline_structures") or []):
            mm_ = self._target(tree, st, str(nm))[0]
            if mm_ is not None:
                extra_ol.append((str(nm), mm_, OL_COLORS[k % len(OL_COLORS)]))
        win, win_txt = self._win(tree, organ, view.get("window"))
        tool = [(lid, tree._lesion_mask(n)) for lid, n in st["L"].items()] if "tool_lesions" in ov else []
        if getattr(self, "no_overlay", False):     # 2026-11-15 variant (structure-probe finding): no outlines / contours / boxes /
            ov -= {"outline", "tool_lesions", "anatomy_map"}   # anatomy map drawn on the images; the code-computed text stays
            extra_ol, tool = [], []
        plane = view.get("plane") or "axial"
        Z = c.shape[2]
        zc = int(np.clip(view.get("z") if isinstance(view.get("z"), int) else (zr if zr is not None else Z // 2), 0, Z - 1))
        ims, panels = [], {}
        x0, x1, y0, y1 = crop

        def ax_mask(m3, a, b):
            return m3[x0:x1 + 1, y0:y1 + 1, a:b + 1].any(2)[::-1, ::-1].T

        if plane == "axial":
            mode = view.get("slices") or "center"
            if mode in ("range", "thick"):
                if m is not None:
                    zz = np.nonzero(m.any((0, 1)))[0]
                    lo, hi = int(zz.min()), int(zz.max())
                else:
                    lo, hi = max(zc - 10, 0), min(zc + 10, Z - 1)
                a = int(np.clip(view.get("z_from") if isinstance(view.get("z_from"), int) else lo, 0, Z - 1))
                b = int(np.clip(view.get("z_to") if isinstance(view.get("z_to"), int) else hi, 0, Z - 1))
                a, b = min(a, b), max(a, b)
                if m is not None:                 # keep the range near the target (12 panels over the whole body are useless)
                    pad = int(round(60 / sp[2]))
                    a, b = max(a, lo - pad), min(b, hi + pad)
                    if a > b:
                        a, b = lo, hi
                # (2026-11-06 fix: targets without a mask - 'full', F# - used to be clamped to the volume centre +-10 slices,
                #  so an explicit z range was ignored, e.g. bench_05259 asked for z 520-580 and got 375-395)
                if mode == "thick":
                    k = max(1, int(round(float(view.get("thick_mm") or 5) / sp[2])))
                    k = max(k, int(np.ceil((b - a + 1) / MAX_PANELS)))
                    slabs = [(s, min(s + k - 1, b)) for s in range(a, b + 1, k)][::-1]
                else:
                    nsl = int(view.get("n_slices") or MAX_PANELS)
                    zs = sorted(set(np.linspace(a, b, min(nsl, MAX_PANELS, b - a + 1)).round().astype(int).tolist()), reverse=True)
                    slabs = [(z, z) for z in zs]
            else:
                d = 1 if (name in st["L"] or name in st["F"]) else tree.d
                slabs = [(min(zc + d, Z - 1),) * 2, (zc, zc), (max(zc - d, 0),) * 2]
            size = 384 if len(slabs) <= 3 else 256
            for i, (a, b) in enumerate(slabs):
                sub = c.ct[x0:x1 + 1, y0:y1 + 1, a:b + 1].astype(np.float32).mean(2)
                im, s = R.upscale(R.to_rgb(R.window(sub[::-1, ::-1].T, win)), size)
                if "outline" in ov and m is not None and name not in st["L"]:
                    im = self._contour(im, ax_mask(m, a, b), YEL)
                for lid, lm in tool:
                    im = self._contour(im, ax_mask(lm, a, b), RED, lid)
                for nm, mm_, col in extra_ol:
                    im = self._contour(im, ax_mask(mm_, a, b), col, nm.replace("_", " ")[:18])
                if (name in st["L"] or name in st["F"]) and a <= zc <= b and "tool_lesions" not in ov and not getattr(self, "no_overlay", False):
                    r0, c0, r1, c1 = V.box_to_rc(box, c.shape)
                    cr0, cc0, _, _ = V.box_to_rc(crop, c.shape)
                    im = R.draw_boxes(im, [(r0 - cr0, c0 - cc0, r1 - cr0, c1 - cc0)], [""], s, width=2)
                pid = f"{pfx}P{i + 1}"
                panels[pid] = ((a + b) // 2, crop)
                ims.append(V.label(im, f"{pid} z{a}" + (f"-{b}" if b > a else ""), 14))
            img = V.montage(ims, 3 if len(ims) <= 3 else 4)
            zs_txt = ", ".join(f"{p} z={a}" + (f"-{b}" if b > a else "") for p, (a, b) in zip(panels, slabs))
            parts = [f"Image 1: {len(slabs)} axial {'slabs' if slabs[0][1] > slabs[0][0] else 'slices'} (superior first) "
                     f"cropped to {desc} (field {(x1 - x0) * sp[0]:.0f} x {(y1 - y0) * sp[1]:.0f} mm), window {win_txt}: {zs_txt}."]
            if (name in st["L"] or name in st["F"]) and "tool_lesions" not in ov and any(a <= zc <= b for a, b in slabs):
                parts.append(f"The {desc} is boxed on the panel containing z={zc}.")
            out = [img]
        else:
            ax = 1 if plane == "coronal" else 0
            if m is not None:
                idx = np.argwhere(m)
                vs = [int(np.quantile(idx[:, ax], q)) for q in (0.25, 0.5, 0.75)]
                lo, hi = idx.min(0), idx.max(0)
            else:
                cx, cy = (box[0] + box[1]) // 2, (box[2] + box[3]) // 2
                v0 = cy if ax == 1 else cx
                vs = [v0 - 5, v0, v0 + 5]
                lo = np.array([crop[0], crop[2], max(zc - 40, 0)])
                hi = np.array([crop[1], crop[3], min(zc + 40, Z - 1)])
            mz = int(round(float(view.get("margin_mm") or 20) / sp[2]))
            z0, z1 = max(int(lo[2]) - mz, 0), min(int(hi[2]) + mz, Z - 1)
            # 2026-11-14 fix 7: an explicit z range is honoured for coronal / sagittal views too (it used to be replaced by the
            # target's extent +- margin, e.g. z 450-530 requested -> 233-363 rendered)
            if isinstance(view.get("z_from"), int) and isinstance(view.get("z_to"), int):
                a_, b_ = sorted((int(np.clip(view["z_from"], 0, Z - 1)), int(np.clip(view["z_to"], 0, Z - 1))))
                if b_ - a_ >= 4:
                    z0, z1 = a_, b_
            plane_content = []
            for i, v in enumerate(vs):
                if ax == 1:
                    sl, px = (slice(x0, x1 + 1), int(np.clip(v, 0, c.shape[1] - 1)), slice(z0, z1 + 1)), sp[0]
                else:
                    sl, px = (int(np.clip(v, 0, c.shape[0] - 1)), slice(y0, y1 + 1), slice(z0, z1 + 1)), sp[1]
                f = lambda a: a[sl][::-1, ::-1].T
                im = R.to_rgb(R.window(f(c.ct).astype(np.float32), win))
                h, w = im.size[1] * sp[2], im.size[0] * px
                s = 320 / max(h, w)
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
                if "outline" in ov and m is not None:
                    im = self._contour(im, f(m), YEL)
                for lid, lm in tool:
                    im = self._contour(im, f(lm), RED, lid)
                for nm, mm_, col in extra_ol:
                    im = self._contour(im, f(mm_), col, nm.replace("_", " ")[:18])
                pid = f"{pfx}P{i + 1}"
                ims.append(V.label(im, f"{pid} {'y' if ax == 1 else 'x'}={v}", 13))
                # 2026-11-14 fix 7: code-computed content of the plane (axial panels always had it, these had none)
                lab = c.labels[sl]
                ids_, cnt_ = np.unique(lab[lab > 0], return_counts=True)
                txt_ = ", ".join(f"{c.names[k - 1]} {n / lab.size * 100:.0f}%" for k, n in sorted(zip(ids_, cnt_), key=lambda t: -t[1])
                                 if n / lab.size >= 0.01) or "no segmented structure >= 1%"
                inp = [lid for lid, lm in tool if lm[sl].any()]
                plane_content.append(f"{pid} {'y' if ax == 1 else 'x'}={v}: {txt_}" + (f"; tool lesions in this plane: {', '.join(inp)}" if inp else ""))
            out = [V.montage(ims, 3)]
            parts = [f"Image 1: 3 {plane} planes P1-P3 through {desc} (image top = superior; image left = "
                     f"{'patient right' if ax == 1 else 'patient anterior'}), z {z0}-{z1}, window {win_txt}. "
                     f"Code-computed content per plane: " + "; ".join(plane_content) + "."]
        if "outline" in ov and m is not None and name not in st["L"]:
            parts.append(f"{desc} outlined in yellow (segmentation, can be imperfect).")
        if tool:
            parts.append("Red contours with ids = segmentation tool lesions (can be wrong).")
        if extra_ol:
            parts.append("Outlined (labelled, segmentation, can be imperfect): " + ", ".join(nm for nm, _, _ in extra_ol) + ".")
        if "anatomy_map" in ov and plane == "axial":
            out.append(tree.anatomy_map(zc, crop, 384, snap=True))
            parts.append(f"Image {len(out)}: ANATOMY MAP of z={zc} (segmented structures and Couinaud segments 'seg k' "
                         "outlined and labelled; outlines can be imperfect).")
        if view.get("locator"):
            out.append(V.label(V.locator(c, zc, crop, 320), f"locator z={zc}", 14))
            parts.append(f"Image {len(out)}: full axial slice z={zc} with the crop region boxed.")
        return out, " ".join(p for p in parts if p), panels

    def _ask(self, tree: Tree, st: dict, view, question: str, ep: Episode):
        ims, receipt, panels = self._render(tree, st, view)
        if ims is None:
            return None, f"ask: {receipt}"
        r = self.per.chat([user_msg(PER2.format(conv=CONV, receipt=receipt, question=question[:800]),
                                    [R.b64png(im) for im in ims])], max_tokens=800)
        ep.cost += r["cost"]
        text = r["text"].strip()
        notes = []
        for pid, nums in re.findall(r"BOX\s+(P\d+)\s*\[([^\]]+)\]", text):
            if pid not in panels or len(st["F"]) >= 12:
                continue
            try:
                frac = [float(v) for v in nums.split(",")]
            except ValueError:
                continue
            if len(frac) != 4:
                continue
            z, crop = panels[pid]
            b = V.crop_frac_to_box(crop, frac)
            try:
                meas = CAND.measure_local(tree.case, z, b)
            except Exception as e:
                meas = {"reliable": False, "reason": type(e).__name__, "seed_hu": float("nan"), "surround_hu": float("nan")}
            fid = f"F{len(st['F']) + 1}"
            st["F"][fid] = {"z": z, "box": b, "inside": meas.get("inside"),
                            "text": f"boxed on {pid} (z={z}); " + CAND.describe_measure(meas)}
            notes.append(f"{fid} = {st['F'][fid]['text']}")
        clean = re.sub(r"BOX\s+P\d+\s*\[[^\]]+\]", "", text).strip()
        h = (f"ask view={json.dumps(view)}\n  RECEIPT: {receipt}\n  QUESTION: {question[:300]}\n  PERCEIVER: {clean[:900]}"
             + ("".join(f"\n  CODE: {n}" for n in notes)))
        return {"receipt": receipt, "reply": text, "findings": notes}, h

    # ------------------------------------------------------------------ loop
    def run(self, case: Case, g: dict, targets: list[str], item: dict) -> Episode:
        tree = Tree(case, g, targets)
        ep = Episode(item)
        st = {"L": self._tool_lesions(tree), "F": {}}
        tool_lines = [self._lesion_line(tree, k, n) for k, n in st["L"].items()]
        tool_txt = "\n".join(tool_lines[:20] + ([f"... {len(tool_lines) - 20} more (smaller)"] if len(tool_lines) > 20 else [])) \
            or "(none in the target organs)"
        base = dict(question=question_stem(item["mc_question"]), options=json.dumps(parse_options(item["mc_question"])),
                    sex=item.get("sex"), age=item.get("age"), kb=knowledge_text(item["mc_question"]),
                    structures=self._structures(tree), tool=tool_txt)
        history, memory, final = [], "(empty)", None
        for turn in range(self.max_turns + 1):
            left = self.max_turns + 1 - turn
            prompt = NAV2.format(**base, findings="\n".join(f"{k}: {v['text']}" for k, v in st["F"].items()) or "(none)",
                                 memory=memory, history="\n".join(history) or "(none)", left=left)
            if left == 1:
                prompt += '\nYou MUST now reply with the "answer" action.'
            r = self.nav.chat([user_msg(prompt)], max_tokens=2500)
            ep.cost += r["cost"]
            j = extract_json(r["text"])
            ep.steps.append({"step": turn, "mode": "nav", "reply": j or r["text"][:600]})
            act = (j or {}).get("action")
            if act == "answer" or left == 1 or not j:
                final = j if act == "answer" else None
                break
            memory = str(j.get("memory") or memory)[:1500]
            if act == "measure":
                h = self._measure(tree, st, j.get("targets") or [])
                ep.steps.append({"step": turn, "mode": "code", "query": "measure", "reply": h})
            elif act == "ask":
                pj, h = self._ask(tree, st, j.get("view"), str(j.get("question") or ""), ep)
                ep.steps.append({"step": turn, "mode": "per", "query": {"view": j.get("view"), "question": j.get("question")}, "reply": pj})
            else:
                h = f"unknown action {act}"
            history.append(f"turn {turn}: {h}")
        if final is None:
            prompt = NAV2.format(**base, findings="\n".join(f"{k}: {v['text']}" for k, v in st["F"].items()) or "(none)",
                                 memory=memory, history="\n".join(history) or "(none)", left=0) + '\nYou MUST now reply with the "answer" action.'
            r = self.nav.chat([user_msg(prompt)], max_tokens=2500)
            ep.cost += r["cost"]
            final = extract_json(r["text"]) or {}
        ep.final = {"answer": final.get("answer"), "confidence": final.get("confidence"), "rationale": final.get("derivation"),
                    "memory": memory}
        ep.evidence = [{"id": k, **v} for k, v in st["F"].items()]
        return ep
