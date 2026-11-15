"""Stage-2 search agent: question-guided exploration of a CT volume over the anatomical graph.

Loop (stateless prompts; every step sees question, graph summary, plan, evidence log, action history
and only the newest observation's images):
  step 0  overview of the target structures (8-level montage + coronal with the levels marked)
  step k  the VLM returns JSON {observation, plan(+confidence), evidence[], action, best, confidence}
          actions: scan (z-range montage) | view (adjacent-slice triplet) | zoom (sub-box triplet)
                   | propose (HU-outlier candidates, optional) | back (previous observation) | answer
  final   a separate decision call with the plan and the evidence log; evidence images optional
          (none / crops / crops + full-slice locator), which is an ablation axis.
All directions given to the VLM are pre-translated to image directions (P2). Plan edits are only
accepted when the VLM's plan_update_confidence >= PLAN_EDIT_CONF (enforced in code).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from . import candidates as CAND
from . import render as R
from . import views as V
from .graph import describe_relation, neighbors
from .index import SpatialIndex
from .reference import flags
from .graph import LESION_ORGAN, lesion_organ
from .llm import LLM, extract_json, user_msg
from .volume import Case

PLAN_EDIT_CONF = 0.7
CONV = ("Display convention for every axial image: image top = patient ANTERIOR, image bottom = POSTERIOR, "
        "image LEFT = patient RIGHT, image RIGHT = patient LEFT. Coronal: image top = SUPERIOR. "
        "Slice index z increases toward SUPERIOR (head).")


def parse_options(mc: str) -> dict[str, str]:
    parts = re.split(r"\s([A-F]):\s", " " + mc)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def question_stem(mc: str) -> str:
    return re.split(r"\sA:\s", mc)[0].strip()


@dataclass
class Obs:
    kind: str
    text: str
    images: list  # PIL images
    z: int | None = None
    crop_box: list | None = None   # voxel (x0, x1, y0, y1) of the CENTER image, for zoom / evidence boxes
    win: str = "soft_tissue"


@dataclass
class Episode:
    item: dict
    steps: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    cost: float = 0.0


class CTEnv:
    """Renders observations for one case; knows nothing about the LLM."""

    def __init__(self, case: Case, g: dict, targets: list[str], use_candidates: bool = False):
        self.case, self.g, self.ix = case, g, SpatialIndex(case)
        self.targets = [t for t in targets if t in g["nodes"]]
        self.sp = case.spacing
        self.d = max(1, int(round(5.0 / self.sp[2])))  # adjacent-slice offset ~5 mm
        self.win = "narrow" if self.targets and self.targets[0] == "liver" else "soft_tissue"
        self.use_candidates = use_candidates
        self._cands = None
        self.graph_mode = "full"      # full | inst | anat | anat_les | nolesion  (graph summary content)
        self.extra_text = ""          # appended to the graph summary (e.g. slice-percentage conversion); "" keeps old prompts
        self.seen: set[int] = set()   # axial slices the agent has looked at (any scale)

    def W(self, win) -> tuple | str:
        """Window name -> render window. 'narrow' adapts to the target organ's mean HU (contrast phases
        differ a lot between scans, a fixed liver window saturates)."""
        if win in ("narrow", "liver"):
            hu = self.g["nodes"][self.targets[0]]["hu_mean"] if self.targets else 60
            return (hu, 200)
        return win if win in R.WINDOWS else "soft_tissue"

    def coverage_text(self) -> str:
        out = []
        for t in self.targets:
            z0, z1 = self.g["nodes"][t]["bbox_vox"][4:6]
            seen = sorted(z for z in self.seen if z0 <= z <= z1)
            pts = [z0 - 1] + seen + [z1 + 1]
            gaps = [(a + 1, b - 1) for a, b in zip(pts, pts[1:]) if (b - a - 1) * self.sp[2] > 10]
            gaps = sorted(gaps, key=lambda g: -(g[1] - g[0]))[:3]
            out.append(f"{t}: seen {len(seen)}/{z1 - z0 + 1} slices; largest unseen gaps (>10 mm): "
                       + (", ".join(f"z={a}..{b} ({(b - a + 1) * self.sp[2]:.0f} mm)" for a, b in gaps) or "none"))
        return "; ".join(out)

    # ---- geometry helpers
    def struct_box(self, names=None, margin_mm=30, min_mm=90):
        names = names or self.targets
        b = V.union_bbox(self.g, names)
        return V.pad_box(b[:4], self.case.shape, margin_mm, self.sp, min_mm), (b[4], b[5])

    def resolve_structs(self, s) -> list[str]:
        if not s:
            return self.targets
        s = [s] if isinstance(s, str) else list(s)
        out = [n for n in s if n in self.g["nodes"]]
        return out or self.targets

    def slice_context(self, z: int, box) -> str:
        """Structures on this slice inside the crop, with image-frame offsets from the crop centre."""
        x0, x1, y0, y1 = box
        sub = self.case.labels[x0:x1 + 1, y0:y1 + 1, z]
        ids, cnt = np.unique(sub, return_counts=True)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        parts = []
        for k, n in sorted(zip(ids, cnt), key=lambda t: -t[1]):
            if k == 0 or n * self.sp[0] * self.sp[1] < 40:
                continue
            xy = np.argwhere(sub == k).mean(0) + [x0, y0]
            ph = V.image_frame_phrase(((xy[0] - cx) * self.sp[0], (xy[1] - cy) * self.sp[1]))
            parts.append(f"{self.case.names[k - 1]} ({ph} from crop centre)")
        return "; ".join(parts[:10]) or "no labelled structure"

    # ---- observations
    def overview(self) -> Obs:
        imgs, txt = [], []
        groups = [self.targets] if (getattr(self, "union_overview", False) or
                                    (len(self.targets) <= 2 and all("kidney" in t for t in self.targets))) else [[t] for t in self.targets]
        for grp in groups[:2]:
            box, (z0, z1) = self.struct_box(grp)
            zs = np.linspace(z0, z1, 8).round().astype(int).tolist()[::-1]
            crops = [V.label(V.axial_crop(self.case, z, box, self.W(self.win), 256), f"z={z}") for z in zs]
            self.seen.update(zs)
            imgs.append(V.montage(crops, 4))
            yc = int(np.mean([self.g["nodes"][n]["centroid_mm"][1] for n in grp]) / self.sp[1])
            imgs.append(V.coronal_with_levels(self.case, yc, zs, 512))
            txt.append(f"Overview of {', '.join(grp)}: image {len(imgs) - 1} = 8 axial crops (same crop box) at "
                       f"z={zs} (superior first, row-major); image {len(imgs)} = coronal slice through it with those "
                       f"levels marked. {', '.join(grp)} spans z={z0}..{z1}.")
        return Obs("overview", " ".join(txt), imgs, None, None)

    def scan(self, structure=None, z_from=None, z_to=None, n=8, win=None) -> Obs:
        names = self.resolve_structs(structure)
        box, (z0, z1) = self.struct_box(names)
        a = int(np.clip(z_from if z_from is not None else z0, 0, self.case.shape[2] - 1))
        b = int(np.clip(z_to if z_to is not None else z1, 0, self.case.shape[2] - 1))
        a, b = min(a, b), max(a, b)
        n = int(np.clip(n or 8, 2, 8))
        zs = sorted(set(np.linspace(a, b, n).round().astype(int).tolist()), reverse=True)
        win = win or self.win
        crops = [V.label(V.axial_crop(self.case, z, box, self.W(win), 256), f"z={z}") for z in zs]
        self.seen.update(zs)
        return Obs("scan", f"Scan of {', '.join(names)} at z={zs} (superior first, row-major), window={win}.",
                   [V.montage(crops, 4)], None, box, win)

    def view(self, z, structure=None, win=None) -> Obs:
        names = self.resolve_structs(structure)
        box, _ = self.struct_box(names)
        z = int(np.clip(z, 0, self.case.shape[2] - 1))
        win = win or self.win
        img = V.triplet(self.case, z, box, self.d, self.W(win), 384)
        self.seen.update((z - self.d, z, z + self.d))
        txt = (f"View at z={z} cropped to {', '.join(names)}; three panels = z+{self.d} (above), z (CENTER), z-{self.d} (below), "
               f"window={win}. Labelled structures on the CENTER slice: {self.slice_context(z, box)}.")
        return Obs("view", txt, [img], z, box, win)

    def zoom(self, z, box_frac, base: Obs, win=None) -> Obs:
        if base is None or base.crop_box is None:
            raise ValueError("zoom needs a previous view/zoom/scan with a crop box")
        z = int(np.clip(z if z is not None else base.z, 0, self.case.shape[2] - 1))
        nb = V.crop_frac_to_box(base.crop_box, box_frac)
        nb = V.pad_box(nb, self.case.shape, 3, self.sp, 30)
        win = win or base.win
        img = V.triplet(self.case, z, nb, self.d, self.W(win), 384)
        self.seen.update((z - self.d, z, z + self.d))
        loc = V.locator(self.case, z, nb, 256)
        txt = (f"Zoom at z={z} (adjacent slices z+{self.d}/z/z-{self.d}), field of view "
               f"{(nb[1] - nb[0]) * self.sp[0]:.0f}x{(nb[3] - nb[2]) * self.sp[1]:.0f} mm, window={win}. Second image: "
               f"full slice with the zoomed region outlined. Labelled structures in view: {self.slice_context(z, nb)}.")
        return Obs("zoom", txt, [img, loc], z, nb, win)

    @property
    def seg_tool(self) -> bool:
        return self.case.meta.get("seg") in ("totalseg", "gt")

    @property
    def seg_gt(self) -> bool:
        """Oracle ablation: the segmentation tool returns the AbdomenAtlas GT masks (organs, Couinaud segments, lesions)."""
        return self.case.meta.get("seg") == "gt"

    def candidates(self) -> list[dict]:
        if self._cands is None:
            self._cands = (CAND.from_segmentation(self.case, self.targets) if self.seg_tool
                           else CAND.propose(self.case, self.targets))
        return self._cands

    def propose(self) -> Obs:
        cs = self.candidates()
        if not cs:
            what = ("The segmentation tool found no liver tumor / kidney cyst components in the target structures "
                    "(it cannot detect pancreatic or renal solid tumors, and can miss small lesions)."
                    if self.seg_tool else "No intensity-outlier candidates found inside the target structures.")
            return Obs("propose", what, [])
        crops = []
        for c in cs[:8]:
            box = V.pad_box(c["box_xy"], self.case.shape, 10, self.sp, 40)
            crops.append(V.label(V.axial_crop(self.case, c["z_peak"], box, self.W(self.win), 256, mark_box=c["box_xy"]),
                                 f"{c['id']} z={c['z_peak']}"))
        if self.seg_tool:
            txt = ("Lesion components from the automatic segmentation tool (TotalSegmentator; detects liver tumors and "
                   "kidney cysts ONLY - not pancreatic or renal solid tumors; may contain false positives and miss lesions). "
                   "Measurements are computed from the masks. Verify each visually (boxes on the crops):\n"
                   + "\n".join(CAND.describe_seg(c) for c in cs))
        else:
            txt = ("Automatic intensity-outlier candidates (UNRELIABLE: many are vessels, partial-volume or normal "
                   "parenchymal variation; small lesions are often missed). Verify visually:\n" +
                   "\n".join(CAND.describe(c) for c in cs))
        return Obs("propose", txt, [V.montage(crops, 4)])

    def evidence_images(self, ev: dict, size=320):
        b, z = ev["box_vox"], ev["z"]
        crop = V.pad_box(b, self.case.shape, 15, self.sp, 50)
        return (V.label(V.axial_crop(self.case, z, crop, self.W(ev.get("win", self.win)), size, mark_box=b), f"E{ev['id']} z={z}"),
                V.label(V.locator(self.case, z, b, size), f"E{ev['id']} location"))

    # ---- graph text
    def _instance_lines(self) -> list[str]:
        """Per-lesion listing (each connected component of the tool's lesion labels) + lesion COUNTS per liver
        segment / kidney side - the units the questions ask about (not volume shares)."""
        g = self.g
        out = [("Lesions found by the segmentation tool (liver, kidney, pancreatic and colon lesions; tumors and cysts are not "
                "distinguished; each separate component listed once; the tool can miss lesions or produce false positives):")
               if self.seg_gt else
               "Lesions found by the segmentation tool (TotalSegmentator: liver tumors and kidney cysts only; each separate "
               "component listed once; the tool can miss lesions or produce false positives):"]
        for t in self.targets:
            labels = [l for l, o in lesion_organ(self.case.meta.get("seg")).items() if o == t]
            inst = sorted([n for n in g["nodes"] if "#" in n and n.split("#")[0] in labels], key=lambda n: -g["nodes"][n]["volume_ml"])
            if not labels:
                continue
            if not inst:
                out.append(f"- {t}: no lesion segmented")
                continue
            seg_count = {}
            out.append(f"- {t}: {len(inst)} lesion(s)")
            for n in inst:
                nd = g["nodes"][n]
                segs = sorted([(e["frac_b"], e["a"].split("_")[-1]) for e in g["edges"]
                               if e["b"] == n and e["a"].startswith("liver_segment")], reverse=True)
                main = segs[0][1] if segs else None
                if main:
                    seg_count[main] = seg_count.get(main, 0) + 1
                out.append(f"    {n}: " + (f"Couinaud segment {main}" + (f" (also {', '.join(s for _, s in segs[1:])})" if len(segs) > 1 else "") + ", " if main else "")
                           + f"volume {nd['volume_ml']:.2f} mL, max diameter {nd['max_diameter_mm']:.0f} mm, slices z={nd['bbox_vox'][4]}..{nd['bbox_vox'][5]}")
            if t == "liver":
                out.append("    lesion COUNT per Couinaud segment (by the main segment of each lesion): "
                           + ", ".join(f"segment {k}: {seg_count.get(str(k), 0)}" for k in range(1, 9)))
        return out if len(out) > 1 else []

    def graph_summary(self, max_edges=14) -> str:
        if self.graph_mode == "none":     # 2026-11-14 ablation: no anatomical-graph text at all; only extra lines (KB, slice %)
            return self.extra_text or "(not provided)"
        g, lines = self.g, []
        meas = self.graph_mode in ("full", "nolesion", "inst")  # organ measurements (volume / diameter / HU / reference)
        lesions = self.graph_mode in ("full", "anat_les")       # segmentation-tool lesion information
        SL = CAND.seg_lesions(self.case)
        les_labels = set(SL.get(t, [None])[0] for t in SL) | set(lesion_organ(self.case.meta.get("seg")))
        for t in self.targets:
            n = g["nodes"][t]
            if meas:
                lines.append(f"- {t}: volume {n['volume_ml']:.0f} mL, max diameter {n['max_diameter_mm']:.0f} mm, "
                             f"mean HU {n['hu_mean']:.0f} (std {n['hu_std']:.0f}), slices z={n['bbox_vox'][4]}..{n['bbox_vox'][5]}"
                             + (f"\n    reference: {flags(g, t)}" if flags(g, t) else ""))
            else:
                lines.append(f"- {t}: slices z={n['bbox_vox'][4]}..{n['bbox_vox'][5]}")
        rel = []
        for t in self.targets:
            nb = [e for e in neighbors(g, t) if e.get("type", "adjacent") == "adjacent"]
            if not lesions:      # drop lesion label nodes and per-instance lesion nodes ("<label>#k")
                nb = [e for e in nb if not ({e["a"], e["b"]} & les_labels) and "#" not in e["a"] + e["b"]]
            for e in nb[:max_edges // max(len(self.targets), 1)]:
                o = e["b"] if e["a"] == t else e["a"]
                rel.append("- " + describe_relation(g, o, t))
        lesion_lines = []
        if self.seg_tool and self.graph_mode == "inst":
            lesion_lines = self._instance_lines()
        elif self.seg_tool and lesions:
            les = {l for t in self.targets for l in SL.get(t, [])}
            for l in sorted(les):
                if l in g["nodes"]:
                    n = g["nodes"][l]
                    vol = f"total {n['volume_ml']:.2f} mL in " if meas or self.graph_mode == "full" else ""
                    lesion_lines.append(f"- {l}: {vol}{n['n_components']} component(s), "
                                        f"slices z={n['bbox_vox'][4]}..{n['bbox_vox'][5]} (use the propose action for per-lesion details)")
                    for e in g["edges"]:
                        if e.get("type") == "overlaps" and l in (e["a"], e["b"]):
                            lesion_lines.append("    " + describe_relation(g, e["a"], e["b"]))
                elif any(l in SL.get(t, []) for t in self.targets):
                    lesion_lines.append(f"- {l}: none segmented")
            lesion_lines = ([("Lesion labels from the segmentation tool (liver, kidney, pancreatic and colon lesions; tumors and cysts "
                              "are not distinguished; can miss or over-call lesions):") if self.seg_gt else
                             ("Lesion labels from the segmentation tool (TotalSegmentator: liver tumors and kidney cysts only; "
                              "no pancreatic/renal solid tumor classes; can miss or over-call lesions):")] + lesion_lines) if lesion_lines else []
        head = ("Target structures (from automatic segmentation" if self.seg_gt else
                "Target structures (from automatic TotalSegmentator segmentation" if self.seg_tool
                else "Target structures (from automatic organ segmentation; organ masks only, no lesion masks")
        return (head + "; NOTE a lesion "
                "may or may not be included in its organ's mask, so an abnormal organ size, diameter or left/right asymmetry "
                "can itself be a sign of a mass):\n"
                + "\n".join(lines) + ("\n" + "\n".join(lesion_lines) if lesion_lines else "")
                + "\nNeighbouring structures (patient frame):\n" + "\n".join(rel)
                + (("\n" + self.extra_text) if self.extra_text else ""))


SYS = """You are an expert abdominal radiologist operating a CT viewer to answer a multiple-choice question.
You cannot see the whole 3D volume at once; you explore it step by step. {conv}
Lesions range from small (1 cm) to very large masses. Check adjacent slices before calling a small spot a lesion
(vessels continue across slices, lesions are focal). Conversely, a LARGE round/heterogeneous structure filling much of
the crop may itself be a mass replacing the organ - compare with the expected normal anatomy, the contralateral
organ and the reference volumes in the graph summary instead of assuming it is normal parenchyma.

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.

ANATOMICAL GRAPH SUMMARY:
{graph}

AVAILABLE ACTIONS (choose exactly one per step):
- {{"type": "scan", "structure": <name or null>, "z_from": int, "z_to": int, "n": <2-8>, "window": <optional>}}  montage of axial crops over a z-range (use to localise along z; narrow the range like a binary search)
- {{"type": "view", "z": int, "structure": <name or null>, "window": <optional>}}  one slice with its neighbours above/below
- {{"type": "zoom", "z": int, "box": [left, top, right, bottom]}}  box as fractions (0-1) of the CENTER panel of the most recent view/zoom (or of the crop of the most recent scan)
{propose_line}- {{"type": "back"}}  return to the previous observation (use when a zoom turned out to be a dead end)
- {{"type": "answer"}}  stop exploring; a final decision step will follow
windows: soft_tissue (W400), narrow (W200 centred on the target organ's mean HU; best for subtle parenchymal lesions), vessel, bone.
Remaining steps: {remaining}. Answer when your confidence is high or when further search is unlikely to help. For existence / counting
questions do not conclude "absent" while large parts of the organ are unseen (see Coverage).

STATE
Plan (clues to look for): {plan}
Evidence so far: {evidence}
Coverage of the target structures: {coverage}
History: {history}

CURRENT OBSERVATION: {obs}

Reply with STRICT JSON only:
{{"observation": "<what you see in the current images, 1-3 sentences>",
  "plan": [{{"clue": "<what to look for and where>", "status": "pending|found|absent"}}],
  "plan_update_confidence": <0-1, how sure you are that the plan must change; plan text edits are ignored below {plan_conf}>,
  "evidence": [{{"z": int, "box": [l, t, r, b] or null, "finding": "<what>", "supports": "<option letter or null>", "confidence": <0-1>}}],
  "action": {{...}},
  "best_option": "<letter>", "confidence": <0-1>}}
Only add NEW evidence seen in the CURRENT images; boxes refer to the CENTER panel of the current view/zoom."""

FINAL = """You are an expert abdominal radiologist. Decide the final answer to a multiple-choice question about a CT scan
based on the evidence collected by an exploration agent. {conv}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.

ANATOMICAL GRAPH SUMMARY:
{graph}

EXPLORATION PLAN: {plan}
EVIDENCE LOG (z = axial slice index; structures = segmented structures inside the evidence box):
{evidence}
{images_note}
Reply with STRICT JSON only: {{"answer": "<letter>", "confidence": <0-1>, "rationale": "<1-2 sentences>"}}"""

BASELINE = """You are an expert abdominal radiologist answering a multiple-choice question about a CT scan. {conv}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.
{graph}
IMAGES: {obs}

Reply with STRICT JSON only: {{"answer": "<letter>", "confidence": <0-1>, "rationale": "<1-2 sentences>"}}"""


def _fmt_plan(plan):
    return "; ".join(f"[{p.get('status', 'pending')}] {p.get('clue', '')}" for p in plan) or "(none yet - propose one)"


def _fmt_evidence(ev):
    if not ev:
        return "(none)"
    return "\n".join(f"E{e['id']}: z={e['z']}, structures={e['structures']}, finding: {e['finding']} "
                     f"(supports {e.get('supports')}, conf {e.get('confidence')})" for e in ev)


def apply_action(env: CTEnv, stack: list, a: dict, use_candidates: bool) -> str | None:
    """Execute one non-answer action on the observation stack (shared by Agent.run and trajectory replay).
    Returns an error string, or None."""
    t = a.get("type", "answer")
    try:
        if t == "scan":
            nob = env.scan(a.get("structure"), a.get("z_from"), a.get("z_to"), a.get("n", 8), a.get("window"))
        elif t == "view":
            nob = env.view(a.get("z", (env.ix.z_range(env.targets[0])[0])), a.get("structure"), a.get("window"))
        elif t == "zoom":
            base = next((o for o in reversed(stack) if o.crop_box is not None), None)
            nob = env.zoom(a.get("z"), a.get("box", [0.25, 0.25, 0.75, 0.75]), base, a.get("window"))
        elif t == "propose" and use_candidates:
            nob = env.propose()
        elif t == "back":
            if len(stack) > 1:
                stack.pop()
            return None
        else:
            raise ValueError(f"unknown action {t}")
    except Exception as e:
        return str(e)
    stack.append(nob)
    return None


class Agent:
    def __init__(self, llm: LLM, max_steps: int = 8, final_mode: str = "crops+locator",
                 use_graph: bool = True, use_candidates: bool = False):
        self.llm, self.max_steps, self.final_mode = llm, max_steps, final_mode
        self.use_graph, self.use_candidates = use_graph, use_candidates

    def _ctx(self, env: CTEnv, item: dict) -> dict:
        return dict(conv=CONV, question=question_stem(item["mc_question"]),
                    options=json.dumps(parse_options(item["mc_question"])), sex=item.get("sex"), age=item.get("age"),
                    graph=env.graph_summary() if self.use_graph else "(not provided)")

    def _call(self, ep: Episode, text: str, imgs) -> dict:
        r = self.llm.chat([user_msg(text, [R.b64png(i) for i in imgs])])
        ep.cost += r["cost"]
        return r

    def baseline(self, env: CTEnv, item: dict) -> Episode:
        ep = Episode(item)
        ob = env.overview()
        ctx = self._ctx(env, item)
        ctx["graph"] = ("ANATOMICAL GRAPH SUMMARY:\n" + ctx["graph"]) if self.use_graph else ""
        r = self._call(ep, BASELINE.format(obs=ob.text, **ctx), ob.images)
        ep.final = {**extract_json(r["text"]), "raw": r["text"][:500]}
        return ep

    def run(self, env: CTEnv, item: dict) -> Episode:
        ep = Episode(item)
        ctx = self._ctx(env, item)
        stack: list[Obs] = [env.overview()]
        history, last_view = [], None
        for step in range(self.max_steps):
            obs = stack[-1]
            prompt = SYS.format(
                propose_line=(('- {"type": "propose"}  lesion components found by the segmentation tool in the target structures, '
                               'with code-computed measurements (liver tumors / kidney cysts only; verify each)\n') if env.seg_tool else
                              ('- {"type": "propose"}  automatic intensity-outlier lesion candidates inside the target '
                               'structures (unreliable; verify each)\n')) if self.use_candidates else "",
                remaining=self.max_steps - step, plan=_fmt_plan(ep.plan), plan_conf=PLAN_EDIT_CONF,
                evidence=_fmt_evidence(ep.evidence), history="; ".join(history) or "(start)",
                coverage=env.coverage_text(),
                obs=obs.text, **ctx)
            r = self._call(ep, prompt, obs.images)
            j = extract_json(r["text"])
            rec = {"step": step, "obs_kind": obs.kind, "obs_text": obs.text, "reply": j or r["text"][:800]}
            ep.steps.append(rec)
            if not j:
                history.append(f"s{step}: unparseable reply")
                continue
            self._update_plan(ep, j)
            self._add_evidence(env, ep, j.get("evidence") or [], obs)
            a = j.get("action") or {}
            t = a.get("type", "answer")
            history.append(f"s{step}: {json.dumps(a)} (best {j.get('best_option')}, conf {j.get('confidence')})")
            if t == "answer":
                break
            err = apply_action(env, stack, a, self.use_candidates)
            if err:
                history[-1] += f" -> ERROR {err[:80]}"
        self._final(env, ep, ctx)
        return ep

    def _update_plan(self, ep: Episode, j: dict):
        new = [p for p in (j.get("plan") or []) if isinstance(p, dict)]
        if not new:
            return
        if not ep.plan:
            ep.plan = new
            return
        old_clues = [p.get("clue") for p in ep.plan]
        new_clues = [p.get("clue") for p in new]
        try:
            conf = float(j.get("plan_update_confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if new_clues != old_clues and conf < PLAN_EDIT_CONF:
            # keep the old clue list, accept status updates by position
            for p, q in zip(ep.plan, new):
                p["status"] = q.get("status", p.get("status"))
            ep.steps[-1]["plan_edit_rejected"] = True
        else:
            ep.plan = new

    def _add_evidence(self, env: CTEnv, ep: Episode, evs: list, obs: Obs):
        for e in evs:
            if not isinstance(e, dict):
                continue
            z = e.get("z", obs.z)
            if z is None or obs.crop_box is None:
                box = None
            else:
                frac = e.get("box")
                box = V.crop_frac_to_box(obs.crop_box, frac) if isinstance(frac, list) and len(frac) == 4 else obs.crop_box
            try:
                z = int(z)
            except (TypeError, ValueError):
                z = None
            structs = {}
            if box is not None and z is not None:
                structs = env.ix.box_nodes(box[0], box[1], box[2], box[3], z, z)
                tot = sum(structs.values()) or 1
                structs = {k: round(v / tot, 2) for k, v in sorted(structs.items(), key=lambda t: -t[1])[:4]}
            ep.evidence.append(dict(id=len(ep.evidence) + 1, z=z, box_vox=box, structures=structs,
                                    finding=str(e.get("finding", ""))[:300], supports=e.get("supports"),
                                    confidence=e.get("confidence"), win=obs.win, step=len(ep.steps) - 1))

    def _final(self, env: CTEnv, ep: Episode, ctx: dict):
        imgs, note = [], ""
        ev = [e for e in ep.evidence if e["box_vox"] is not None and e["z"] is not None]
        ev = sorted(ev, key=lambda e: -(float(e.get("confidence") or 0)))[:4]
        if self.final_mode != "text" and ev:
            pairs = [env.evidence_images(e) for e in ev]
            ids = ", ".join(f"E{e['id']}" for e in ev)
            if self.final_mode == "crops":
                imgs = [V.montage([p[0] for p in pairs], 2)]
                note = f"IMAGES: one montage of the evidence crops {ids} (box = evidence region)."
            else:
                imgs = [V.montage([x for p in pairs for x in p], 2)]
                note = (f"IMAGES: one montage; each row = evidence crop (box = evidence region) and the full axial slice "
                        f"with its location, for {ids}.")
        r = self._call(ep, FINAL.format(plan=_fmt_plan(ep.plan), evidence=_fmt_evidence(ep.evidence),
                                        images_note=note, **ctx), imgs)
        ep.final = {**extract_json(r["text"]), "raw": r["text"][:500], "n_images": len(imgs)}
