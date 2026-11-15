"""NV5 (design: runs/design_nv5.md, 2026-11-08): the VLM starts from the question only (no images, no metadata such as
the benchmark's `organ` field) and drives the reading with tools:
  - view      -> the NAVIGATOR (multi-step): parses the free-text need against the segmented anatomy (+ anatomical
                 priors for unsegmented structures, marked approximate), renders candidates with deterministic rules,
                 looks at them (vision mode), adjusts (<= 3 renderings per request), and returns the chosen image set
                 with an explanation (why / what it shows / how to read / how the anatomy was located);
  - measure / compute / box / knowledge -> executed by code directly (VLM's own tools; names from the vocabulary).
Memories are kept by code: the VLM's step log + evidence ledger; the navigator's record of the image sets already shown
(request, localisation, final view, content, the radiologist's next conclusion) + a coverage table.
Every VLM step reports (current_answer, confidence) on a fixed verbal scale for calibration analysis.
navi_mode: "vision" (sees its renderings), "text" (code descriptions only).
"""
from __future__ import annotations

import ast
import json
import operator
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import render as R
from . import views as V
from . import candidates as CAND
from . import nni_client as NNI
from . import slicepct
from . import paths as P
from .agent import CONV, Episode, parse_options, question_stem
from .graph import LESION_ORGAN_TS
from .hee import Tree
from .knowledge import lookup as kb_lookup
from .llm import LLM, extract_json, user_msg
from .navi2 import Navi2Agent

MAX_TURNS = 6
MAX_RENDERS = 3
KEEP_SETS = 2
METRICS_PY = P.DTVQA_SRC / "deeptumorvqa/eval/metrics.py"

CONF_SCALE = """CONFIDENCE SCALE (probability that your current answer is correct; use the band that matches your belief):
0.0-0.1 almost no chance | 0.1-0.2 highly unlikely | 0.2-0.3 chances are slight | 0.3-0.4 unlikely |
0.4-0.5 less than even | 0.5-0.6 better than even | 0.6-0.7 likely | 0.7-0.8 very good chance |
0.8-0.9 highly likely | 0.9-1.0 almost certain"""

VLM5 = """You are an expert abdominal radiologist answering a multiple-choice question about ONE CT scan. You start with NO images:
use the tools below to look at the scan and to measure it, then answer. {conv}

QUESTION: {question}
OPTIONS: {options}
{patient}
STRUCTURE VOCABULARY - the ONLY names you can use in measure targets and compute expressions (view requests are free text):
- segmented structures of THIS scan (automatic segmentation, names only):
{vocab}
- Couinaud liver segments: {segments}
- segmentation-tool lesions of this scan: {lesions}
- findings you boxed: {boxed}
Anything not in this vocabulary (e.g. the pancreatic head / body / tail, a kidney pole, the superior mesenteric artery or
vein, the celiac trunk) is NOT segmented: to MEASURE it, first box it on an image you have seen and measure its F#.

TOOLS
You can call several of the non-image tools (measure, compute, box, knowledge) in one turn; at most ONE "view" per turn.
1. view: a navigator assistant prepares new images for you. Describe in your own words WHAT you need to see and WHY
   (e.g. "the relation of the pancreatic-head mass to the superior mesenteric artery, to judge whether contact exceeds
   180 degrees"). You do not need structure names from the vocabulary: the navigator knows the segmented anatomy, can
   locate structures that are not segmented from anatomical knowledge (it then says so and how), generates candidate
   images, looks at them, adjusts them if needed, and returns the image set with an explanation: why it chose it, what
   each panel shows and how to read it. Possible renderings: axial / coronal / sagittal planes, thin slices, slice ranges
   or thick slabs, zoom with a chosen field of view, soft-tissue / narrow (W200 around the organ's mean HU) / vessel /
   bone windows, outlines of structures, contours of the segmentation tool's lesions, a labelled anatomy map, a
   full-slice locator, and combined panels (zoomed detail + where it lies, several planes through one point, several
   slices). The navigator's explanation is for REFERENCE ONLY and can be wrong: it is meant to describe normal anatomy and
   how the images were chosen; if it says anything about findings or how something looks, ignore that; its statements
   about where structures lie (especially "approximate" ones) and about tool lesions come from segmentation or anatomical
   priors - verify them on the images. Judge the images yourself.
   With every new image set you also get, for each panel, the list of ALL segmented structures covering >= 1% of that
   panel's area (with their area share) and, for the liver, the Couinaud segment shares; plus the tool lesions (L#) and
   boxed findings (F#) inside the field of view. This is code-computed from automatic segmentation (can be imperfect).
2. measure: code measurements from automatic segmentation (TotalSegmentator). Targets: a name from the STRUCTURE
   VOCABULARY above (a structure, liver_segment_k, a tool lesion L#, or a boxed finding F#). Returns: volume (mL), max
   diameter (mm), mean HU (and std), slice range, comparison with typical adult reference ranges, touching structures with
   contact area; for lesions also the organ, Couinaud segment and share of the organ volume.
3. compute: exact arithmetic on measured values, e.g. "volume(kidney_left) + volume(pancreas)" or
   "mean_hu(liver) / mean_hu(spleen)" (functions: volume, mean_hu, max_diameter). Use it instead of doing arithmetic yourself.
4. box: mark something you see on an AXIAL panel of an image set you currently see (panel id such as S1P2 + [left, top,
   right, bottom] as fractions 0-1 of that panel). Code segments the boxed object around the box centre (region growing
   on HU) and returns its volume, max diameter, mean HU vs its surroundings, slice range, the structure it mostly lies in,
   touching structures, and whether the measurement is reliable. It gets an id F# that you can measure, zoom on or refer
   to. A FAILED or UNRELIABLE box measurement means the box probably missed the object; it is NOT evidence of absence.
5. knowledge: look up the benchmark's clinical definitions / thresholds (a fixed list of 27 criteria, keyword search),
   e.g. "splenomegaly grading", "PDAC vs PNET", "pancreatic cyst resection criteria".
Segmentation tool limits: lesion classes are ONLY liver tumors and kidney cysts (no pancreatic, renal solid or colon
tumor class); organ outlines and Couinaud segments can be imperfect; it can miss lesions or over-call them.

IMAGES YOU CURRENTLY SEE (in order): {images}

YOUR STEP LOG (previous turns: conclusion, answer, confidence, tool results):
{log}
YOUR EVIDENCE LEDGER (from your last turn): {ledger}
Turns left (including this one): {left}.

{conf_scale}

Rules: keep a previously given answer unless a ledger entry changes status because of NEW, specific evidence (a tool
result, or a finding you can localise on a NEW image). A new image that merely looks different, or a failed measurement,
is not enough. Answer when you are confident or when further tools are unlikely to change the answer.

Reply with STRICT JSON only:
{{"step_conclusion": "<what you learned this turn, 1-3 sentences>",
 "ledger": [{{"id": "H1", "claim": "<finding / hypothesis that bears on the answer>", "evidence": "<tool result or panel + z>",
             "status": "confirmed|refuted|pending"}}],
 "current_answer": "<option letter, your best answer now (also before you have seen images)>",
 "answer_change_reason": "<if current_answer changed: which new evidence changed which ledger entry; else null>",
 "confidence": <0-1, see the scale>,
 "calls": [{{"tool": "view", "request": "<what you need to see and why, in your own words>"}},
           {{"tool": "measure", "target": "<a name from the STRUCTURE VOCABULARY | L# | F#>"}},
           {{"tool": "compute", "expression": "<...>"}},
           {{"tool": "box", "panel": "<S#P#>", "box": [l, t, r, b], "label": "<short>"}},
           {{"tool": "knowledge", "query": "<keywords>"}}],
 "final": true|false}}
"final": true means you answer now (calls is then ignored).{force}"""

NAV5 = """You are the NAVIGATOR for a radiologist (a vision model) reading a CT. The radiologist described what they need to see.
Your job: work out which anatomy the request refers to, render images that show it as clearly and unambiguously as
possible, check the renderings yourself, adjust them if needed, and finally hand over the best image set with an
explanation. You may identify and locate NORMAL anatomical structures (also ones that are not segmented, from anatomical
knowledge - mark those "approximate" and say what you based the location on). You must NOT judge findings: never state
or suggest whether something is abnormal, present, absent, invasive or normal, and never describe how something looks
(density, enhancement, margins, size impression). Refer to lesions / findings only by their id (L#, F#). Anything taken from the
segmentation tool (which lesion lies in which segment / organ, sizes) must be written as "according to the segmentation
tool (unverified)", and never as a count or conclusion for the question. Overlay the segmentation tool's lesion contours only when the request
is about those tool lesions.

QUESTION: {question}
OPTIONS: {options}
RADIOLOGIST'S REQUEST: "{request}"
RADIOLOGIST'S STEP CONCLUSION: {conclusion}
SEGMENTED STRUCTURES with slice ranges (axial index, z=0 inferior; volume {nz} slices of {sz:.1f} mm): {zranges}
NEIGHBOUR RELATIONS from the anatomical graph (closest surfaces; z of the closest point): {relations}
TOOL LESIONS (segmentation tool; unverified): {tool}
FINDINGS BOXED BY THE RADIOLOGIST (code-measured): {findings}
IMAGE SETS ALREADY SHOWN TO THE RADIOLOGIST (your memory):
{memory}
COVERAGE OF WHAT HAS BEEN SHOWN (structure / plane / window / zoom -> slices): {coverage}
{conv}

RENDERINGS SO FAR FOR THIS REQUEST ({seen_note}; at most {max_r} renderings per request, {n_left} left):
{renders}

TOOLS (one per step):
- {{"do": "render", "rule": "overview", "args": {{"organ": "<name>"}}}}   8 axial crops over the organ + coronal with levels
- {{"do": "render", "rule": "organ", "args": {{"organ": "<name>", "window": "narrow|soft_tissue"}}}}   3 thin slices at its largest cross-section
- {{"do": "render", "rule": "relation", "args": {{"a": "<name|L#|F#>", "b": "<name|L#|F#>"}}}}   slices where a and b come closest, both outlined, crop covering both
- {{"do": "render", "rule": "finding", "args": {{"id": "<L#|F#>", "fov_mm": 40-80}}}}   zoom on the finding, adjacent slices + full-slice locator
- {{"do": "render", "rule": "location", "args": {{"target": "<name|L#|F#>"}}}}   labelled anatomy map (Couinaud segments / structures) + locator
- {{"do": "render", "rule": "extent", "args": {{"target": "<name>", "z_from": <int>, "z_to": <int>}}}}   evenly spaced slices over a range, outlined
- {{"do": "render", "rule": "spec", "args": {{"target": "<name|L#|F#|full>", "plane": "axial|coronal|sagittal",
    "slices": "center|range|thick", "z": <int>, "z_from": <int>, "z_to": <int>, "thick_mm": <n>, "fov_mm": <n>,
    "outline_structures": ["<names>"], "overlay": ["tool_lesions", "anatomy_map"], "locator": true|false,
    "window": "soft_tissue|narrow|vessel|bone"}}}}   (for structures that are not segmented: centre a zoomed view on a
    nearby segmented structure at the slice you expect them, and mark the localisation approximate)
- {{"do": "finish", "choice": ["R#"] or ["R#", "R#"], "why": "<why this serves the request better than the alternatives;
   what it adds to the image sets already shown>", "what_it_shows": "<for each panel: plane, slices, field of view, which
   structures / L# / F# are in it>", "how_to_read": "<what outlines / boxes / colours / locator mean>",
   "localisation": "<how the requested anatomy was located; which parts are approximate and on what basis>"}}
Before finishing, check on the renderings: is the requested anatomy actually shown, clear, not cut off and not hidden by
overlays; does the set add information beyond the image sets already shown (see coverage / overlap); could it be misread
(e.g. a small crop without context)? Do not render something nearly identical to a rendering you already have for this
request (see "overlap with R#"). Reply with STRICT JSON only:
{{"parse": "<which anatomy the request refers to and how you locate it>",
 "check": "<what the renderings so far show about the requested ANATOMY (not findings)>",
 "next": {{<one tool>}}}}{force}"""


# v2 (2026-11-08, user): anatomy lookup for the navigator + explicit direction definitions
ANATOMY_KB = P.ANATOMY_KB
DIRECTIONS = """DIRECTIONS: all anatomical relations (anatomy index, neighbour relations, your explanations) are in the PATIENT frame
(standard anatomical position): right / left = the PATIENT's right / left; anterior = toward the front of the body;
posterior = toward the back; superior = toward the head; inferior = toward the feet.
How they appear here: axial images - patient anterior at the image TOP, posterior at the BOTTOM, patient RIGHT on the
IMAGE LEFT, patient LEFT on the IMAGE RIGHT; coronal images - superior at the top, patient right on the image left;
sagittal images - superior at the top, patient anterior on the image left. Slice index z increases toward SUPERIOR.
In explanations for the radiologist, write patient-frame terms and add the image side where it helps, e.g.
"to the patient's left of the aorta (image right)"."""


@lru_cache(1)
def anatomy_kb() -> list[dict]:
    return json.load(open(ANATOMY_KB))


def anatomy_lookup(query: str, k: int = 2) -> str:
    """Entries whose id / aliases match the query (substring either way, case-insensitive; best = most matches)."""
    stop = {"the", "of", "and", "a", "an", "to", "in", "on", "its", "with", "part", "parts"}
    norm = lambda t: [w.rstrip("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", t.lower().replace("_", " ")) if w not in stop]
    qt = set(norm(str(query)))
    q = " ".join(norm(str(query)))
    scored = []
    for e in anatomy_kb():
        best = 0.0
        for n in [e["id"]] + list(e.get("aliases", [])):
            nt = norm(n)
            if not nt:
                continue
            if " ".join(nt) == q:
                best = max(best, 2.0)
            best = max(best, sum(w in qt for w in nt) / len(nt))
        if best >= 0.6:
            scored.append((best, e))
    scored.sort(key=lambda t: -t[0])
    if not scored:
        return f"anatomy '{query}': no entry (index covers: " + ", ".join(e["id"] for e in anatomy_kb()) + ")"
    out = []
    for _, e in scored[:k]:
        rel = "; ".join(f"{d}: {', '.join(v)}" for d, v in (e.get("relations") or {}).items() if v)
        out.append(f"[{e['id']}] {e.get('definition', '')} | level: {e.get('vertebral_level') or '-'} | relations (patient frame): {rel} | "
                   f"on axial CT: {e.get('axial_ct_identification', '')} | locate from segmented structures: "
                   f"{e.get('localize_from_segmented', '')}" + (f" | variants: {'; '.join(e['variants'])}" if e.get("variants") else "")
                   + f" | sources: {'; '.join(sorted({x['title'] for x in e.get('sources', [])}))}")
    return f"anatomy '{query}':\n" + "\n".join(out)


@lru_cache(1)
def subtype_answer_type() -> dict:
    tree = ast.parse(METRICS_PY.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == "SUBTYPE_ANSWER_TYPE":
                return ast.literal_eval(node.value)
    return {}


def patient_lines(item: dict) -> str:
    """Exactly the patient / spacing information of the benchmark's own prompt (data/prompt.py)."""
    def ok(v):      # the benchmark's records use null for missing values; our jsonl copy has NaN -> treat as missing
        return v is not None and v == v and str(v).strip().lower() not in ("", "nan", "none")
    parts = [f"{k}={item[k]}" for k in ("sex", "age", "scanner", "contrast") if ok(item.get(k))]
    out = [f"Patient: {', '.join(parts)}."] if parts else []
    if subtype_answer_type().get(item.get("question_subtype", "")) == "numeric" and item.get("spacing"):
        out.append(f"Voxel spacing (mm): {item['spacing']}.  Volume shape: {item.get('shape')}.")
    return "\n".join(out)


ALIASES = {"left kidney": "kidney_left", "right kidney": "kidney_right", "ivc": "inferior_vena_cava",
           "inferior vena cava": "inferior_vena_cava", "portal vein": "portal_vein_and_splenic_vein",
           "splenic vein": "portal_vein_and_splenic_vein", "gall bladder": "gallbladder", "small intestine": "small_bowel",
           "left adrenal": "adrenal_gland_left", "right adrenal": "adrenal_gland_right"}
GROUPS = [("abdominal organs", ("liver", "spleen", "pancreas", "kidney", "gallbladder", "stomach", "duodenum", "small_bowel",
                                "colon", "esophagus", "adrenal", "urinary_bladder", "prostate")),
          ("vessels", ("aorta", "vena", "portal", "iliac", "pulmonary", "brachio", "subclavian", "carotid", "atrial")),
          ("chest", ("lung", "heart", "trachea", "thyroid")),
          ("bones / muscles", ("vertebrae", "rib_", "sternum", "costal", "sacrum", "hip_", "femur", "humerus", "scapula",
                               "clavicula", "autochthon", "iliopsoas", "gluteus", "spinal"))]


def _range_compress(names: list[str]) -> list[str]:
    """vertebrae_T10, vertebrae_T11 ... -> 'vertebrae_T10 ... vertebrae_L5'; ribs likewise per side."""
    out, rest = [], []
    lvl = {f"{p}{i}": k for k, (p, i) in enumerate([("C", i) for i in range(1, 8)] + [("T", i) for i in range(1, 13)]
                                                    + [("L", i) for i in range(1, 6)] + [("S", 1)])}
    for pref in ("vertebrae_", "rib_left_", "rib_right_"):
        grp = [n for n in names if n.startswith(pref)]
        grp.sort(key=lambda n: lvl.get(n.split("_")[-1], 99) if pref == "vertebrae_" else int(n.split("_")[-1]))
        if len(grp) > 2:
            out.append(f"{grp[0]} ... {grp[-1]}")
        else:
            out += grp
    rest = [n for n in names if not n.startswith(("vertebrae_", "rib_left_", "rib_right_"))]
    return rest + out


NAV5_V2 = NAV5.replace("""{conv}

RENDERINGS SO FAR FOR THIS REQUEST""", """{conv}
{directions}

ANATOMY LOOKUPS FOR THIS REQUEST (from the cited anatomy index; anatomy only): {lookups}

RENDERINGS SO FAR FOR THIS REQUEST""").replace("""TOOLS (one per step):
""", """TOOLS (one per step):
- {{"do": "anatomy", "query": "<structure, e.g. superior mesenteric artery, pancreatic head, Couinaud segments>"}}   look up a
    cited anatomy entry (definition, vertebral level, relations, axial CT appearance, how to locate it from segmented
    structures, variants); does not use up a rendering; at most {max_lookups} lookups per request. Use it before locating a
    structure that is not segmented.
""")
_A = """priors - verify them on the images. Judge the images yourself."""
assert VLM5.count(_A) == 1
VLM5_V2 = VLM5.replace(_A, """priors - verify them on the images. Judge the images yourself. Left / right in the navigator's
   explanations always mean the PATIENT's left / right (the image side is added where helpful).""")
for _a in ("""{conv}

RENDERINGS SO FAR FOR THIS REQUEST""", """TOOLS (one per step):
"""):
    assert NAV5.count(_a) == 1


# v3 (2026-11-09, user): turn 0 = question + per-lesion tool summary; the first image set is a deterministic overview of
# the organs the VLM names (no navigator); later view requests go to the navigator; counting / comparison rule.
_A1 = """- segmentation-tool lesions of this scan: {lesions}"""
_A2 = """Segmentation tool limits: lesion classes are ONLY liver tumors and kidney cysts (no pancreatic, renal solid or colon
tumor class); organ outlines and Couinaud segments can be imperfect; it can miss lesions or over-call them."""
_A3 = """TOOLS
You can call several of the non-image tools (measure, compute, box, knowledge) in one turn; at most ONE "view" per turn."""
_A4 = """ "calls": [{{"tool": "view", "request": "<what you need to see and why, in your own words>"}},"""
for _a in (_A1, _A2, _A3, _A4):
    assert VLM5_V2.count(_a) == 1, _a[:40]
VLM5_V3 = VLM5_V2.replace(_A1, """- segmentation-tool lesions of this scan (automatic; one line per separate lesion component):
{lesions}""").replace(_A2, """Segmentation tool facts: its lesion classes are ONLY liver tumors and kidney cysts (no pancreatic, renal solid or colon
tumor class); organ outlines and Couinaud segments can be imperfect; like any tool it can have false positives and misses.
COUNTING / COMPARING / SIZING lesions of the classes the tool covers: start from the tool lesion list above. Count an
additional lesion only if you box it and the code measurement is reliable and lies in the expected organ; drop a tool
lesion only if a code measurement shows it is not a lesion (e.g. it lies in a vessel or bowel, or has air / fat HU).
A visual impression alone does not change the count.""").replace(_A3, """TOOLS
At turn 0 you have no images: request an "overview" of the organ(s) the question is about (at most 2 names from the
vocabulary); it returns 8 axial crops evenly spaced over each organ + 1 coronal slice. Later, use "view" for anything more
specific. You can call several of the non-image tools (measure, compute, box, knowledge) in one turn; at most ONE
"overview" or "view" per turn.""").replace(_A4, """ "calls": [{{"tool": "overview", "organs": ["<name from the vocabulary>", ...]}},
           {{"tool": "view", "request": "<what you need to see and why, in your own words>"}},""")
# v3b (2026-11-10, after the S1 lesion-type regression): two general clarifications on what the tool / box results imply
_B1 = "tumor class); organ outlines and Couinaud segments can be imperfect; like any tool it can have false positives and misses."
assert VLM5_V3.count(_B1) == 1
VLM5_V3B = VLM5_V3.replace(_B1, _B1 + """
For anything the tool does not cover (lesion classes it lacks, or structures not in the vocabulary), the absence of a
tool result is NOT evidence that it is absent - judge it from the images. Use a box measurement's HU to characterise a lesion (e.g. cyst vs solid
tumor, attenuation class) only when it is reliable and its volume / diameter / slice range match the object you see; a
failed, UNRELIABLE or mismatching measurement says nothing about the lesion's density - then judge from the images and
the tool lesions' own HU.""")
_B4 = """4. box: mark something you see on an AXIAL panel of an image set you currently see (panel id such as S1P2 + [left, top,
   right, bottom] as fractions 0-1 of that panel). Code segments the boxed object around the box centre (region growing
   on HU) and returns its volume, max diameter, mean HU vs its surroundings, slice range, the structure it mostly lies in,
   touching structures, and whether the measurement is reliable. It gets an id F# that you can measure, zoom on or refer
   to. A FAILED or UNRELIABLE box measurement means the box probably missed the object; it is NOT evidence of absence."""
assert VLM5_V3.count(_B4) == 1
# v4 (2026-11-10, user decision): the box tool uses a promptable 3D segmentation model (nnInteractive) + a code-side
# reliability check; it is meant for objects the tool lesion list does not cover, at most MAX_SEG boxes per question
VLM5_V4 = VLM5_V3.replace(_B4, """4. box: mark a possible lesion you see that the tool lesion list does not cover (or that you need to measure) on an AXIAL
   panel of an image set you currently see (panel id such as S1P2 + [left, top, right, bottom] as fractions 0-1 of that
   panel). A promptable 3D segmentation model (nnInteractive) outlines the object in your box; code returns its volume,
   max diameter, mean HU vs the surrounding organ, slice range, the structure it mostly lies in, touching structures and
   a reliability check. It gets an id F# that you can measure, zoom on or refer to. The model outputs a region for ANY
   box, even one with no lesion in it, and it can be wrong; the code check marks the result UNRELIABLE when the region is
   not distinct from the surrounding organ, covers most of the organ, or is much smaller than your box. Only a RELIABLE
   result is evidence that a separate lesion exists; do not count an UNRELIABLE one as a lesion or use its size (it is
   not evidence of absence either). Draw the box tightly around the object's borders on that slice; a loose box gives a
   wrong outline. At most {max_seg} boxes per question - use them for objects you think are lesions.""")
MAX_SEG = 3

# Ablation "self" (2026-11-12, user: queue the no-navi ablation): NO navigator - the VLM picks the views itself with the SAME
# rendering rules the navigator uses (one rule per view call; rendered as-is, no parsing / self-check / explanation by a
# navigator). Everything else equals NV5v3b, including the code-computed panel contents. Built from VLM5_V3B by replacing
# only the view-tool paragraph and the view-call example.
_S1 = VLM5_V3B[VLM5_V3B.index("1. view: a navigator assistant"):VLM5_V3B.index("   With every new image set you also get")]
assert VLM5_V3B.count(_S1) == 1
_S2 = """ {{"tool": "view", "request": "<what you need to see and why, in your own words>"}},"""
assert VLM5_V3B.count(_S2) == 1
# v3c (2026-11-14, user idea): the navigator as a GROUNDED view curator. Per candidate rendering it gets the code-computed
# content of every panel (structures + share, tool lesions in view; for axial panels also where each structure lies in the
# image). It must describe renderings ONLY from these code facts (never from its own reading of the pixels, never
# answering the question) and flag risks that the images could mislead the radiologist; the flags are passed on.
NAV5_V3C = NAV5_V2.replace("""Your job: work out which anatomy the request refers to, render images that show it as clearly and unambiguously as
possible, check the renderings yourself, adjust them if needed, and finally hand over the best image set with an
explanation.""", """Your job: work out which anatomy the request refers to, render images that show it as clearly and unambiguously as
possible, CHECK each rendering against its code-computed CONTENT / POSITIONS (computed from the segmentation-based
anatomical graph: which structures are in each panel, what share of the panel, where in the image; tool lesions in view),
adjust it if needed, and hand over the best image set with an explanation that is GROUNDED in those code facts. You are a
curator of views, not a reader: your descriptions must come from CONTENT / POSITIONS, not from your impression of the
pixels, and you never answer or hint at the answer to the question.""").replace(""""what_it_shows": "<for each panel: plane, slices, field of view, which
   structures / L# / F# are in it>",""", """"what_it_shows": "<for each panel, ONLY from its CONTENT / POSITIONS: plane, slices, which
   structures / L# / F# are in it with their share and image position; say explicitly if the requested anatomy is absent
   from a panel>", "risks": "<how these images could MISLEAD the radiologist, from the code facts: requested target small
   (< 2% of a panel) or absent in some panels, located only approximately (not segmented), cut off by the crop, tool-lesion
   overlay that may be wrong, panels without any target; write 'none' if none>",""").replace("""Before finishing, check on the renderings: is the requested anatomy actually shown, clear, not cut off and not hidden by
overlays;""", """Before finishing, check on the renderings, using CONTENT / POSITIONS: is the requested anatomy actually in the panels
(which share, where), clear, not cut off and not hidden by overlays;""")

VLM5_V3B_SELF = VLM5_V3B.replace(_S1, """1. view: render new images yourself with ONE rendering rule per call (there is no assistant: you choose the rule and its
   arguments; names must come from the STRUCTURE VOCABULARY, or be a tool lesion L# / boxed finding F#):
   - "overview", {{"organ": "<name>"}}: 8 axial crops over the organ + a coronal slice with these levels marked
   - "organ", {{"organ": "<name>", "window": "narrow|soft_tissue"}}: 3 thin slices at its largest cross-section
   - "relation", {{"a": "<name|L#|F#>", "b": "<name|L#|F#>"}}: slices where a and b come closest, both outlined
   - "finding", {{"id": "<L#|F#>", "fov_mm": 40-80}}: zoom on the finding, adjacent slices + full-slice locator
   - "location", {{"target": "<name|L#|F#>"}}: labelled anatomy map (Couinaud segments / structures) + locator
   - "extent", {{"target": "<name>", "z_from": <int>, "z_to": <int>}}: evenly spaced slices over a range, outlined
   - "spec", {{"target": "<name|L#|F#|full>", "plane": "axial|coronal|sagittal", "slices": "center|range|thick", "z": <int>,
     "z_from": <int>, "z_to": <int>, "thick_mm": <n>, "fov_mm": <n>, "outline_structures": ["<names>"],
     "overlay": ["tool_lesions", "anatomy_map"], "locator": true|false, "window": "soft_tissue|narrow|vessel|bone"}}:
     any custom view (for structures that are not segmented: centre a zoomed view on a nearby segmented structure at the
     slice where you expect them)
""").replace(_S2, """ {{"tool": "view", "rule": "<rule>", "args": {{<arguments of that rule>}}}},""")


def _nav_json(text: str) -> dict:
    """Navigator reply -> dict: extract_json first (unchanged behaviour for every reply it parses); if that finds no "next",
    decode the first JSON object and ignore trailing text (e.g. a stray closing brace) (2026-11-13 fix)."""
    j = extract_json(text) or {}
    j = j if isinstance(j, dict) else {}
    if isinstance(j.get("next"), dict):
        return j
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    i = t.find("{")
    if i >= 0:
        try:
            k, _ = json.JSONDecoder().raw_decode(t[i:])
            if isinstance(k, dict) and isinstance(k.get("next"), dict):
                return k
        except Exception:
            pass
    return j


class Navi5Agent:
    def __init__(self, vlm: LLM, nav: LLM, navi_mode: str = "vision", max_turns: int = MAX_TURNS, max_renders: int = MAX_RENDERS,
                 anatomy: bool = False, max_lookups: int = 3, v3: bool = False, v4: bool = False, v3b: bool = False,
                 slice_pct: bool = False, v3c: bool = False, nomem_nav: bool = False, nomem_vlm: bool = False,
                 no_overlay: bool = False):
        """max_renders: renderings per view request (3 for the first S0 run NV5v_r1; 2 from 2026-11-08 on, user decision).
        anatomy: v2 prompts - navigator anatomy lookup tool + explicit direction definitions (VLM and navigator)."""
        self.vlm, self.nav, self.mode, self.max_turns, self.max_renders = vlm, nav, navi_mode, max_turns, max_renders
        self.anatomy, self.max_lookups, self.v3, self.v4, self.v3b = anatomy, max_lookups, v3 or v4 or v3b or v3c, v4, v3b or v3c
        # memory ablations (2026-11-14, user): nomem_nav = the navigator gets no memory of the image sets already shown and no
        # coverage summary; nomem_vlm = the VLM gets no log of its earlier turns and no ledger (it still sees the image sets
        # currently attached and the tool results of this turn). Templates unchanged, only these fields are emptied.
        self.nomem_nav, self.nomem_vlm = nomem_nav, nomem_vlm
        self.v3c = v3c      # navigator only: grounded view curator (NAV5_V3C + POSITIONS + risks); VLM prompt = v3b
        # slice_pct: when the QUESTION TEXT mentions slices, add the mapping from our z index to the slice position (%)
        # of the original benchmark volume (our CT is a crop of it); offset estimated without ground truth (slicepct "ts")
        self.slice_pct = slice_pct
        self.r = Navi2Agent(nav, vlm)
        self.r.no_overlay = no_overlay      # 2026-11-15: images without outlines / contours / boxes (text content unchanged)

    # ------------------------------------------------------------------ vocabulary / names
    def _vocab(self, case) -> tuple[str, set]:
        names = [n for n in case.names if n not in LESION_ORGAN_TS]
        left = list(names)
        lines = []
        for title, keys in GROUPS:
            grp = [n for n in left if n.startswith(keys) or any(k in n for k in keys)]
            left = [n for n in left if n not in grp]
            if grp:
                lines.append(f"  {title}: " + ", ".join(_range_compress(grp)))
        if left:
            lines.append("  other: " + ", ".join(left))
        return "\n".join(lines), set(names)

    def _resolve(self, name: str, st: dict) -> str | None:
        n = str(name).strip()
        if re.fullmatch(r"[LF]\d+", n):
            return n if (n in st["L"] or n in st["F"]) else None
        k = n.lower().replace("-", " ").strip()
        m = re.fullmatch(r"(?:couinaud |liver )?segment (\d)", k)
        if m:
            k = f"liver_segment_{m.group(1)}"
        k = ALIASES.get(k, k).replace(" ", "_")
        segs = set(st["case"].layers["liver_segments"][1]) if "liver_segments" in st["case"].layers else set()
        return k if (k in st["vocab"] or k in segs) else None

    # ------------------------------------------------------------------ VLM's own tools
    def _value(self, st, tree, fn: str, name: str) -> float:
        t = self._resolve(name, st)
        if t is None:
            raise ValueError(f"unknown name {name}")
        if t in st["F"]:
            v = st["F"][t]["meas"].get({"volume": "vol_ml", "mean_hu": "mean_hu", "max_diameter": "max_diam_mm"}[fn])
            if v is None:
                raise ValueError(f"{t} has no {fn} (measurement failed)")
            return float(v)
        n = st["L"][t] if t in st["L"] else t
        nd = tree.g["nodes"].get(n)
        if nd is None:
            raise ValueError(f"no measurement for {name}")
        return float({"volume": nd["volume_ml"], "mean_hu": nd["hu_mean"], "max_diameter": nd["max_diameter_mm"]}[fn])

    def _compute(self, st, tree, expr: str) -> str:
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("volume", "mean_hu", "max_diameter")
                    and len(node.args) == 1):
                a = node.args[0]
                nm = a.id if isinstance(a, ast.Name) else a.value if isinstance(a, ast.Constant) else None
                return self._value(st, tree, node.func.id, str(nm))
            raise ValueError("unsupported expression (use volume(x), mean_hu(x), max_diameter(x), numbers and + - * /)")
        try:
            return f"compute {expr} = {ev(ast.parse(str(expr), mode='eval')):.4g}"
        except Exception as e:
            return f"compute {expr}: error: {e}"

    def _measure(self, st, tree, target: str) -> str:
        t = self._resolve(target, st)
        if t is None:
            return f"measure {target}: unknown name; use a name from the STRUCTURE VOCABULARY, or box it and use the F#"
        return self.r._measure(tree, {"L": st["L"], "F": {k: {"text": v["text"]} for k, v in st["F"].items()}}, [t])

    def _box(self, st, tree, call: dict) -> str:
        pid = str(call.get("panel"))
        if pid not in st["panels"]:
            return f"box {pid}: unknown panel (use an axial panel id of an image set you currently see)"
        try:
            frac = [float(v) for v in call.get("box")]
            assert len(frac) == 4
        except Exception:
            return f"box {pid}: invalid box (need [left, top, right, bottom] fractions)"
        z, crop = st["panels"][pid]
        vb = V.crop_frac_to_box(crop, frac)
        if self.v4:
            if st.get("n_seg", 0) >= MAX_SEG:
                return f"box {pid}: limit of {MAX_SEG} boxes per question reached"
            st["n_seg"] = st.get("n_seg", 0) + 1
            meas = NNI.measure_nni(tree.case, z, vb)
            desc = CAND.describe_measure(meas).replace("code measurement", "nnInteractive segmentation", 1)
            if meas.get("reliable"):
                desc = desc.replace("(approximate)", "(RELIABLE: distinct from the surrounding organ)", 1)
        else:
            try:
                meas = CAND.measure_local(tree.case, z, vb)
            except Exception as e:
                meas = {"reliable": False, "reason": type(e).__name__, "seed_hu": float("nan"), "surround_hu": float("nan")}
            desc = CAND.describe_measure(meas)
        fid = f"F{len(st['F']) + 1}"
        st["F"][fid] = {"z": z, "box": vb, "inside": meas.get("inside"), "meas": meas,
                        "text": f"'{str(call.get('label', ''))[:40]}' boxed on {pid} (z={z}); " + desc}
        return f"box -> {fid}: {st['F'][fid]['text']}"

    def _knowledge(self, query: str) -> str:
        hits = kb_lookup(str(query), 3)
        return f"knowledge '{query}': " + (" | ".join(hits) if hits else "no matching entry")

    # ------------------------------------------------------------------ image content (code-computed)
    @staticmethod
    def _positions(tree, panels) -> str:
        """v3c: where each structure (>= 1% of the panel) lies in each AXIAL panel, in IMAGE terms (image left = patient
        right, image top = anterior), from the segmentation."""
        c = tree.case
        out = []
        for pid, (z, crop) in list(panels.items())[:12]:
            if crop is None:
                continue
            x0, x1, y0, y1 = crop
            lab = c.labels[x0:x1 + 1, y0:y1 + 1, z]
            ids, cnt = np.unique(lab[lab > 0], return_counts=True)
            items = []
            for i, n in sorted(zip(ids, cnt), key=lambda t: -t[1]):
                if n / lab.size < 0.01:
                    continue
                xs, ys = np.nonzero(lab == i)
                col = 1 - xs.mean() / max(lab.shape[0] - 1, 1)     # image column fraction (x reversed)
                row = 1 - ys.mean() / max(lab.shape[1] - 1, 1)     # image row fraction (y reversed: anterior on top)
                h = "left" if col < 0.33 else "right" if col > 0.67 else "centre"
                v = "top" if row < 0.33 else "bottom" if row > 0.67 else "middle"
                items.append(f"{c.names[i - 1]} {v}-{h}")
            out.append(f"{pid} z={z}: " + (", ".join(items[:8]) or "no segmented structure"))
        return "; ".join(out)

    def _panel_content(self, tree, st, z: int, crop) -> str:
        c = tree.case
        x0, x1, y0, y1 = crop
        lab = c.labels[x0:x1 + 1, y0:y1 + 1, z]
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        tot = lab.size
        parts = [f"{c.names[i - 1]} {n / tot * 100:.0f}%" for i, n in sorted(zip(ids, cnt), key=lambda t: -t[1]) if n / tot >= 0.01]
        txt = ", ".join(parts) or "no segmented structure >= 1%"
        if "liver_segments" in c.layers and "liver" in c.names:
            arr, names = c.layers["liver_segments"]
            liv = lab == c.name_to_id("liver")
            v = arr[x0:x1 + 1, y0:y1 + 1, z][liv]
            v = v[v > 0]
            if len(v):
                bc = np.bincount(v, minlength=len(names) + 1)[1:]
                txt += "; liver segments: " + ", ".join(f"{names[i].split('_')[-1]} {k / bc.sum() * 100:.0f}%"
                                                         for i, k in sorted(enumerate(bc), key=lambda t: -t[1]) if k / bc.sum() >= 0.05)
        inv = self._in_view(tree, st, crop, z, z)
        return txt + (f"; in view: {inv}" if inv else "")

    @staticmethod
    def _in_view(tree, st, crop, z0, z1) -> str:
        x0, x1, y0, y1 = crop
        hits = []
        for lid, n in st["L"].items():
            b = tree.g["nodes"][n]["bbox_vox"]
            if b[5] >= z0 and b[4] <= z1 and b[1] >= x0 and b[0] <= x1 and b[3] >= y0 and b[2] <= y1:
                hits.append(lid)
        for fid, f in st["F"].items():
            b = f["box"]
            if z0 <= f["z"] <= z1 and b[1] >= x0 and b[0] <= x1 and b[3] >= y0 and b[2] <= y1:
                hits.append(fid)
        return ", ".join(hits)

    # ------------------------------------------------------------------ navigator: rules -> specs
    def _closest_z(self, tree, st, a: str, b: str):
        """z of the closest approach of a and b (graph edge for two segmented structures; else mask-based)."""
        g, sp = tree.g, tree.sp
        for e in g["edges"]:
            if {e["a"], e["b"]} == {a, b} and e.get("closest_a_mm"):
                pa, pb = np.array(e["closest_a_mm"]), np.array(e["closest_b_mm"])
                return int(round((pa[2] + pb[2]) / 2 / sp[2])), float(e["min_dist_mm"])
        ma, ba, za = self.r._target(tree, st, a)[:3]
        mb, bb, zb = self.r._target(tree, st, b)[:3]
        if za is None or zb is None:
            return None, None
        if ma is not None and mb is not None:
            both = np.nonzero(ma.any((0, 1)) & mb.any((0, 1)))[0]
            if len(both):
                areas = [min(ma[:, :, z].sum(), mb[:, :, z].sum()) for z in both]
                return int(both[int(np.argmax(areas))]), None
        return int(round((za + zb) / 2)), None

    def _rule_specs(self, tree, st, rule: str, args: dict):
        """-> (list of view specs, extra images list, note) for a rule; raises ValueError with a message."""
        args = args if isinstance(args, dict) else {}
        res = lambda k: self._resolve(args.get(k, ""), st)
        if rule == "overview":
            o = res("organ")
            if o is None:
                raise ValueError("overview needs a segmented organ name")
            m = self.r._target(tree, st, o)[0]
            zz = np.nonzero(m.any((0, 1)))[0]
            zs = np.linspace(zz.min(), zz.max(), 8).round().astype(int).tolist()[::-1]
            yc = int(np.argwhere(m)[:, 1].mean())
            cor = V.coronal_with_levels(tree.case, yc, zs, 512)
            return [{"target": o, "slices": "range", "z_from": int(zz.min()), "z_to": int(zz.max()), "n_slices": 8,
                     "window": "soft_tissue"}], [cor], f"overview of {o}: 8 axial crops z={zs} + coronal slice with these levels marked"
        if rule == "organ":
            o = res("organ")
            if o is None:
                raise ValueError("organ needs a segmented organ name")
            w = args.get("window") if args.get("window") in ("narrow", "soft_tissue") else "narrow"
            return [{"target": o, "slices": "center", "window": w, "outline_structures": [o]}], [], f"{o}: 3 thin slices at its largest cross-section"
        if rule == "relation":
            a, b = res("a"), res("b")
            if a is None or b is None:
                raise ValueError("relation needs two names / L# / F#")
            z, dist = self._closest_z(tree, st, a, b)
            if z is None:
                raise ValueError("cannot locate both")
            ba, bb = self.r._target(tree, st, a)[1], self.r._target(tree, st, b)[1]
            union = [min(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), max(ba[3], bb[3])]
            crop = V.pad_box(union, tree.case.shape, 15, tree.sp, 60)
            ol = [x for x in (a, b) if not re.fullmatch(r"[LF]\d+", x)]
            spec = {"target": a, "slices": "center", "z": z, "crop_box": crop, "outline_structures": ol,
                    "overlay": ["tool_lesions"] if any(re.fullmatch(r"L\d+", x) for x in (a, b)) else []}
            note = f"relation {a} - {b}: closest approach around z={z}" + (f" (surface distance {dist:.1f} mm)" if dist is not None else "")
            return [spec], [], note
        if rule == "finding":
            i = res("id")
            if i is None or not re.fullmatch(r"[LF]\d+", i):
                raise ValueError("finding needs an L# or F#")
            fov = args.get("fov_mm") if isinstance(args.get("fov_mm"), (int, float)) else 60
            return [{"target": i, "slices": "center", "fov_mm": float(fov), "locator": True}], [], f"zoom on {i} (fov {fov} mm)"
        if rule == "location":
            t = res("target")
            if t is None:
                raise ValueError("location needs a name / L# / F#")
            return [{"target": t, "slices": "center", "overlay": ["anatomy_map"], "locator": True}], [], f"location of {t}: anatomy map + locator"
        if rule == "extent":
            t = res("target")
            if t is None:
                raise ValueError("extent needs a segmented name")
            spec = {"target": t, "slices": "range", "outline_structures": [t]}
            for k in ("plane", "window"):      # 2026-11-14 fix 7: plane / window used to be ignored by "extent"
                if args.get(k) in (("axial", "coronal", "sagittal") if k == "plane" else ("narrow", "soft_tissue")):
                    spec[k] = args[k]
            for k in ("z_from", "z_to"):
                if isinstance(args.get(k), int):
                    spec[k] = args[k]
            return [spec], [], f"extent of {t}: evenly spaced slices"
        if rule == "spec":
            keep = ("target", "plane", "slices", "z", "z_from", "z_to", "thick_mm", "fov_mm", "margin_mm", "overlay",
                    "locator", "window", "outline_structures")
            spec = {k: v for k, v in args.items() if k in keep}
            t = str(spec.get("target", "full"))
            if t != "full":
                rt = self._resolve(t, st)
                if rt is None:      # 2026-11-13 fix: a list of names -> the first name that resolves
                    rt = next((x for x in (self._resolve(p_, st) for p_ in re.split(r",|\band\b|/|&", t)) if x), None)
                if rt is None:
                    raise ValueError(f"unknown target {t} (use a segmented name, L#, F# or full)")
                spec["target"] = rt
            spec["outline_structures"] = [x for x in (self._resolve(n, st) for n in spec.get("outline_structures") or []) if x]
            return [spec], [], "custom view"
        raise ValueError(f"unknown rule {rule}")

    def _render_set(self, tree, st, specs, extra, prefix: str):
        """Render specs with panel ids '<prefix>P#' -> (images, panels, description)."""
        ims, panels, desc = [], {}, []
        for k, sp_ in enumerate(specs):
            sp_ = dict(sp_, panel_prefix=prefix if len(specs) == 1 else f"{prefix}{chr(97 + k)}")
            out, receipt, pn = self.r._render(tree, st, sp_)
            if out is None:
                raise ValueError(receipt)
            ims += out
            panels.update({p: v for p, v in (pn or {}).items()})
            desc.append(receipt)
        ims += extra
        content = "; ".join(f"{p} z={z}: {self._panel_content(tree, st, z, cr)}" for p, (z, cr) in panels.items())
        return ims, panels, " ".join(desc), content

    def _overlap(self, st, specs, panels) -> str:
        """Code-computed overlap of a candidate with the image sets already shown (same plane + window)."""
        best = None
        if not panels:      # 2026-11-13 fix: coronal / sagittal renderings (no axial panels) used to crash here
            return "overlap with earlier image sets not computed (non-axial view)"
        zs = {z for z, _ in panels.values()}
        for rec in st["shown"]:
            if rec.get("crop") is None:
                continue
            for sp_ in rec["specs"]:
                for c_sp in specs:
                    if (sp_.get("plane") or "axial") != (c_sp.get("plane") or "axial") or \
                            (sp_.get("window") or "soft_tissue") != (c_sp.get("window") or "soft_tissue"):
                        continue
                    zo = len(zs & rec["zs"]) / max(len(zs), 1)
                    a, b = next(iter(panels.values()))[1], rec["crop"]
                    ix = max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[2], b[2]))
                    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - ix
                    iou = ix / max(ua, 1)
                    if best is None or zo + iou > best[0] + best[1]:
                        best = (zo, iou, rec["turn"])
        return "no overlap with image sets already shown" if best is None else \
            f"overlap with the set shown at turn {best[2]}: {best[0] * 100:.0f}% of slices, field-of-view IoU {best[1]:.2f} (same plane and window)"

    # ------------------------------------------------------------------ navigator loop
    def _navigate(self, tree, st, request: str, conclusion: str, turn: int, ep: Episode):
        g = tree.g
        renders = []      # dicts: rid, rule, args, specs, extra, ims, panels, desc, content, overlap, note
        st.pop("_finish_reask", None)
        q = st["q"]
        nav_steps = []
        dups, dup_note = 0, ""
        lookups = []
        for step in range(self.max_renders + 3 + (self.max_lookups if self.anatomy else 0)):
            # 2026-11-13 fix (user): a failed rendering (unknown name etc.) does not use up the rendering budget; the step
            # loop above still caps the number of navigator calls per request
            n_left = self.max_renders - sum(1 for r_ in renders if r_["ims"])
            force = dup_note + ("" if n_left > 0 else '\nNo renderings left: your "next" MUST be "finish".')
            rtxt = "\n".join(f"{r['rid']}: rule {r['rule']} {json.dumps(r['args'])}; {r['note']}; {r['desc']} CONTENT: {r['content']}; "
                             + (f"POSITIONS: {self._positions(tree, r['panels'])}; " if self.v3c and r.get("panels") else "")
                             + f"{r['overlap']}" for r in renders) or "(none yet)"
            extra = dict(directions=DIRECTIONS, lookups="\n".join(lookups) or "(none)", max_lookups=self.max_lookups) if self.anatomy else {}
            prompt = (NAV5_V3C if self.v3c else NAV5_V2 if self.anatomy else NAV5).format(**extra, question=q["question"], options=q["options"], request=request[:600], conclusion=conclusion[:500],
                                 nz=tree.case.shape[2], sz=tree.sp[2], zranges=st["zranges"], relations=st["relations"],
                                 tool=st["tool_txt"], findings="\n".join(f"{k}: {v['text']}" for k, v in st["F"].items()) or "(none)",
                                 memory=(self._memory_text(st) if not self.nomem_nav else "") or "(none)",
                                 coverage=(self._coverage_text(st) if not self.nomem_nav else "") or "(nothing shown yet)",
                                 conv=CONV, seen_note=("images of each rendering attached in order" if self.mode == "vision"
                                                       else "you do not see the images, only the code-computed descriptions"),
                                 max_r=self.max_renders, n_left=n_left, renders=rtxt, force=force)
            imgs = [im for r in renders for im in r["ims"]] if self.mode == "vision" else []
            nmsgs = [user_msg(prompt, [R.b64png(i) for i in imgs])]
            r = self.nav.chat(nmsgs, max_tokens=1500)
            ep.cost += r["cost"]
            j = _nav_json(r["text"])
            retried = None
            if not isinstance(j.get("next"), dict):
                # 2026-11-13 fix: a navigator reply without a parsable "next" used to end the request with no image (no
                # resend, unlike the VLM). Resend once as a new sample; a reply cut at the token limit gets more room
                retried = r["text"][:300]
                r = self.nav.chat(nmsgs, max_tokens=3000 if r["out"] >= 1500 else 1500, cache_salt="retry1")
                ep.cost += r["cost"]
                j = _nav_json(r["text"])
            nx = j.get("next") if isinstance(j.get("next"), dict) else {}
            nav_steps.append({"parse": j.get("parse"), "check": j.get("check"), "next": nx,
                              **({"retried_after": retried} if retried is not None else {})})
            if self.anatomy and nx.get("do") == "anatomy":
                if len(lookups) < self.max_lookups:
                    lookups.append(anatomy_lookup(nx.get("query", "")))
                else:
                    dup_note = "\nNOTE: no anatomy lookups left for this request."
                continue
            if nx.get("do") == "render" and n_left > 0:
                rule, args = str(nx.get("rule")), nx.get("args") or {}
                try:
                    specs, extra, note = self._rule_specs(tree, st, rule, args)
                    key = json.dumps(specs, sort_keys=True, default=str)
                    dup = next((r_["rid"] for r_ in renders if r_.get("key") == key), None)
                    if dup and dups < 2:      # identical parameters: not rendered again, does not use up a rendering
                        dups += 1
                        dup_note = (f"\nNOTE: your last request had exactly the same parameters as {dup}; it was NOT rendered "
                                    f"again. Choose different parameters or finish.")
                        continue
                    dup_note = ""
                    rid = f"R{len(renders) + 1}"
                    ims, panels, desc, content = self._render_set(tree, st, specs, extra, rid)
                    ov = self._overlap(st, specs, panels)
                    for r_ in renders:
                        if r_["panels"] and panels:     # 2026-11-13 fix: non-axial renderings have no axial panels (used to crash)
                            zo = len({z for z, _ in panels.values()} & {z for z, _ in r_["panels"].values()}) / max(len(panels), 1)
                            a_, b_ = next(iter(panels.values()))[1], next(iter(r_["panels"].values()))[1]
                            ix = max(0, min(a_[1], b_[1]) - max(a_[0], b_[0])) * max(0, min(a_[3], b_[3]) - max(a_[2], b_[2]))
                            iou = ix / max((a_[1] - a_[0]) * (a_[3] - a_[2]) + (b_[1] - b_[0]) * (b_[3] - b_[2]) - ix, 1)
                            ov += f"; overlap with {r_['rid']}: {zo * 100:.0f}% of slices, field-of-view IoU {iou:.2f}"
                    renders.append(dict(rid=rid, rule=rule, args=args, specs=specs, extra_rule=(rule, args), ims=ims,
                                        panels=panels, desc=desc, content=content, overlap=ov, note=note, key=key))
                except Exception as e:
                    renders.append(dict(rid=f"R{len(renders) + 1}", rule=rule, args=args, specs=[], ims=[], panels={},
                                        desc=f"RENDER FAILED: {str(e)[:200]}", content="-", overlap="-", note="-"))
                continue
            # 2026-11-14 fix 7: the navigator asked for another rendering although none is left (24.5% of requests): this
            # used to fall through to "finish" with an EMPTY explanation; ask it once more to finish (only in this case)
            if nx.get("do") in ("render", "anatomy") and n_left <= 0 and not st.get("_finish_reask") and any(r_["ims"] for r_ in renders):
                st["_finish_reask"] = True
                dup_note = ('\nNOTE: no renderings are left and you asked for another one. Your reply MUST now be "finish": '
                            'choose among the renderings above and fill why / what_it_shows / how_to_read / localisation '
                            'describing ONLY what their code-computed CONTENT supports.')
                continue
            st.pop("_finish_reask", None)
            # finish (or anything else / no renderings left)
            choice = [c for c in (nx.get("choice") or []) if any(r_["rid"] == c and r_["ims"] for r_ in renders)][:2]
            if not choice:
                ok = [r_["rid"] for r_ in renders if r_["ims"]]
                choice = ok[-1:] if ok else []
            break
        else:       # step cap reached without "finish" (now possible: failed renderings no longer use up the budget)
            ok = [r_["rid"] for r_ in renders if r_["ims"]]
            choice = ok[-1:] if ok else []
        # final set: re-render the chosen renderings with VLM-facing panel ids S<k>
        st["n_sets"] += 1
        sk = f"S{st['n_sets']}"
        ims, panels, specs_all, desc, content, zs = [], {}, [], [], [], set()
        for i, rid in enumerate(choice):
            rr = next(r_ for r_ in renders if r_["rid"] == rid)
            specs, extra, note = self._rule_specs(tree, st, *rr["extra_rule"])
            pre = sk if len(choice) == 1 else f"{sk}{chr(97 + i)}"
            a, p, d, cnt = self._render_set(tree, st, specs, extra, pre)
            ims += a; panels.update(p); specs_all += specs; desc.append(d); content.append(cnt)
        # 2026-11-13 fix (user: every label the navigator uses must be one the VLM knows): the navigator writes about its
        # internal renderings R# / panels R#P#, the VLM only ever sees the re-labelled set S# / S#P# (S#a / S#b for two
        # renderings) -> translate; renderings that were not kept become "(a rendering not shown to you)"
        rmap = {rid: (sk if len(choice) == 1 else f"{sk}{chr(97 + i)}") for i, rid in enumerate(choice)}
        relabel = lambda t: re.sub(r"\bR(\d+)(P\d+)?\b", lambda mm: rmap[f"R{mm.group(1)}"] + (mm.group(2) or "")
                                   if f"R{mm.group(1)}" in rmap else "(a rendering not shown to you)", t)
        expl = {k: relabel(str(nx.get(k) or ""))[:700] for k in ("why", "what_it_shows", "how_to_read", "localisation")
                + (("risks",) if self.v3c else ())}
        if choice and not any(expl.values()):     # 2026-11-14 fix 7: never pass a blank explanation; say so instead
            expl["why"] = "(the navigator wrote no explanation; this is its last rendering - rely on the images and the code-computed content)"
        ep.steps.append({"step": turn, "mode": "nav", "request": request, "nav_steps": nav_steps, "lookups": lookups,
                         "renders": [{k: r_[k] for k in ("rid", "rule", "args", "note", "desc", "content", "overlap")} for r_ in renders],
                         "choice": choice, "explanation": expl})
        if not ims:
            # 2026-11-13 fix (user): a navigator that finishes by pointing to image sets ALREADY shown (S#) judged that no new
            # rendering is needed - say so instead of "could not produce images"
            prev = [c for c in (nx.get("choice") or []) if isinstance(c, str) and any(c == rec["set"] for rec in st["shown"])]
            if prev and nx.get("do") == "finish":
                return None, (f"view: the navigator judged that the image set(s) {', '.join(prev)} already shown to you answer this "
                              f"request, so no new rendering was made; its reason: {relabel(str(nx.get('why') or ''))[:400]}")
            return None, f"view: the navigator could not produce images for this request ({relabel(str(nav_steps[-1].get('check')))})"
        st["panels_by_set"][sk] = panels
        crop = next(iter(panels.values()))[1] if panels else None
        st["shown"].append({"turn": turn, "set": sk, "request": request, "localisation": expl["localisation"], "specs": specs_all,
                            "zs": {z for z, _ in panels.values()}, "crop": crop, "desc": " ".join(desc),
                            "content": "; ".join(content), "next_conclusion": None})
        text = (f"NEW IMAGES {sk} from the navigator (turn {turn}, for your request \"{request[:200]}\"; the navigator's "
                f"explanation is for reference only - judge the images yourself):\n"
                f"- why chosen: {expl['why']}\n- what it shows: {expl['what_it_shows']}\n- how to read: {expl['how_to_read']}\n"
                f"- how the anatomy was located: {expl['localisation']}\n"
                + (f"- risks the navigator flags (from the code-computed content): {expl.get('risks') or 'none stated'}\n" if self.v3c else "")
                + f"- rendering: {' '.join(desc)}\n"
                f"- code-computed content per panel: {'; '.join(content)}")
        return (sk, ims, panels, text), None

    def _overview_set(self, tree, st, organs, turn: int, ep: Episode):
        """v3: deterministic overview of the named organs (no navigator), registered like a navigator image set."""
        names = [self._resolve(o, st) for o in (organs if isinstance(organs, list) else [organs])][:2]
        names = [n for n in names if n and not re.fullmatch(r"[LF]\d+", n)]
        if not names:
            return None, "overview: no valid organ name (use names from the STRUCTURE VOCABULARY)"
        st["n_sets"] += 1
        sk = f"S{st['n_sets']}"
        ims, panels, specs_all, desc, content = [], {}, [], [], []
        for i, o in enumerate(names):
            specs, extra, note = self._rule_specs(tree, st, "overview", {"organ": o})
            a, p, d, cnt = self._render_set(tree, st, specs, extra, sk if len(names) == 1 else f"{sk}{chr(97 + i)}")
            ims += a; panels.update(p); specs_all += specs; desc.append(f"{note}. {d}"); content.append(cnt)
        ep.steps.append({"step": turn, "mode": "overview", "organs": names})
        st["panels_by_set"][sk] = panels
        crop = next(iter(panels.values()))[1] if panels else None
        st["shown"].append({"turn": turn, "set": sk, "request": f"overview of {', '.join(names)}", "localisation": "segmented",
                            "specs": specs_all, "zs": {z for z, _ in panels.values()}, "crop": crop, "desc": " ".join(desc),
                            "content": "; ".join(content), "next_conclusion": None})
        text = (f"NEW IMAGES {sk} (turn {turn}, overview of {', '.join(names)}; deterministic, no navigator):\n"
                f"- rendering: {' '.join(desc)}\n- code-computed content per panel: {'; '.join(content)}")
        return (sk, ims, panels, text), None

    def _self_view(self, tree, st, call: dict, turn: int, ep: Episode):
        """Ablation "self": render the VLM's own rule + args (no navigator), registered like a navigator image set."""
        rule, args = str(call.get("rule") or ""), call.get("args") if isinstance(call.get("args"), dict) else {}
        try:
            specs, extra, note = self._rule_specs(tree, st, rule, args)
            st["n_sets"] += 1
            sk = f"S{st['n_sets']}"
            ims, panels, desc, content = self._render_set(tree, st, specs, extra, sk)
        except Exception as e:
            ep.steps.append({"step": turn, "mode": "selfview", "rule": rule, "args": args, "error": str(e)[:200]})
            return None, f"view {rule} {json.dumps(args)[:150]}: failed ({str(e)[:200]})"
        ep.steps.append({"step": turn, "mode": "selfview", "rule": rule, "args": args, "set": sk})
        st["panels_by_set"][sk] = panels
        crop = next(iter(panels.values()))[1] if panels else None
        st["shown"].append({"turn": turn, "set": sk, "request": f"rule {rule} {json.dumps(args)[:200]}", "localisation": "rule",
                            "specs": specs, "zs": {z for z, _ in panels.values()}, "crop": crop, "desc": desc,
                            "content": content, "next_conclusion": None})
        text = (f"NEW IMAGES {sk} (turn {turn}, your rule {rule} {json.dumps(args)[:200]}):\n"
                f"- rendering: {note}. {desc}\n- code-computed content per panel: {content}")
        return (sk, ims, panels, text), None

    def _memory_text(self, st) -> str:
        return "\n".join(f"- turn {s['turn']} ({s['set']}): request \"{s['request'][:200]}\"; localisation: {s['localisation'][:200]}; "
                         f"images: {s['desc'][:300]}; content: {s['content'][:400]}; radiologist's next conclusion: "
                         f"{(s['next_conclusion'] or '(pending)')[:300]}" for s in st["shown"])

    def _coverage_text(self, st) -> str:
        rows = []
        for s in st["shown"]:
            for sp_ in s["specs"]:
                zoom = f"fov {sp_['fov_mm']:.0f} mm" if sp_.get("fov_mm") else ("explicit crop" if sp_.get("crop_box") else "unzoomed")
                rows.append(f"{sp_.get('target')} | {sp_.get('plane') or 'axial'} | {sp_.get('window') or 'soft_tissue'} | {zoom} | "
                            f"z {min(s['zs']) if s['zs'] else '-'}-{max(s['zs']) if s['zs'] else '-'} ({len(s['zs'])} slices) [{s['set']}]")
        return "; ".join(rows)

    # ------------------------------------------------------------------ main loop
    def run(self, case, g, item: dict) -> Episode:
        les_organs = sorted({o for l, o in LESION_ORGAN_TS.items() if any(n.startswith(l + "#") for n in g["nodes"])})
        tree = Tree(case, g, les_organs)
        ep = Episode(item)
        vocab_txt, vocab = self._vocab(case)
        st = {"case": case, "vocab": vocab, "L": self.r._tool_lesions(tree), "F": {}, "panels_by_set": {}, "shown": [], "n_sets": 0}
        st["tool_txt"] = "\n".join(self.r._lesion_line(tree, k, n) for k, n in st["L"].items()) or "(none)"
        sp = tree.sp
        st["zranges"] = "; ".join(f"{n} {g['nodes'][n]['bbox_vox'][4]}-{g['nodes'][n]['bbox_vox'][5]}" for n in case.names
                                  if n in g["nodes"] and n not in LESION_ORGAN_TS)
        if "liver_segments" in case.layers:
            st["zranges"] += "; " + "; ".join(f"{n} {g['nodes'][n]['bbox_vox'][4]}-{g['nodes'][n]['bbox_vox'][5]}"
                                              for n in case.layers["liver_segments"][1] if n in g["nodes"])
        skip = ("rib_", "vertebrae_", "costal", "sternum", "hip_", "femur", "autochthon", "iliopsoas", "gluteus", "spinal",
                "sacrum", "lung_", "heart", "humerus", "scapula", "clavicula")
        rel = []
        for e in g["edges"]:
            if e.get("type", "adjacent") != "adjacent" or e["min_dist_mm"] > 5 or "#" in e["a"] + e["b"]:
                continue
            if e["a"].startswith(skip) or e["b"].startswith(skip) or e["a"] in LESION_ORGAN_TS or e["b"] in LESION_ORGAN_TS:
                continue
            zc = int(round((e["closest_a_mm"][2] + e["closest_b_mm"][2]) / 2 / sp[2])) if e.get("closest_a_mm") else None
            gap = "touching" if e["min_dist_mm"] < 1 else f"gap {e['min_dist_mm']:.0f} mm"
            rel.append(f"{e['a']}-{e['b']}: {gap}" + (f" @z{zc}" if zc is not None else ""))
        st["relations"] = "; ".join(rel)
        st["q"] = dict(question=question_stem(item["mc_question"]), options=json.dumps(parse_options(item["mc_question"])))
        segs = ", ".join(case.layers["liver_segments"][1]) if "liver_segments" in case.layers else "(none)"
        if self.v3:      # per-lesion tool summary (organ / side, segment, volume, diameter, HU, slices)
            lesions = "\n".join(f"  {self.r._lesion_line(tree, k, n)}, mean HU {g['nodes'][n]['hu_mean']:.0f}"
                                 for k, n in st["L"].items()) or "  none"
        else:
            lesions = ", ".join(f"{k} ({n.split('#')[0]})" for k, n in st["L"].items()) or "none"
        slice_line = ""
        if self.slice_pct and re.search(r"\bslice", item["mc_question"], re.I):
            d = slicepct.describe(case.case_id, case.shape[2] - 1, source="ts")
            if d:
                slice_line = "\n" + d + " (z = the slice index used by all tools and image labels here)"
        log, ledger, final, current = [], "(empty)", None, []     # current: list of (sk, ims, panels, text)
        for turn in range(self.max_turns):
            left = self.max_turns - turn
            shown = current[-KEEP_SETS:]
            st["panels"] = {p: v for s in shown for p, v in s[2].items()}
            imgs = [im for s in shown for im in s[1]]
            idesc = " | ".join(f"set {s[0]} (images {sum(len(x[1]) for x in shown[:i]) + 1}-{sum(len(x[1]) for x in shown[:i + 1])}): "
                               f"{s[3]}" for i, s in enumerate(shown)) or "(none yet)"
            force = '\nNo turns left after this one: set "final": true.' if left == 1 else ""
            vis = {s[0] for s in shown}
            logtxt = ("\n".join(log) if not self.nomem_vlm else "") or "(none)"
            for s in current:       # sets still visible: described under IMAGES; older sets: their description stays in the log
                logtxt = logtxt.replace("{SETDESC:" + s[0] + "}", "described under IMAGES YOU CURRENTLY SEE" if s[0] in vis
                                        else "no longer attached as images; its description:\n  " + s[3])
            prompt = (VLM5_V4 if self.v4 else VLM5_V3B_SELF if (self.v3b and self.mode == "self") else VLM5_V3B if self.v3b else VLM5_V3 if self.v3 else VLM5_V2 if self.anatomy else VLM5).format(max_seg=MAX_SEG, conv=CONV, patient=patient_lines(item) + slice_line, vocab=vocab_txt, segments=segs, lesions=lesions,
                                 boxed=", ".join(st["F"]) or "(none yet)", images=idesc, log=logtxt,
                                 ledger=(ledger if not self.nomem_vlm else "(none)"), left=left, conf_scale=CONF_SCALE, force=force, **st["q"])
            msgs = [user_msg(prompt, [R.b64png(i) for i in imgs])]
            r = self.vlm.chat(msgs, max_tokens=2000)
            ep.cost += r["cost"]
            j = extract_json(r["text"]) or {}
            j = j if isinstance(j, dict) else {}      # a bare JSON string / list counts as a broken reply (used to crash)
            retried = None
            if not j:       # broken / empty / unrelated reply (gateway glitches, e.g. "Hi! How can I help you today?"):
                retried = r["text"][:300]                    # resend the identical request once as a new sample
                r = self.vlm.chat(msgs, max_tokens=2000, cache_salt="retry1")
                ep.cost += r["cost"]
                j = extract_json(r["text"]) or {}
                j = j if isinstance(j, dict) else {}
            ep.steps.append({"step": turn, "mode": "vlm", "reply": j or r["text"][:800],
                             **({"retried_after": retried} if retried is not None else {})})
            if st["shown"] and st["shown"][-1]["next_conclusion"] is None:
                st["shown"][-1]["next_conclusion"] = str(j.get("step_conclusion"))
            if isinstance(j.get("ledger"), list):
                ledger = json.dumps(j["ledger"])[:2000]
            entry = [f"turn {turn}: conclusion: {str(j.get('step_conclusion'))[:400]} | answer {j.get('current_answer')} "
                     f"(confidence {j.get('confidence')})"]
            if j.get("final") is True or left == 1 or not j:
                final = j
                log.append(entry[0])
                break
            calls = j.get("calls") if isinstance(j.get("calls"), list) else []
            views = [c for c in calls if isinstance(c, dict) and c.get("tool") in (("overview", "view") if self.v3 else ("view",))][:1]
            for c in calls:
                if not isinstance(c, dict):
                    continue
                t = c.get("tool")
                if t == "measure":
                    entry.append("  " + self._measure(st, tree, c.get("target")))
                elif t == "compute":
                    entry.append("  " + self._compute(st, tree, c.get("expression")))
                elif t == "box":
                    entry.append("  " + self._box(st, tree, c))
                elif t == "knowledge":
                    entry.append("  " + self._knowledge(c.get("query")))
            for c in views:
                if c.get("tool") == "overview":
                    out, err = self._overview_set(tree, st, c.get("organs"), turn, ep)
                elif self.mode == "self":
                    out, err = self._self_view(tree, st, c, turn, ep)
                else:
                    out, err = self._navigate(tree, st, str(c.get("request") or ""), str(j.get("step_conclusion") or ""), turn, ep)
                if err:
                    entry.append("  " + err)
                else:
                    current.append(out)
                    entry.append(f"  view -> image set {out[0]} ({{SETDESC:{out[0]}}})")
            log.append("\n".join(entry))
        steps = [s["reply"] for s in ep.steps if s["mode"] == "vlm" and isinstance(s["reply"], dict)]
        if not final and steps:     # the last reply was unusable even after one resend: keep the last valid answer
            final = dict(steps[-1], fallback="last valid step")
        final = final or {}
        ep.final = {"answer": final.get("current_answer"), "confidence": final.get("confidence"),
                    "rationale": final.get("step_conclusion"),
                    "initial_answer": steps[0].get("current_answer") if steps else None,
                    "trajectory": [(s.get("current_answer"), s.get("confidence")) for s in steps]}
        ep.evidence = [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "meas"}} for k, v in st["F"].items()]
        return ep
