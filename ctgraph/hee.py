"""v2 search agent (round 2): HEE-style hierarchical entity exploration over the anatomical graph of a CT.

Search tree (built lazily; every level covers its parent, as HEE's entity + residual regions):
  root      = the whole body (all slices, voxels > -500 HU)
  root   -> anatomical region clusters of segmented structures (liver & gallbladder, kidneys & adrenals,
            pancreas/duodenum/central vessels, spleen & stomach, bowel, pelvis, thorax, bones & muscles, ...)
            + one residual node (body voxels not covered by any cluster)
  cluster-> its organs (small clusters) or z-slabs (clusters with many structures)
  organ  -> lesion instances from the segmentation tool (each connected component; > MAX_LESIONS are
            grouped) + anatomical parts (liver: 4 Couinaud groups + rest; kidney: poles; pancreas:
            head/body/tail; others: z-thirds) + residual (tissue within 10 mm, not organ / lesion)
  part / slab / residual -> z-slabs while taller than LEAF_MM, else LEAF
  leaf   = the node's FULL z-range as consecutive THICK_MM-thick mean slabs (tiles T1..Tn): no slice skipped
  lesion / patch = adjacent ORIGINAL thin slices (patch = zoom requested by the VLM on a leaf tile)
Each step the VLM either scores the children of the current node (one row of 3 slices per child +
full-slice locator with numbered boxes + coronal with z-ranges) or inspects a leaf.
Expansion / backtracking in code: expand the best child if relevance >= TAU_REL; after a leaf, go to the
best unvisited node in the global ranking. For existence / counting / burden questions, negative
conclusions need COVERAGE: every part of the target organs must have been inspected at leaf level; the
agent is routed to uncovered parts before it may stop.
Memory = ONE text: anatomical graph of the target organs + search tree with scores and notes + coverage
status + evidence list (re-rendered every step).
Round 3 (2026-10-21): no organ outline on leaf tiles (probe: the outline hid a renal cyst at the organ
edge), outward-offset outlines on previews, per-tile findings in leaf replies, coverage for every
question type except tumour-type classification, liver split along z instead of Couinaud groups
(segment of each evidence box is looked up in code), slice percentages for slice questions.
Round 4 (2026-10-22): tool-candidates + VLM calibration (each segmented lesion listed with code measurements
and contacts; VLM confirms / rejects / adds), code measurement of every VLM evidence box
(candidates.measure_local: slice-wise region growing -> volume, diameter, largest cross-section slice,
segment, contacts), coverage policy by question type (full sweep only where all lesions matter;
"largest ..." questions must visit every segmented candidate), step budget (MAX_STEPS, +EXTRA_AFTER_REQUIRED).
Round 5 (2026-10-29): an ANATOMY MAP image (representative slice with every segmented structure and the
Couinaud segments outlined + labelled) next to the plain tiles of every leaf, and an outlined CENTER panel next
to zoom triplets (P5: drawing the graph lets the VLM verify / correct it; plain tiles keep lesions at organ edges
visible); a text line with the crop's structure composition; all segmented lesion candidates of an organ in ONE
leaf ("candidates" node); full-coverage sweeps alternate between target organs (least-covered first).
Round 6 (2026-10-29): the benchmark's own knowledge base (ctgraph.knowledge, = its lookup_medical_knowledge
tool) retrieved by the question + options and added to every prompt; the round-2 instruction that contradicted the
benchmark's PDAC / PNET rule is removed.
Frozen earlier versions: hee_r1.py ... hee_r5.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import render as R
from . import views as V
from .agent import CONV, Episode, parse_options, question_stem
from .graph import LESION_ORGAN, neighbors
from .index import SpatialIndex
from .llm import LLM, extract_json, user_msg
from .reference import flags
from . import slicepct
from . import candidates as CAND
from .knowledge import knowledge_text
from .volume import Case

TAU_REL, TAU_STOP = 0.3, 0.75
MAX_LESIONS = 2           # more lesion instances than this are grouped into one child
LEAF_MM = 80.0            # nodes up to this z-extent are leaves (<= 16 thick tiles)
THICK_MM = 5.0
TILE = 224
MAX_CLUSTER_ORGANS = 6    # clusters with more structures are split along z instead of per structure
PALETTE = [(255, 80, 80), (80, 200, 255), (255, 210, 0), (90, 255, 90), (255, 90, 255), (255, 150, 40),
           (160, 160, 255), (200, 255, 160), (255, 160, 200), (160, 255, 255)]
NO_COVERAGE_SUBTYPES = ("PDAC vs PNET",)   # tumour-type classification: characterise the mass, no sweep needed
FULL_COVERAGE_SUBTYPES = ("existence", "counting", "count by location", "burden", "clustering", "inter-segment",
                          "asymmetry", "outlier")      # every lesion matters -> sweep all parts of the target organs
EXTRA_AFTER_REQUIRED = 2   # steps allowed after the required nodes have been visited

CLUSTERS = [  # (name, explicit structures, name prefixes)
    ("liver & gallbladder", ["liver", "gallbladder"], []),
    ("kidneys & adrenal glands", ["kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right",
                                  "adrenal_gland_left", "adrenal_gland_right"], []),
    ("pancreas, duodenum & central vessels", ["pancreas", "duodenum", "portal_vein_and_splenic_vein", "aorta",
                                              "inferior_vena_cava"], []),
    ("spleen & stomach", ["spleen", "stomach"], []),
    ("bowel", ["small_bowel", "colon"], []),
    ("pelvic organs & iliac vessels", ["urinary_bladder", "prostate"], ["iliac_"]),
    ("thorax & neck", ["heart", "trachea", "esophagus", "thyroid_gland", "pulmonary_vein", "superior_vena_cava",
                       "brachiocephalic_trunk", "atrial_appendage_left", "brain"],
     ["lung_", "brachiocephalic_vein", "common_carotid", "subclavian"]),
    ("bones & muscles", ["sacrum", "sternum", "costal_cartilages", "skull", "spinal_cord"],
     ["vertebrae_", "rib_", "hip_", "femur_", "humerus_", "scapula_", "clavicula_", "autochthon_", "iliopsoas_", "gluteus_"]),
]

TASK_SEMANTICS = """BENCHMARK DEFINITIONS (follow them exactly):
- A "lesion" is ANY focal tumor OR cyst of ANY size (even a few mm), regardless of clinical significance;
  e.g. a small simple renal cyst IS a kidney lesion. Existence questions only ask whether any lesion is present.
- Counting questions count every separate lesion of the asked type. "Lesion burden" compares the total number /
  volume of all lesions (tumors and cysts) in each organ."""


def cluster_of(name: str) -> str:
    for cname, names, prefixes in CLUSTERS:
        if name in names or any(name.startswith(p) for p in prefixes):
            return cname
    return "other structures"


def _num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class TNode:
    id: str
    name: str
    kind: str                  # root | cluster | organ | group | part | slab | residual | lesion | patch
    organ: str | None
    box: list                  # voxel x0, x1, y0, y1, z0, z1 (inclusive)
    mask: np.ndarray | None    # bool, shape of box; None = whole box
    parent: str | None
    depth: int
    z_rep: int
    graph_node: str | None = None
    members: list = field(default_factory=list)   # groups: (name, kind, graph_node, mask fn); clusters: structure names
    children: list = field(default_factory=list)
    expanded: bool = False
    score: float | None = None
    why: str = ""
    visited: bool = False
    notes: list = field(default_factory=list)


def _box_of(m: np.ndarray):
    # per-axis projections (2026-11-13): same box as argwhere min / max, but argwhere on a whole-body mask of a large
    # scan built a multi-GB index array and stalled episodes for minutes
    out = []
    for ax in range(3):
        nz = np.flatnonzero(m.any(tuple(a for a in range(3) if a != ax)))
        out += [int(nz[0]), int(nz[-1])]
    return out


class Tree:
    def __init__(self, case: Case, g: dict, targets: list[str]):
        self.case, self.g, self.sp = case, g, np.asarray(case.spacing)
        self.ix = SpatialIndex(case)
        self.targets = [t for t in targets if t in g["nodes"]]
        self.nodes: dict[str, TNode] = {}
        self._lab_cache: dict = {}
        self._info: dict = {}
        self.d = max(1, int(round(5.0 / self.sp[2])))
        self.k_thick = max(1, int(round(THICK_MM / self.sp[2])))
        self.body = case.ct > -500
        self.show_pct = False
        self.root = self._from_mask("whole body", "root", self.body, None, 0)

    # ---------------------------------------------------------------- node construction
    def _add(self, n: TNode) -> TNode:
        n.id = f"N{len(self.nodes)}"
        self.nodes[n.id] = n
        if n.parent:
            self.nodes[n.parent].children.append(n.id)
        return n

    def _from_mask(self, name, kind, full, parent, depth, organ=None, graph_node=None):
        if full is None or not full.any():
            return None
        box = _box_of(full)
        m = full[box[0]:box[1] + 1, box[2]:box[3] + 1, box[4]:box[5] + 1]
        z_rep = int(np.argmax(m.sum((0, 1)))) + box[4]
        return self._add(TNode("", name, kind, organ, box, m, parent, depth, z_rep, graph_node))

    def full_mask(self, n: TNode) -> np.ndarray:
        out = np.zeros(self.case.shape, bool)
        x0, x1, y0, y1, z0, z1 = n.box
        out[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True if n.mask is None else n.mask
        return out

    def lesions_of(self, organ: str) -> list[str]:
        """All segmented lesion instances whose label belongs to this organ (relation only describes position)."""
        labels = [l for l, o in LESION_ORGAN.items() if o == organ]
        out = [n for n in self.g["nodes"] if "#" in n and n.split("#")[0] in labels]
        return sorted(out, key=lambda n: -self.g["nodes"][n]["volume_ml"])

    def _lesion_mask(self, name: str) -> np.ndarray:
        label = name.split("#")[0]
        if label not in self._lab_cache:
            self._lab_cache[label] = ndimage.label(self.case.mask(label))[0]
        lab = self._lab_cache[label]
        x0, x1, y0, y1, z0, z1 = self.g["nodes"][name]["bbox_vox"]
        sub = lab[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        ids, cnt = np.unique(sub[sub > 0], return_counts=True)
        return lab == ids[np.argmax(cnt)]

    def lesion_info(self, name: str) -> dict:
        if name not in self._info:
            m = self._lesion_mask(name)
            area = m.sum((0, 1))
            organ = LESION_ORGAN.get(name.split("#")[0])
            self._info[name] = {"z_max_area": int(area.argmax()),
                                "contacts": CAND.contacts(self.case, m, exclude=(organ, name.split("#")[0]))}
        return self._info[name]

    def _organ_region(self, organ: str) -> np.ndarray:
        m = self.case.mask(organ)
        for l in self.lesions_of(organ):
            m |= self._lesion_mask(l)
        return m

    def cluster_members(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for n in self.case.names:
            out.setdefault(cluster_of(n), []).append(n)
        return out

    # ---------------------------------------------------------------- partition (HEE entity + residual)
    def is_leaf(self, n: TNode) -> bool:
        if n.kind in ("lesion", "patch", "candidates"):
            return True
        if n.kind in ("part", "slab", "residual"):
            return (n.box[5] - n.box[4] + 1) * self.sp[2] <= LEAF_MM
        return False

    def expand(self, n: TNode) -> list[TNode]:
        if n.expanded:
            return [self.nodes[c] for c in n.children]
        n.expanded = True
        if self.is_leaf(n):
            return []
        dep = n.depth + 1
        if n.kind == "root":
            covered = np.zeros(self.case.shape, bool)
            for cname, members in self.cluster_members().items():
                m = np.zeros(self.case.shape, bool)
                for s in members:
                    m |= self.case.mask(s)
                if "liver" in members and self.case.has("liver_tumor"):
                    m |= self.case.mask("liver_tumor")
                c = self._from_mask(cname, "cluster", m, n.id, dep)
                if c is not None:
                    c.members = members
                    covered |= m
            self._from_mask("residual body tissue (not covered by any segmented structure)", "residual",
                            self.body & ~covered, n.id, dep)
        elif n.kind == "cluster":
            organs = [s for s in n.members if s not in LESION_ORGAN and s in self.g["nodes"]]
            if len(organs) <= MAX_CLUSTER_ORGANS:
                for o in sorted(organs, key=lambda o: -self.g["nodes"][o]["volume_ml"]):
                    self._from_mask(o, "organ", self._organ_region(o), n.id, dep, organ=o, graph_node=o)
            else:
                self._z_split(n, force=True)
        elif n.kind == "organ":
            self._expand_organ(n)
        elif n.kind == "group":
            for name, kind, gnode, full in n.members:
                self._from_mask(name, kind, full(), n.id, dep, organ=n.organ, graph_node=gnode)
        elif n.kind in ("part", "slab", "residual"):
            self._z_split(n)
        return [self.nodes[c] for c in n.children]

    def _expand_organ(self, n: TNode):
        o, nid, dep = n.organ, n.id, n.depth + 1
        les = self.lesions_of(o)
        if les:   # all segmentation-tool lesion candidates of this organ in ONE leaf
            full = np.zeros(self.case.shape, bool)
            for l in les:
                full |= self._lesion_mask(l)
            cn = self._from_mask(f"all {len(les)} segmentation-tool lesion candidates in {o}", "candidates", full, nid, dep, organ=o)
            cn.members = les
        for name, full in self._parts(o, self.case.mask(o)):
            self._from_mask(name, "part", full, nid, dep, organ=o)
        x0, x1, y0, y1, z0, z1 = n.box
        p = np.ceil(10.0 / self.sp).astype(int)
        res = np.zeros(self.case.shape, bool)
        res[max(x0 - p[0], 0):x1 + p[0] + 1, max(y0 - p[1], 0):y1 + p[1] + 1, max(z0 - p[2], 0):z1 + p[2] + 1] = True
        res &= ~self.full_mask(n) & self.body
        self._from_mask(f"tissue around {o} (within 10 mm, not segmented as {o})", "residual", res, nid, dep, organ=o)

    def _parts(self, o: str, m: np.ndarray):
        if o == "liver_r2_couinaud_groups" and "liver_segments" in self.case.layers:   # disabled in round 3
            arr, names = self.case.layers["liver_segments"]
            groups = [("right posterior liver (segments 6, 7)", [6, 7]), ("right anterior liver (segments 5, 8)", [5, 8]),
                      ("left lateral liver (segments 2, 3)", [2, 3]), ("left medial liver + caudate (segments 4, 1)", [4, 1])]
            out, used = [], np.zeros(m.shape, bool)
            for gname, segs in groups:
                full = np.isin(arr, [names.index(f"liver_segment_{s}") + 1 for s in segs if f"liver_segment_{s}" in names]) & m
                used |= full
                out.append((gname, full))
            out.append(("liver parenchyma outside the segment masks", m & ~used))
            return out
        idx = np.argwhere(m)
        if o == "pancreas":
            pts = idx * self.sp
            mu = pts.mean(0)
            sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
            v = np.linalg.svd(sub - mu, full_matrices=False)[2][0]
            v = -v if v[0] < 0 else v           # +x = patient right = head
            proj = (pts - mu) @ v
            q1, q2 = np.quantile(proj, [1 / 3, 2 / 3])
            out = []
            for name, sel in (("pancreatic head", proj >= q2), ("pancreatic body", (proj >= q1) & (proj < q2)),
                              ("pancreatic tail", proj < q1)):
                full = np.zeros(m.shape, bool)
                full[tuple(idx[sel].T)] = True
                out.append((name, full))
            return out
        z = idx[:, 2]
        if (z.max() - z.min() + 1) * self.sp[2] <= LEAF_MM / 2:
            return [(f"{o} (whole)", m)]
        q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
        labels = (("upper pole", "interpolar region", "lower pole") if o.startswith("kidney")
                  else ("upper third", "middle third", "lower third"))
        out = []
        for name, (lo, hi) in zip(labels, ((q2, z.max() + 1), (q1, q2), (z.min(), q1))):
            full = m.copy()
            full[:, :, :int(np.ceil(lo))] = False
            full[:, :, int(np.ceil(hi)):] = False
            out.append((f"{o} {name}", full))
        return out

    def _z_split(self, n: TNode, force: bool = False):
        x0, x1, y0, y1, z0, z1 = n.box
        ext = (z1 - z0 + 1) * self.sp[2]
        if ext <= LEAF_MM and not force:
            return
        k = int(np.clip(np.ceil(ext / LEAF_MM), 2, 4))
        cuts = np.linspace(z0, z1 + 1, k + 1).round().astype(int)
        for a, b in zip(cuts[:-1], cuts[1:]):
            full = self.full_mask(n)
            full[:, :, :a] = False
            full[:, :, b:] = False
            self._from_mask(f"{n.name} [z {a}-{b - 1}]", "slab", full, n.id, n.depth + 1, organ=n.organ)

    def add_patch(self, n: TNode, box_frac, z: int) -> TNode:
        crop = self.leaf_crop(n)
        b = V.pad_box(V.crop_frac_to_box(crop, box_frac), self.case.shape, 2, self.sp, 25)
        z = int(np.clip(z, 0, self.case.shape[2] - 1))
        z0, z1 = max(z - self.d, 0), min(z + self.d, self.case.shape[2] - 1)
        return self._add(TNode("", f"zoom patch of {n.name} at z={z}", "patch", n.organ,
                               [b[0], b[1], b[2], b[3], z0, z1], None, n.id, n.depth + 1, z))

    # ---------------------------------------------------------------- coverage
    def covered(self, n: TNode) -> bool:
        if self.is_leaf(n):
            return n.visited
        if not n.expanded:
            return False
        kids = [self.nodes[c] for c in n.children if self.nodes[c].kind not in ("residual", "patch")]
        return all(self.covered(c) for c in kids)

    def organ_node(self, t: str):
        return next((n for n in self.nodes.values() if n.kind == "organ" and n.graph_node == t), None)

    def next_required(self):
        """Next node to visit so that every part of every target organ gets inspected at leaf level."""
        if not self.root.expanded:
            return self.root
        order = sorted(self.targets, key=self._cov_frac)       # alternate: least-covered target organ first
        for t in order:
            cl = next((n for n in self.nodes.values() if n.kind == "cluster" and t in n.members), None)
            if cl is None:
                continue
            if not cl.expanded:
                return cl
            on = self.organ_node(t)
            if on is None:
                continue
            r = self._first_uncovered(on)
            if r is not None:
                return r
        return None

    def _cov_frac(self, t: str) -> float:
        on = self.organ_node(t)
        if on is None or not on.expanded:
            return 0.0
        parts = [self.nodes[c] for c in on.children if self.nodes[c].kind not in ("residual", "patch")]
        return sum(self.covered(p) for p in parts) / max(len(parts), 1)

    def next_candidate(self):
        """'candidates' mode: every segmented lesion of the target organs must be inspected."""
        if not self.root.expanded:
            return self.root
        for t in self.targets:
            cl = next((n for n in self.nodes.values() if n.kind == "cluster" and t in n.members), None)
            if cl is None:
                continue
            if not cl.expanded:
                return cl
            on = self.organ_node(t)
            if on is None:
                continue
            if not on.expanded:
                return on
            for c in [self.nodes[c] for c in on.children]:
                if c.kind == "candidates" and not c.visited:
                    return c
        return None

    def _first_uncovered(self, n: TNode):
        if self.covered(n):
            return None
        if self.is_leaf(n) or not n.expanded:
            return n
        kids = [self.nodes[c] for c in n.children if self.nodes[c].kind not in ("residual", "patch")]
        for c in sorted(kids, key=lambda c: -(c.score or 0)):
            r = self._first_uncovered(c)
            if r is not None:
                return r
        return None

    def coverage_text(self) -> str:
        out = []
        for t in self.targets:
            on = self.organ_node(t)
            if on is None or not on.expanded:
                out.append(f"{t}: not inspected yet")
                continue
            parts = [self.nodes[c] for c in on.children if self.nodes[c].kind not in ("residual", "patch")]
            todo = [p.name for p in parts if not self.covered(p)]
            out.append(f"{t}: {len(parts) - len(todo)}/{len(parts)} sub-regions fully inspected"
                       + (f"; NOT yet: {', '.join(todo)}" if todo else " (complete)"))
        return "; ".join(out)

    # ---------------------------------------------------------------- rendering
    def crop_box(self, n: TNode, margin=15, min_mm=60):
        if n.kind == "patch":
            return n.box[:4]
        return V.pad_box(n.box[:4], self.case.shape, margin, self.sp, min_mm)

    def leaf_crop(self, n: TNode):
        return self.crop_box(n, margin=20, min_mm=80)

    def window(self, n: TNode):
        if n.organ == "liver":
            return (self.g["nodes"]["liver"]["hu_mean"], 200)
        return "soft_tissue"

    def mask_slab(self, n: TNode, za: int, zb: int) -> np.ndarray:
        """2D (x, y) footprint of node n over slices za..zb."""
        out = np.zeros(self.case.shape[:2], bool)
        x0, x1, y0, y1, z0, z1 = n.box
        a, b = max(za, z0), min(zb, z1)
        if a <= b:
            out[x0:x1 + 1, y0:y1 + 1] = True if n.mask is None else n.mask[:, :, a - z0:b - z0 + 1].any(2)
        return out

    def _tile(self, n: TNode, za: int, zb: int, crop, size: int, color, label: str, outline: bool = True):
        """Axial crop of the MEAN of slices za..zb (za == zb: one original slice). Optional outline of n,
        drawn 2 voxels OUTSIDE its boundary so it never covers the organ edge."""
        x0, x1, y0, y1 = crop
        Z = self.case.shape[2]
        za, zb = int(np.clip(za, 0, Z - 1)), int(np.clip(zb, 0, Z - 1))
        sub = self.case.ct[x0:x1 + 1, y0:y1 + 1, za:zb + 1].astype(np.float32).mean(2)
        im, _ = R.upscale(R.to_rgb(R.window(sub[::-1, ::-1].T, self.window(n))), size)
        if outline and n.kind not in ("patch", "root"):
            m2 = ndimage.binary_dilation(self.mask_slab(n, za, zb), iterations=2)[x0:x1 + 1, y0:y1 + 1]
            if m2.any() and not m2.all():
                im = R.overlay_mask(im, m2[::-1, ::-1].T, color=color, alpha=0.0)
        return V.label(im, label, 15)

    def child_levels(self, c: TNode) -> list[int]:
        z0, z1 = c.box[4], c.box[5]
        if c.kind in ("lesion", "patch") or z1 - z0 < 3 * self.d:
            zs = [c.z_rep + self.d, c.z_rep, c.z_rep - self.d]
        else:
            zz = np.nonzero(c.mask.any((0, 1)))[0] + z0 if c.mask is not None else np.arange(z0, z1 + 1)
            zs = [int(np.quantile(zz, 0.85)), c.z_rep, int(np.quantile(zz, 0.15))]
        return sorted((int(np.clip(z, 0, self.case.shape[2] - 1)) for z in zs), reverse=True)

    def render_children(self, n: TNode, kids: list[TNode]):
        rows = []
        for k, c in enumerate(kids):
            col = PALETTE[k % len(PALETTE)]
            crop = self.crop_box(c)
            rows.append(V.montage([self._tile(c, z, z, crop, TILE, col, f"C{k + 1} z={z}") for z in self.child_levels(c)], 3))
        grid = V.montage(rows, 1)
        z = n.z_rep
        loc, s = R.upscale(R.to_rgb(R.axial(self.case.ct, z, self.window(n))), 512)
        d = ImageDraw.Draw(loc)
        f = R._font(18)
        for k, c in enumerate(kids):
            m2 = self.mask_slab(c, z, z)
            if m2.any():
                rr, cc = np.nonzero(m2[::-1, ::-1].T)
                col = PALETTE[k % len(PALETTE)]
                d.rectangle([cc.min() * s, rr.min() * s, cc.max() * s, rr.max() * s], outline=col, width=2)
                d.text((cc.min() * s + 2, max(rr.min() * s - 20, 0)), f"C{k + 1}", fill=col, font=f)
        loc = V.label(loc, f"locator: full axial slice z={z}", 16)
        return [grid, loc, self._coronal_ranges(n, kids)], z

    def _coronal_ranges(self, n: TNode, kids: list[TNode]):
        y = int((n.box[2] + n.box[3]) / 2)
        img = V.coronal_with_levels(self.case, y, [], 512)
        sz = self.case.spacing[2]
        Z = self.case.shape[2]
        sc = img.size[1] / (Z * sz)
        d = ImageDraw.Draw(img)
        f = R._font(13)
        for k, c in enumerate(kids):
            col = PALETTE[k % len(PALETTE)]
            x = img.size[0] - 14 - 16 * k
            ytop, ybot = (Z - 1 - c.box[5]) * sz * sc, (Z - c.box[4]) * sz * sc
            d.line([(x, ytop), (x, ybot)], fill=col, width=5)
            d.text((x - 6, max(ytop - 15, 0)), f"C{k + 1}", fill=col, font=f)
        return V.label(img, "coronal: bars = z-range of each child", 14)

    def _pct_label(self, a: int, b: int) -> str:
        if not self.show_pct:
            return ""
        p = sorted(v for v in (slicepct.pct(self.case.case_id, a), slicepct.pct(self.case.case_id, b)) if v is not None)
        return f" ({p[0]:.1f}-{p[-1]:.1f}%)" if p else ""

    def thick_tiles(self, n: TNode) -> list[tuple[int, int]]:
        z0, z1 = n.box[4], n.box[5]
        tiles = [(a, min(a + self.k_thick - 1, z1)) for a in range(z0, z1 + 1, self.k_thick)]
        return tiles[::-1]                      # superior first

    def anatomy_map(self, z: int, crop, size: int = 384, mark_box=None, snap: bool = False):
        """Slice z cropped, every segmented structure outlined + labelled; the liver drawn as Couinaud segments.
        Lesion labels are not drawn (lesions must be judged on the plain images).
        snap: anchor each label on a pixel of its own structure (the median of a thin curved structure can fall
        outside it) and keep labels off mark_box."""
        img = V.axial_crop(self.case, z, crop, "soft_tissue", size, mark_box=mark_box)
        x0, x1, y0, y1 = crop
        lab = self.case.labels[x0:x1 + 1, y0:y1 + 1, z]
        liver_id = self.case.name_to_id("liver") if "liver" in self.case.names else -1
        regions = []
        min_px = 0.005 * lab.size          # skip tiny fragments; ribs / costal cartilage only add clutter
        for k in np.unique(lab):
            if k == 0 or k == liver_id or self.case.names[k - 1] in LESION_ORGAN:
                continue
            nm = self.case.names[k - 1]
            if nm.startswith(("rib_", "costal")) or (lab == k).sum() < min_px:
                continue
            regions.append((nm.replace("_", " "), lab == k))
        if liver_id > 0 and "liver_segments" in self.case.layers:
            arr, names = self.case.layers["liver_segments"]
            seg = arr[x0:x1 + 1, y0:y1 + 1, z]
            liv = (lab == liver_id) | np.isin(lab, [self.case.name_to_id(l) for l in LESION_ORGAN if l in self.case.names])
            for i in np.unique(seg[liv]):
                if i > 0:
                    regions.append((f"seg {names[i - 1].split('_')[-1]}", (seg == i) & liv))
        s = size / max(x1 - x0 + 1, y1 - y0 + 1)
        f = R._font(15)
        for j, (name, m) in enumerate(regions):
            if m.sum() < 15:
                continue
            rc = m[::-1, ::-1].T
            col = PALETTE[j % len(PALETTE)]
            img = R.overlay_mask(img, rc, color=col, alpha=0.0)
            d = ImageDraw.Draw(img)
            rr, cc = np.nonzero(rc)
            r, c = np.median(rr) * s, np.median(cc) * s
            if snap:
                pr, pc = rr * s, cc * s
                ok = np.ones(len(pr), bool)
                if mark_box is not None:     # label rectangle [pc-2, pr-9, pc+8*len, pr+9] must not overlap the box
                    br0, br1 = (y1 - mark_box[3]) * s, (y1 - mark_box[2] + 1) * s
                    bc0, bc1 = (x1 - mark_box[1]) * s, (x1 - mark_box[0] + 1) * s
                    ok = ~((pr + 9 >= br0) & (pr - 9 <= br1) & (pc + 8 * len(name) >= bc0) & (pc - 2 <= bc1))
                if ok.any():
                    i = np.argmin(np.where(ok, (pr - r) ** 2 + (pc - c) ** 2, np.inf))
                    r, c = pr[i], pc[i]
            d.rectangle([c - 2, r - 9, c + 8 * len(name), r + 9], fill=(0, 0, 0))
            d.text((c, r - 9), name, fill=col, font=f)
        return V.label(img, f"ANATOMY MAP z={z} (segmentation outlines)", 15)

    def composition(self, crop, z0: int, z1: int) -> str:
        """Structure composition of the crop over slices z0..z1 (liver also as Couinaud segment shares)."""
        x0, x1, y0, y1 = crop
        bn = self.ix.box_nodes(x0, x1, y0, y1, max(z0, 0), min(z1, self.case.shape[2] - 1))
        bn = {k: v for k, v in bn.items() if k not in LESION_ORGAN}
        tot = sum(bn.values()) or 1
        parts = [f"{k.replace('_', ' ')} {v / tot * 100:.0f}%" for k, v in sorted(bn.items(), key=lambda t: -t[1]) if v / tot >= 0.03]
        if "liver_segments" in self.case.layers and "liver" in bn and bn["liver"] / tot >= 0.03:
            arr, names = self.case.layers["liver_segments"]
            liv = self.case.labels[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] == self.case.name_to_id("liver")
            cnt = np.bincount(arr[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1][liv], minlength=len(names) + 1)[1:]
            if cnt.sum():
                parts.append("liver segments in the crop: " + ", ".join(
                    f"{names[i].split('_')[-1]} {c / cnt.sum() * 100:.0f}%" for i, c in sorted(enumerate(cnt), key=lambda t: -t[1]) if c / cnt.sum() >= 0.05))
        return ", ".join(parts) or "no segmented structure"

    def render_candidates(self, n: TNode):
        """One montage with every segmentation-tool lesion candidate (thin CENTER slice, box) + an anatomy map."""
        ims, info = [], []
        for i, l in enumerate(n.members[:12]):
            nd = self.g["nodes"][l]
            m = self._lesion_mask(l)
            x0, x1, y0, y1, z0, z1 = nd["bbox_vox"]
            z = z0 + int(m[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1].sum((0, 1)).argmax())
            idx = np.argwhere(m[:, :, z])
            box = [int(idx[:, 0].min()), int(idx[:, 0].max()), int(idx[:, 1].min()), int(idx[:, 1].max())]
            crop = V.pad_box(box, self.case.shape, 20, self.sp, 50)
            ims.append(V.label(V.axial_crop(self.case, z, crop, self.window(n), TILE, mark_box=box), f"L{i + 1} z={z}", 15))
            info.append({"tag": f"L{i + 1}", "lesion": l, "z": z, "box": box, "crop": crop})
        maps = [self.anatomy_map(info[0]["z"], self.leaf_crop(n), 384)] if info else []
        return [V.montage(ims, 4)] + maps, info

    def render_leaf(self, n: TNode):
        """(images, crop, tiles): thin adjacent slices (+ outlined CENTER) for lesions/patches; else the node's full
        z-range as plain thick-slab tiles T1..Tn + an anatomy map + locator."""
        crop = self.leaf_crop(n)
        loc = V.label(V.locator(self.case, n.z_rep, crop, 384), f"locator: full axial slice z={n.z_rep}", 16)
        if n.kind in ("lesion", "patch"):
            ims = [self._tile(n, z, z, crop, 384, (255, 80, 80), f"z={z} ({t})", outline=False)
                   for z, t in zip([n.z_rep + self.d, n.z_rep, n.z_rep - self.d], ("above", "CENTER", "below"))]
            ims.append(self.anatomy_map(n.z_rep, crop, 384))
            return [V.montage(ims, 4), loc], crop, None
        tiles = self.thick_tiles(n)
        ims = [self._tile(n, a, b, crop, TILE, (255, 80, 80), f"T{i + 1} z{a}-{b}" + self._pct_label(a, b), outline=False)
               for i, (a, b) in enumerate(tiles)]
        return [V.montage(ims, 4), self.anatomy_map(n.z_rep, crop, 384), loc], crop, tiles


# ---------------------------------------------------------------------- text: graph + notes

def _zr(nd) -> str:
    return f"z {nd['bbox_vox'][4]}-{nd['bbox_vox'][5]}"


def graph_text(tree: Tree, evidence: list, use_graph: bool = True) -> str:
    g, c = tree.g, tree.case
    Z = c.shape[2]
    lines = [f"CT volume: {Z} axial slices, z=0 = most INFERIOR (feet side), z={Z - 1} = most SUPERIOR (head side); "
             f"slice spacing {c.spacing[2]:.1f} mm, pixel {c.spacing[0]:.2f} mm. Target structure(s) of the question: "
             f"{', '.join(tree.targets)}."]
    if tree.show_pct:
        lines.append(slicepct.describe(c.case_id, Z - 1))
    if use_graph:
        lesion_labels = set(LESION_ORGAN)
        for t in tree.targets:
            n = g["nodes"][t]
            fl = flags(g, t)
            lines.append(f"[{t}] {_zr(n)}, volume {n['volume_ml']:.0f} mL, max diameter {n['max_diameter_mm']:.0f} mm, "
                         f"mean HU {n['hu_mean']:.0f}" + (f"; reference: {fl}" if fl else ""))
            les = tree.lesions_of(t)
            lab = [l for l, o in LESION_ORGAN.items() if o == t]
            if lab:
                lines.append(f"  segmentation-tool lesions in {t} ({', '.join(lab)}; the tool only detects liver tumors and "
                             f"kidney cysts, it can MISS lesions (including the largest one) or produce false positives - "
                             f"treat this list as a starting point to verify, not as ground truth): " + ("none" if not les else ""))
            for l in les:
                nd = g["nodes"][l]
                rel = [e for e in g["edges"] if e["b"] == l]
                where = "; ".join(f"{'inside' if e['type'] == 'contains' else e['type']} {e['a']}" for e in rel) or f"near {t}"
                info = tree.lesion_info(l)
                zm = nd["bbox_vox"][4] + info["z_max_area"]
                pc = slicepct.pct(c.case_id, zm) if tree.show_pct else None
                touch = ", ".join(f"{k} {v * 100:.0f}%" for k, v in info["contacts"].items()) or "nothing else"
                lines.append(f"    {l}: {_zr(nd)}, volume {nd['volume_ml']:.2f} mL, max diameter {nd['max_diameter_mm']:.0f} mm, "
                             f"mean HU {nd['hu_mean']:.0f} ({where}); largest cross-section z={zm}"
                             + (f" ({pc:.1f}%)" if pc is not None else "") + f"; touches: {touch}")
            for e in [e for e in neighbors(g, t) if e.get("type", "adjacent") == "adjacent" and "#" not in e["a"] + e["b"]
                      and not ({e["a"], e["b"]} & lesion_labels)][:6]:
                o = e["b"] if e["a"] == t else e["a"]
                no, nt = g["nodes"][o], g["nodes"][t]
                dx, dy = (np.array(no["centroid_mm"][:2]) - np.array(nt["centroid_mm"][:2]))
                lines.append(f"  neighbour {o} ({_zr(no)}): {V.image_frame_phrase((dx, dy))} of {t} on axial images; "
                             + ("touching" if e["contact_area_mm2"] > 0 else f"gap {e['min_dist_mm']:.0f} mm"))
    lines.append("SEARCH TREE (ids, z-range, your relevance scores and notes):")
    for nid, n in tree.nodes.items():
        if not (n.visited or n.score is not None or n is tree.root):
            continue
        tag = "inspected" if (n.visited and tree.is_leaf(n)) else ("visited" if n.visited else "scored")
        sc = f", relevance {n.score:.2f}" if n.score is not None else ""
        lines.append(f"  {'  ' * n.depth}{nid} {n.name} (z {n.box[4]}-{n.box[5]}{sc}, {tag})"
                     + "".join(f"\n  {'  ' * n.depth}   note: {x}" for x in n.notes))
    lines.append("COVERAGE of target organs at leaf level: " + tree.coverage_text())
    if evidence:
        lines.append("EVIDENCE:")
        for e in evidence:
            lines.append(f"  E{e['id']} (node {e['node']}, z={e['z']}, structures {e['structures']}): {e['finding']} "
                         f"[supports {e.get('supports')}, conf {e.get('confidence')}]"
                         + (f"\n     {e['measure']}" if e.get("measure") else ""))
    return "\n".join(lines)


PROMPT_SCORE = """You are an expert abdominal radiologist searching a 3D CT for the evidence needed to answer a multiple-choice question.
The search is a hierarchical tree over anatomical entities from automatic segmentation (whole body -> anatomical region
clusters -> organs -> lesions / parts -> slabs -> zoom patches); every level covers its parent completely.
{conv}
{task}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.

ANATOMICAL GRAPH WITH SEARCH NOTES:
{graph}

CURRENT NODE: {node_id} {node_name} (z {z0}-{z1}). Its {k} children are shown:
- Image 1: one row per child C1..C{k}; each row = 3 axial slices of that child (upper / representative / lower level of its
  z-range, labelled with z); the child's segmented outline is drawn in its colour slightly OUTSIDE its boundary.
- Image 2: the full axial slice z={zloc} with numbered boxes (same colours) of the children present on that slice.
- Image 3: coronal view; coloured bars = z-range of each child.
Children: {children}

TASK FOR THIS STEP: these are coarse previews to decide WHERE to look - children you score >= {tau} will later be inspected
slice by slice. For EACH child estimate the probability (0-1) that it contains evidence relevant to the question; regions
that cannot contain the asked finding should get low scores. Then say whether the evidence gathered so far is already
sufficient to answer.
Reply with STRICT JSON only:
{{"scores": {{"C1": {{"relevance": <0-1>, "why": "<short>"}}, ...}},
  "notes": [{{"node": "<C# or N#>", "text": "<what you saw / concluded there, kept as memory>"}}],
  "evidence": [{{"child": "C#", "finding": "<visible finding>", "supports": "<option letter or null>", "confidence": <0-1>}}],
  "sufficient": true|false, "best_option": "<letter>", "confidence": <0-1>}}"""

PROMPT_LEAF = """You are an expert abdominal radiologist searching a 3D CT for the evidence needed to answer a multiple-choice question.
The search is a hierarchical tree over anatomical entities from automatic segmentation.
{conv}
{task}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.

ANATOMICAL GRAPH WITH SEARCH NOTES:
{graph}

CURRENT NODE (leaf): {node_id} {node_name}. {extra}
{images}

If you see a lesion, put a TIGHT box around it in "evidence" (one entry per lesion): code will then measure it
(volume, diameter, largest cross-section slice, Couinaud segment, touching structures) and add the result to the record.
If a segmentation-tool lesion is shown, say whether it is real and whether it is the lesion the question asks about.
TASK: first go through the images one by one ({tile_list}) and note every focal region whose density differs from the
surrounding parenchyma (not normal vessels / renal sinus / bowel), then describe what this region shows with respect to the
question, decide whether the evidence is now sufficient, and
optionally request a zoom {zoom_how}. Unexplored relevant nodes and uncovered parts of the target organs will be visited
automatically if the evidence is not sufficient. As soon as the decisive evidence for the question has been seen and
measured, set "sufficient": true - the search is expensive, do not keep looking for confirmation you do not need.
Reply with STRICT JSON only:
{{"per_image": {{"<T# or panel>": "<focal findings or 'none'>", ...}},
  "observation": "<1-3 sentences>",
  "notes": [{{"node": "<N# of this or another node>", "text": "<memory note>"}}],
  "evidence": [{{{ev_tile}"box": [l, t, r, b] or null, "finding": "<visible finding>", "supports": "<option letter or null>", "confidence": <0-1>}}],
  "zoom": {zoom_fmt} or null,
  "sufficient": true|false, "best_option": "<letter>", "confidence": <0-1>}}"""

IMG_THICK = """- Image 1: the region's COMPLETE z-range as consecutive {mm:.0f} mm-thick axial slabs T1..T{n} (each = mean of the original
  thin slices in its z-range, labelled; superior first, row-major), all with the same crop. No outline is drawn so that
  lesions at organ edges stay visible; no slice of this region is skipped. Lesions smaller than ~5 mm can look faint on
  thick slabs - zoom to check them.
- Image 2: ANATOMY MAP - the same crop at z={zc} with every segmented structure outlined and labelled (the liver drawn as its
  Couinaud segments 'seg k'). Use it to place findings anatomically; outlines can be imperfect.
- Image 3: full axial slice z={zc} with the displayed crop outlined."""
IMG_THIN = """- Image 1: four panels: three adjacent ORIGINAL thin axial slices z={za} (above) / z={zc} (CENTER) / z={zb} (below), without
  outlines, and an ANATOMY MAP of the CENTER slice (every segmented structure outlined and labelled; liver as Couinaud
  segments 'seg k'; outlines can be imperfect).
- Image 2: full axial slice z={zc} with the displayed crop outlined."""
IMG_CAND = """- Image 1: every lesion candidate found by the segmentation tool in this organ, one crop each (L1..L{n}, ORIGINAL thin slice
  through its largest cross-section; the box marks the tool's lesion). The tool can be wrong: decide for EACH candidate
  whether it is a real lesion.
- Image 2: ANATOMY MAP of the organ at z={zc} (segmented structures / Couinaud segments outlined and labelled).
Candidates: {listing}"""

PROMPT_FINAL = """You are an expert abdominal radiologist. Answer the multiple-choice question about a CT scan using the search record.
{conv}
{task}

QUESTION: {question}
OPTIONS: {options}
Patient: sex={sex}, age={age}.

ANATOMICAL GRAPH WITH SEARCH NOTES AND EVIDENCE:
{graph}
{images_note}
Reply with STRICT JSON only: {{"answer": "<letter>", "confidence": <0-1>, "rationale": "<1-2 sentences>"}}"""


def _tile_index(v, tiles):
    if tiles is None or v is None:
        return None
    m = re.search(r"\d+", str(v))
    if not m:
        return None
    i = int(m.group()) - 1
    return i if 0 <= i < len(tiles) else None


class HEEAgent:
    def __init__(self, llm: LLM, max_steps: int = 12, use_graph: bool = True, final_mode: str = "crops+locator"):
        self.llm, self.max_steps, self.use_graph, self.final_mode = llm, max_steps, use_graph, final_mode

    def _call(self, ep, text, imgs):
        r = self.llm.chat([user_msg(text, [R.b64png(i) for i in imgs])])
        ep.cost += r["cost"]
        return r

    def _ctx(self, item):
        kb = knowledge_text(item["mc_question"])
        return dict(conv=CONV, task=TASK_SEMANTICS + ("\n" + kb if kb else ""), question=question_stem(item["mc_question"]),
                    options=json.dumps(parse_options(item["mc_question"])), sex=item.get("sex"), age=item.get("age"))

    def run(self, case: Case, g: dict, targets: list[str], item: dict) -> Episode:
        tree = Tree(case, g, targets)
        tree.show_pct = "slice" in item["question_subtype"]
        ep, ctx = Episode(item), self._ctx(item)
        st_name = item["question_subtype"]
        mode_cov = ("none" if any(k in st_name for k in NO_COVERAGE_SUBTYPES) else
                    "full" if any(k in st_name for k in FULL_COVERAGE_SUBTYPES) else "candidates")
        required = {"full": tree.next_required, "candidates": tree.next_candidate, "none": lambda: None}[mode_cov]
        need_cov = mode_cov != "none"
        n_extra = 0
        cur = tree.root
        for step in range(self.max_steps):
            if step > 0 and need_cov and required() is None:     # step budget once the required nodes are done
                n_extra += 1
                if n_extra > EXTRA_AFTER_REQUIRED:
                    break
            cur.visited = True
            kids = tree.expand(cur)
            graph = graph_text(tree, ep.evidence, self.use_graph)
            tiles, crop = None, None
            cand = None
            if kids:
                imgs, zloc = tree.render_children(cur, kids)
                desc = "; ".join(f"C{k + 1} = {c.id} {c.name} (z {c.box[4]}-{c.box[5]})" for k, c in enumerate(kids))
                prompt = PROMPT_SCORE.format(graph=graph, node_id=cur.id, node_name=cur.name, z0=cur.box[4], z1=cur.box[5],
                                             k=len(kids), zloc=zloc, children=desc, tau=TAU_REL, **ctx)
                mode = "score"
            elif cur.kind == "candidates":
                imgs, cand = tree.render_candidates(cur)
                crop, tiles = None, None
                listing = []
                for ci in cand:
                    nd, info = g["nodes"][ci["lesion"]], tree.lesion_info(ci["lesion"])
                    seg = [e["a"].split("_")[-1] for e in g["edges"] if e["b"] == ci["lesion"] and e["a"].startswith("liver_segment")]
                    pc = slicepct.pct(case.case_id, ci["z"]) if tree.show_pct else None
                    listing.append(f"{ci['tag']} = {ci['lesion']}: volume {nd['volume_ml']:.2f} mL, max diameter {nd['max_diameter_mm']:.0f} mm, "
                                   f"mean HU {nd['hu_mean']:.0f}, z {nd['bbox_vox'][4]}-{nd['bbox_vox'][5]}, largest cross-section z={ci['z']}"
                                   + (f" ({pc:.1f}%)" if pc is not None else "") + (f", Couinaud segment(s) {'/'.join(seg)}" if seg else "")
                                   + ", touches " + (", ".join(info["contacts"]) or "nothing else"))
                im_txt = IMG_CAND.format(n=len(cand), zc=cand[0]["z"] if cand else cur.z_rep, listing="; ".join(listing))
                prompt = PROMPT_LEAF.format(graph=graph, node_id=cur.id, node_name=cur.name, extra="", images=im_txt,
                                            zoom_how="(not available for candidate crops: set zoom to null)", zoom_fmt="null",
                                            ev_tile='"tile": "L#", ', tile_list=f"L1..L{len(cand)}", **ctx)
                mode = "leaf"
            else:
                imgs, crop, tiles = tree.render_leaf(cur)
                cand = None
                extra = ""
                if cur.graph_node and "#" in cur.graph_node:
                    nd = g["nodes"][cur.graph_node]
                    extra = (f"Segmentation-tool measurements: volume {nd['volume_ml']:.2f} mL, max diameter "
                             f"{nd['max_diameter_mm']:.0f} mm, mean HU {nd['hu_mean']:.0f}.")
                if tiles:
                    im_txt = IMG_THICK.format(mm=tree.k_thick * tree.sp[2], n=len(tiles), zc=cur.z_rep)
                    zoom_how = ('into the ORIGINAL thin slices of one tile: give the tile and a box as fractions '
                                '[left, top, right, bottom] of that tile')
                    zoom_fmt, ev_tile = '{"tile": "T#", "box": [l, t, r, b]}', '"tile": "T#", '
                else:
                    im_txt = IMG_THIN.format(za=cur.z_rep + tree.d, zc=cur.z_rep, zb=cur.z_rep - tree.d)
                    zoom_how = 'into a sub-region of the CENTER panel (box as fractions [left, top, right, bottom])'
                    zoom_fmt, ev_tile = '[l, t, r, b]', ''
                tile_list = f"T1..T{len(tiles)}" if tiles else "above / CENTER / below"
                za, zb = (tiles[-1][0], tiles[0][1]) if tiles else (cur.z_rep - tree.d, cur.z_rep + tree.d)
                extra += f" Structure composition of the displayed crop (z {za}-{zb}): {tree.composition(crop, za, zb)}."
                prompt = PROMPT_LEAF.format(graph=graph, node_id=cur.id, node_name=cur.name, extra=extra, images=im_txt,
                                            zoom_how=zoom_how, zoom_fmt=zoom_fmt, ev_tile=ev_tile, tile_list=tile_list, **ctx)
                mode = "leaf"
            r = self._call(ep, prompt, imgs)
            j = extract_json(r["text"])
            rec = {"step": step, "mode": mode, "node": cur.id, "node_name": cur.name,
                   "children": [c.id for c in kids], "reply": j or r["text"][:800]}
            ep.steps.append(rec)
            if j:
                self._apply(tree, cur, kids, j, ep, crop, tiles, cand)
                conf = float(j.get("confidence")) if _num(j.get("confidence")) else 0.0
                leaf_seen = any(st.get("mode") == "leaf" for st in ep.steps if "mode" in st)
                if j.get("sufficient") is True and conf >= TAU_STOP and leaf_seen:
                    positive = any(isinstance(e, dict) and _num(e.get("confidence")) and float(e["confidence"]) >= TAU_STOP
                                   for e in (j.get("evidence") or []))
                    if not need_cov or required() is None or ("existence" in st_name and positive):
                        break
                if mode == "leaf" and cur.kind not in ("patch", "candidates"):
                    zr = self._zoom_request(j.get("zoom"), tiles, cur)
                    if zr is not None:
                        cur = tree.add_patch(cur, zr[0], zr[1])
                        rec["patch"] = [cur.parent, zr[0], zr[1]]
                        continue
                if mode == "score":
                    best = max(kids, key=lambda c: c.score or 0)
                    if (best.score or 0) >= TAU_REL and not best.visited:
                        cur = best
                        continue
            nxt = required() or self._next(tree)
            if nxt is None:
                break
            cur = nxt
        ep.final = self._final(tree, ep, ctx)
        ep.steps.append({"tree": {nid: {k: v for k, v in asdict(n).items() if k not in ("mask", "members")}
                                  for nid, n in tree.nodes.items()},
                         "coverage": tree.coverage_text(), "coverage_mode": mode_cov})
        return ep

    @staticmethod
    def _zoom_request(z, tiles, cur):
        if tiles:
            if isinstance(z, dict) and isinstance(z.get("box"), list) and len(z["box"]) == 4:
                i = _tile_index(z.get("tile"), tiles)
                if i is not None:
                    a, b = tiles[i]
                    return z["box"], (a + b) // 2
            return None
        if isinstance(z, list) and len(z) == 4:
            return z, cur.z_rep
        return None

    @staticmethod
    def _next(tree: Tree):
        cand = [n for n in tree.nodes.values() if not n.visited and (n.score or 0) >= TAU_REL]
        return max(cand, key=lambda n: n.score) if cand else None

    def _apply(self, tree: Tree, cur: TNode, kids: list[TNode], j: dict, ep: Episode, crop, tiles, cand=None):
        ids = {f"C{k + 1}": c for k, c in enumerate(kids)}
        for cid, v in (j.get("scores") or {}).items():
            if cid in ids and isinstance(v, dict):
                ids[cid].score = float(v.get("relevance")) if _num(v.get("relevance")) else 0.0
                ids[cid].why = str(v.get("why", ""))[:200]
        for nt in j.get("notes") or []:
            if isinstance(nt, dict):
                key = str(nt.get("node", ""))
                (ids.get(key) or tree.nodes.get(key) or cur).notes.append(str(nt.get("text", ""))[:300])
        for e in j.get("evidence") or []:
            if not isinstance(e, dict):
                continue
            node = ids.get(str(e.get("child", ""))) or cur
            z = node.z_rep
            ti = _tile_index(e.get("tile"), tiles)
            if ti is not None:
                z = (tiles[ti][0] + tiles[ti][1]) // 2
            ci = _tile_index(e.get("tile"), cand) if cand else None
            crop_e = crop
            if ci is not None:            # evidence on a candidate crop: its own slice and crop
                z, crop_e = cand[ci]["z"], cand[ci]["crop"]
            if crop_e is not None and isinstance(e.get("box"), list) and len(e["box"]) == 4:
                box = V.crop_frac_to_box(crop_e, e["box"])
            elif ci is not None:
                box = cand[ci]["box"]
            else:
                box = tree.crop_box(node, margin=5, min_mm=20) if node.kind != "patch" else node.box[:4]
            st = tree.ix.box_nodes(box[0], box[1], box[2], box[3], z, z)
            tot = sum(st.values()) or 1
            st = {k: round(v / tot, 2) for k, v in sorted(st.items(), key=lambda t: -t[1])[:4]}
            if "liver_segments" in tree.case.layers:          # Couinaud segment(s) under the evidence box
                arr, names = tree.case.layers["liver_segments"]
                sub = arr[box[0]:box[1] + 1, box[2]:box[3] + 1, z]
                cnt = np.bincount(sub[sub > 0], minlength=len(names) + 1)
                if cnt.sum():
                    st.update({names[i - 1]: round(float(cnt[i] / cnt.sum()), 2) for i in np.argsort(-cnt)[:2] if cnt[i]})
            if tree.show_pct and slicepct.pct(tree.case.case_id, z) is not None:
                st["slice_pct"] = round(slicepct.pct(tree.case.case_id, z), 1)
            meas = None
            if (crop_e is not None and isinstance(e.get("box"), list) and len(e["box"]) == 4) or ci is not None:
                try:
                    m = CAND.measure_local(tree.case, z, box)
                    pc = slicepct.pct(tree.case.case_id, m["z_max_area"]) if (tree.show_pct and m.get("z_max_area") is not None) else None
                    meas = CAND.describe_measure(m, pc)
                except Exception as ex:          # measurement is best-effort
                    meas = f"code measurement failed ({type(ex).__name__})"
            ep.evidence.append(dict(id=len(ep.evidence) + 1, node=node.id, z=z, box_vox=box, structures=st,
                                    finding=str(e.get("finding", ""))[:300], supports=e.get("supports"),
                                    confidence=e.get("confidence"), win=tree.window(node), measure=meas))

    def _final(self, tree: Tree, ep: Episode, ctx: dict) -> dict:
        imgs, note = [], ""
        ev = sorted(ep.evidence, key=lambda e: -(float(e["confidence"]) if _num(e.get("confidence")) else 0))[:4]
        if self.final_mode != "text" and ev:
            pairs = []
            for e in ev:
                crop = V.pad_box(e["box_vox"], tree.case.shape, 15, tree.sp, 50)
                pairs.append(V.label(V.axial_crop(tree.case, e["z"], crop, e["win"], 320, mark_box=e["box_vox"]), f"E{e['id']} z={e['z']}"))
                if self.final_mode == "crops+locator":
                    pairs.append(V.label(V.locator(tree.case, e["z"], e["box_vox"], 320), f"E{e['id']} location"))
            imgs = [V.montage(pairs, 2)]
            note = ("IMAGES: evidence crops (box = evidence region)" +
                    (", each followed by the full axial slice showing its location." if self.final_mode == "crops+locator" else "."))
        r = self._call(ep, PROMPT_FINAL.format(graph=graph_text(tree, ep.evidence, self.use_graph), images_note=note, **ctx), imgs)
        return {**extract_json(r["text"]), "raw": r["text"][:500], "n_images": len(imgs)}
