"""Out-of-plane localisation probe (user, 2026-11-15): the target structure is NOT in the shown axial slice; the model answers
(1) direction: is it SUPERIOR or INFERIOR to this slice, (2) distance from the slice to the nearest edge of the structure:
A <= 3 cm (about one lumbar vertebral level), B 3-6 cm, C > 6 cm. GT masks are used only to build / score items.
Targets: untooled structures (GT only: SMA, celiac trunk, common bile duct, pancreatic head / tail); low-priority variant
"masked": TotalSeg organs (pancreas, kidneys) removed from all code facts.
Conditions: C0 image only; C3 + text map of the slice (segmented structures present, image position, vertebrae);
C4 + code facts on how far the segmented neighbours extend above / below this slice (never the target); C5 = C4 + KB entry.
Usage: python3 scripts/zprobe.py build [masked] | run <model> [workers] [masked] | score"""
import base64, io, json, random, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
from ctgraph import paths as P  # noqa: E402
OUT = P.RUNS / "zprobe"
UNTOOLED = {"superior_mesenteric_artery": "superior mesenteric artery", "celiac_trunk": "celiac trunk",
            "common_bile_duct": "common bile duct", "pancreas_head": "pancreatic head", "pancreas_tail": "pancreatic tail"}
MASKED = {"pancreas": ("pancreas", ["pancreas"]),
          "kidney_left": ("left kidney", ["kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right"]),
          # same organ set as the QA organ-masking experiment (mask540 + extension mask_ext363), 2026-11-15
          "liver": ("liver", ["liver"]), "spleen": ("spleen", ["spleen"]), "colon": ("colon", ["colon"])}
NEIGH = ["liver", "spleen", "stomach", "pancreas", "kidney_left", "kidney_right", "gallbladder", "duodenum", "aorta",
         "inferior_vena_cava", "portal_vein_and_splenic_vein", "adrenal_gland_left", "adrenal_gland_right", "colon", "small_bowel",
         "vertebrae_T11", "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
BINS = {"A": (10, 30), "B": (30, 60), "C": (60, 100)}       # mm from the slice to the nearest edge of the target
N_PER = 25


def cases():
    from structure_probe import cases as c
    return c()


def build_one(args):
    cid, tname, masked = args
    from ctgraph.volume import load_ts_case, _canon, GT_DIR
    case = load_ts_case(cid)
    if masked:
        if tname not in case.names:
            return None
        m = case.labels == case.name_to_id(tname)
    else:
        f = GT_DIR / cid / "segmentations" / f"{tname}.nii.gz"
        if not f.exists():
            return None
        m = _canon(f)[0] > 0.5
    if m.sum() < 20:
        return None
    zz = np.nonzero(m.any((0, 1)))[0]; zlo, zhi = int(zz.min()), int(zz.max())
    sz = float(case.spacing[2]); Z = case.shape[2]
    rng = random.Random(zlib.crc32(f"{tname}__{cid}".encode()))
    for _ in range(20):
        b = rng.choice("ABC"); d = rng.uniform(*BINS[b]); up = rng.random() < 0.5     # up: slice ABOVE the target
        k = int(round(d / sz))
        z = zhi + k if up else zlo - k
        if 0 <= z < Z and not m[:, :, z].any():
            dist = (z - zhi if up else zlo - z) * sz
            bb = "A" if dist <= 30 else "B" if dist <= 60 else "C"
            return dict(id=f"{tname}__{cid}", case=cid, target=tname, masked=masked, z=int(z), dist_mm=round(dist, 1),
                        direction="inferior" if up else "superior", dist_bin=bb, spacing=list(map(float, case.spacing)))
    return None


def render(item):
    from ctgraph.volume import load_ts_case
    case = load_ts_case(item["case"])
    sl = case.ct[:, :, item["z"]].astype(np.float32)
    w = np.clip((sl - (50 - 200)) / 400.0, 0, 1)
    im = Image.fromarray((w[::-1, ::-1].T * 255).astype(np.uint8)).convert("RGB")
    s = 512 / max(im.size)
    return im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)


def facts(item, level):
    """level 3: structures in this slice (+ image position); level 4: + extent of segmented neighbours relative to the slice"""
    from ctgraph.volume import load_ts_case
    case = load_ts_case(item["case"]); z = item["z"]; sz = item["spacing"][2]
    drop = set(MASKED[item["target"]][1]) if item["masked"] else set()
    names = [n for n in case.names if n not in drop]
    lab = case.labels[:, :, z]; H, W = lab.shape
    rows = []
    for n in names:
        i = case.name_to_id(n); mk = lab == i
        if mk.mean() < 0.002:
            continue
        xs, ys = np.nonzero(mk)
        col, row = 1 - xs.mean() / (H - 1), 1 - ys.mean() / (W - 1)
        rows.append((mk.sum(), f"{n.replace('_', ' ')} ({'left' if col < .33 else 'right' if col > .67 else 'centre'}"
                                f"-{'top' if row < .33 else 'bottom' if row > .67 else 'middle'})"))
    txt = "\nSegmented structures in this slice (automatic segmentation, can be imperfect): " + \
          "; ".join(r for _, r in sorted(rows, reverse=True)[:14]) + "."
    if level >= 4:
        rel = []
        for n in NEIGH:
            if n in drop or n not in case.names:
                continue
            zz = np.nonzero((case.labels == case.name_to_id(n)).any((0, 1)))[0]
            if not len(zz):
                continue
            lo, hi = (zz.min() - z) * sz / 10, (zz.max() - z) * sz / 10      # cm, + = superior to this slice
            if lo <= 0 <= hi:
                rel.append(f"{n.replace('_', ' ')} spans this slice (from {abs(lo):.1f} cm below to {hi:.1f} cm above)")
            elif lo > 0:
                rel.append(f"{n.replace('_', ' ')} lies entirely ABOVE this slice ({lo:.1f}-{hi:.1f} cm above)")
            else:
                rel.append(f"{n.replace('_', ' ')} lies entirely BELOW this slice ({abs(hi):.1f}-{abs(lo):.1f} cm below)")
        txt += "\nCode-computed craniocaudal extent of segmented structures relative to this slice: " + "; ".join(rel) + "."
    return txt


PROMPT = ("This is one axial contrast-enhanced CT slice of the abdomen (display: image left = patient right, image top = "
          "anterior). The {name} is NOT visible in this slice.{facts}{kb}\n"
          "Answer two questions: (1) is the {name} SUPERIOR (toward the head) or INFERIOR (toward the feet) to this slice? "
          "(2) how far is its nearest part from this slice? A: within 3 cm (about one lumbar vertebral level), B: 3-6 cm, C: more than 6 cm.\n"
          'Reply with STRICT JSON only: {{"direction": "<superior|inferior>", "distance": "<A|B|C>", "confidence": <0-1>}}')


def run(model, workers, masked):
    from ctgraph.llm import LLM, user_msg, extract_json
    from ctgraph.navi5 import anatomy_lookup
    items = json.load(open(OUT / f"items{'_masked' if masked else ''}.json"))
    llm = LLM(model, tag="zprobe")
    od = OUT / (model + ("_masked" if masked else "")); od.mkdir(parents=True, exist_ok=True)

    def job(a):
        it, cond = a
        f = od / f"{it['id']}__{cond}.json"
        if f.exists():
            return
        name = MASKED[it["target"]][0] if masked else UNTOOLED[it["target"]]
        lvl = {"C0": 0, "C3": 3, "C4": 4, "C5": 4}[cond]
        txt = PROMPT.format(name=name, facts=facts(it, lvl) if lvl else "",
                            kb=("\nAnatomy reference: " + anatomy_lookup(name, k=1)[:1500]) if cond == "C5" else "")
        buf = io.BytesIO(); render(it).save(buf, format="PNG")
        r = llm.chat([user_msg(txt, [base64.b64encode(buf.getvalue()).decode()])], max_tokens=100, temperature=0.0)
        j = extract_json(r["text"]) or {}
        dr = str(j.get("direction", "")).strip().lower(); db = str(j.get("distance", "")).strip().upper()[:1]
        json.dump(dict(id=it["id"], target=it["target"], cond=cond, direction=dr, dist=db, gt_dir=it["direction"], gt_bin=it["dist_bin"],
                       ok_dir=dr == it["direction"], ok_dist=db == it["dist_bin"], conf=j.get("confidence"), raw=r["text"][:200]), open(f, "w"))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, [(it, c) for it in items for c in ("C0", "C3", "C4", "C5")]))


def score():
    from math import comb
    mcn = lambda b, c: 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
    for md in sorted(p for p in OUT.iterdir() if p.is_dir()):
        recs = [json.load(open(f)) for f in md.glob("*.json")]
        if not recs:
            continue
        print(f"== {md.name}  (direction chance 0.50, distance chance 0.33)")
        tg = sorted({r["target"] for r in recs})
        for t in tg + ["ALL"]:
            row = []
            for c in ("C0", "C3", "C4", "C5"):
                v = [r for r in recs if r["cond"] == c and (t == "ALL" or r["target"] == t)]
                if v:
                    row.append(f"{c} dir {np.mean([r['ok_dir'] for r in v]):.2f} dist {np.mean([r['ok_dist'] for r in v]):.2f}")
            print(f"  {t:28} " + " | ".join(row))
        D = {(r["id"], r["cond"]): r for r in recs}; ids = sorted({r["id"] for r in recs})
        for a, b in (("C3", "C0"), ("C4", "C0"), ("C4", "C3"), ("C5", "C4")):
            for k in ("ok_dir", "ok_dist"):
                x = sum(D.get((i, a), {}).get(k, False) and not D.get((i, b), {}).get(k, False) for i in ids)
                y = sum(D.get((i, b), {}).get(k, False) and not D.get((i, a), {}).get(k, False) for i in ids)
                print(f"  {a} vs {b} {k}: +{x}/-{y} p={mcn(x, y):.3f}")


if __name__ == "__main__":
    masked = "masked" in sys.argv
    if sys.argv[1] == "build":
        OUT.mkdir(parents=True, exist_ok=True)
        cs = cases(); targets = MASKED if masked else UNTOOLED; items = []
        for t in targets:
            order = cs[:]; random.Random(2026).shuffle(order); got = []
            with Pool(4) as p:
                for i in range(0, len(order), 12):
                    for it in p.map(build_one, [(c, t, masked) for c in order[i:i + 12]]):
                        if it and len(got) < N_PER:
                            got.append(it)
                    if len(got) >= N_PER:
                        break
            print(t, len(got)); items += got
        from collections import Counter
        print(Counter(i["direction"] for i in items), Counter(i["dist_bin"] for i in items))
        json.dump(items, open(OUT / f"items{'_masked' if masked else ''}.json", "w"))
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 4, masked)
    else:
        score()
