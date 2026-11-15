"""Untooled-structure recognition probe (design: runs/design_structure_probe_1114.md, approved by the user 2026-11-15).
Can a VLM find a structure that TotalSegmentator does not segment, and do anatomical priors help?
  build: 5 structures x 25 items (one axial slice per case); 160 mm crop centred on the TotalSeg pancreas; four candidate
         points A-D (one on the GT structure); GT masks only for building / scoring, never shown.
  run:   conditions C0 plain / C1 + TotalSeg anatomy map (segmented structures outlined + labelled) / C2 = C1 + our anatomy
         KB entry; temperature 0. Outputs runs/structure_probe/.
Usage: python3 scripts/structure_probe.py build
       python3 scripts/structure_probe.py run <model> [workers]
       python3 scripts/structure_probe.py score"""
import base64, io, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from ctgraph import paths as P  # noqa: E402
OUT = P.RUNS / "structure_probe"
STRUCTS = {"superior_mesenteric_artery": "superior mesenteric artery", "celiac_trunk": "celiac trunk",
           "common_bile_duct": "common bile duct", "pancreas_head": "pancreatic head", "pancreas_tail": "pancreatic tail"}
N_PER = 25
FOV_MM, PX = 160.0, 512
COLS = [(255, 220, 0), (0, 200, 255), (255, 120, 0), (120, 255, 120), (255, 0, 200), (200, 200, 255), (255, 255, 255), (0, 255, 180),
        (170, 110, 255), (255, 160, 160), (140, 200, 0), (0, 130, 255), (255, 255, 140), (180, 120, 60)]


def cases():
    ids = json.load(open(P.SPLITS / "cases_all.json"))["ids"]
    ok = lambda c: all((P.TOTALSEG_OUT / c / f"{t}.nii.gz").exists() for t in ("total", "liver_vessels", "liver_segments"))
    return sorted({c for c in ids if ok(c)})


def crop_xy(cx, cy, sp, shape):
    hx, hy = int(round(FOV_MM / 2 / sp[0])), int(round(FOV_MM / 2 / sp[1]))
    x0, y0 = max(0, cx - hx), max(0, cy - hy)
    x1, y1 = min(shape[0] - 1, x0 + 2 * hx), min(shape[1] - 1, y0 + 2 * hy)
    return x0, x1, y0, y1


def to_img(a2d):
    """canonical (x, y) slice -> display image (image left = patient right, top = anterior), as in our renderer"""
    return a2d[::-1, ::-1].T


def build_one(args):
    cid, sname, seed = args
    from ctgraph.volume import load_ts_case, _canon, GT_DIR
    f = GT_DIR / cid / "segmentations" / f"{sname}.nii.gz"
    if not f.exists():
        return None
    gt = _canon(f)[0] > 0.5
    if gt.sum() < 20:
        return None
    case = load_ts_case(cid)
    if "pancreas" not in case.names:
        return None
    pan = case.labels == case.name_to_id("pancreas")
    pz = np.nonzero(pan.any((0, 1)))[0]
    area = gt.sum((0, 1))
    if sname in ("superior_mesenteric_artery", "celiac_trunk", "common_bile_duct") and len(pz):
        area = area * ((np.arange(len(area)) >= pz.min()) & (np.arange(len(area)) <= pz.max()))
    z = int(np.argmax(area))
    if area[z] < 5:
        return None
    pm = pan[:, :, z] if pan[:, :, z].any() else pan.any(2)
    cx, cy = [int(v) for v in np.argwhere(pm).mean(0)]
    sp = case.spacing
    x0, x1, y0, y1 = crop_xy(cx, cy, sp, case.shape)
    g2 = gt[x0:x1 + 1, y0:y1 + 1, z]
    if g2.sum() < 5:
        return None
    ct = case.ct[x0:x1 + 1, y0:y1 + 1, z].astype(np.float32)
    rng = random.Random(seed)
    pts_t = np.argwhere(g2)
    c = pts_t.mean(0)
    tgt = pts_t[np.argmin(((pts_t - c) ** 2).sum(1))]
    body = np.argwhere((ct > -200) & ~g2)
    mm = lambda a, b: float(np.sqrt((((np.array(a) - np.array(b)) * np.array(sp[:2])) ** 2).sum()))
    pts = [tuple(int(v) for v in tgt)]
    for _ in range(5000):
        if len(pts) == 4:
            break
        p = tuple(int(v) for v in body[rng.randrange(len(body))])
        if all(mm(p, q) >= 15 for q in pts):
            pts.append(p)
    if len(pts) < 4:
        return None
    import zlib
    order = list(range(4)); random.Random(zlib.crc32(f"{sname}__{cid}".encode())).shuffle(order)   # per-item letter order (fixed 2026-11-15: per-chunk seeds repeated)
    letters = {"ABCD"[i]: pts[order[i]] for i in range(4)}
    answer = "ABCD"[order.index(0)]
    lab = case.labels[x0:x1 + 1, y0:y1 + 1, z]
    names = {}
    for i in np.unique(lab[lab > 0]):
        if (lab == i).mean() >= 0.005:
            names[case.names[i - 1]] = int(i)
    return dict(id=f"{sname}__{cid}", case=cid, structure=sname, z=z, crop=[x0, x1, y0, y1], points=letters, answer=answer,
                ts_structs=names, spacing=list(map(float, sp)))


def render(item, cond):
    from ctgraph.volume import load_ts_case
    from scipy import ndimage
    case = load_ts_case(item["case"])
    x0, x1, y0, y1 = item["crop"]; z = item["z"]
    ct = case.ct[x0:x1 + 1, y0:y1 + 1, z].astype(np.float32)
    w = np.clip((ct - (50 - 200)) / 400.0, 0, 1)
    im = Image.fromarray((to_img(w) * 255).astype(np.uint8)).convert("RGB")
    s = PX / max(im.size)
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    H, W = ct.shape   # display: rows = y reversed, cols = x reversed
    tp = lambda x, y: ((H - 1 - x) * s, (W - 1 - y) * s)     # canonical (x, y) in crop -> display (col, row)
    legend = []
    if cond in ("C1", "C2"):
        lab = case.labels[x0:x1 + 1, y0:y1 + 1, z]
        for k, (n, i) in enumerate(sorted(item["ts_structs"].items())):
            col = COLS[k % len(COLS)]
            m = lab == i
            for px, py in np.argwhere(m & ~ndimage.binary_erosion(m)):      # boundary pixels
                c_, r_ = tp(px, py)
                d.rectangle([c_ - 1, r_ - 1, c_ + 1, r_ + 1], fill=col)
            legend.append((n.replace("_", " ")[:28], col))
    # v2 (2026-11-15): labels in a legend strip below the image, never over the anatomy (v1 drew them at the structure
    # centroids, where they could hide a candidate point, e.g. "inferior vena cava" over the SMA point)
    for L, (px, py) in item["points"].items():
        c, r = tp(px, py)
        d.ellipse([c - 7, r - 7, c + 7, r + 7], outline=(255, 0, 0), width=3)
        d.text((c + 9, r - 9), L, fill=(255, 40, 40), font=font)
    if legend:
        rows = (len(legend) + 1) // 2
        out = Image.new("RGB", (im.size[0], im.size[1] + 22 * rows + 6), (0, 0, 0))
        out.paste(im, (0, 0)); d2 = ImageDraw.Draw(out)
        for k, (n, col) in enumerate(legend):
            x, y = 6 + (k % 2) * (im.size[0] // 2), im.size[1] + 4 + 22 * (k // 2)
            d2.rectangle([x, y + 4, x + 14, y + 16], outline=col, width=3); d2.text((x + 20, y), n, fill=col, font=font)
        return out
    return im


PROMPT = ("This is one axial contrast-enhanced CT slice of the abdomen (display: image left = patient right, image top = "
          "anterior). Four candidate points are marked A, B, C, D. Which point lies on the {name}?{c1}{c2}\n"
          'Reply with STRICT JSON only: {{"answer": "<A|B|C|D>", "confidence": <0-1>}}')
C1 = ("\nSegmented structures are outlined in colour (automatic segmentation, can be imperfect; colour key below the "
      "image); the {name} itself is not segmented.")


def text_map(item, points=False):
    """C3 / C4 (2026-11-15): the anatomy map as TEXT (no overlay): segmented structures in the image with share and image
    position; C4 adds per-point code facts (inside which segmented structure / nearest one and distance)."""
    from ctgraph.volume import load_ts_case
    from scipy import ndimage
    case = load_ts_case(item["case"])
    x0, x1, y0, y1 = item["crop"]; z = item["z"]
    lab = case.labels[x0:x1 + 1, y0:y1 + 1, z]
    H, W = lab.shape
    rows = []
    for n, i in sorted(item["ts_structs"].items(), key=lambda kv: -(lab == kv[1]).sum()):
        xs, ys = np.nonzero(lab == i)
        col, row = 1 - xs.mean() / max(H - 1, 1), 1 - ys.mean() / max(W - 1, 1)
        h = "left" if col < 0.33 else "right" if col > 0.67 else "centre"
        v = "top" if row < 0.33 else "bottom" if row > 0.67 else "middle"
        rows.append(f"{n.replace('_', ' ')} ({(lab == i).mean() * 100:.0f}% of the image, {v}-{h})")
    txt = "\nSegmented structures in this image (automatic segmentation, can be imperfect): " + "; ".join(rows) + "."
    if points:
        facts = []
        for L, (px, py) in sorted(item["points"].items()):
            v = int(lab[px, py])
            if v and case.names[v - 1] in item["ts_structs"]:
                facts.append(f"point {L} lies inside the segmented {case.names[v - 1].replace('_', ' ')}")
            else:
                best = None
                for n, i in item["ts_structs"].items():
                    d = ndimage.distance_transform_edt(lab != i)[px, py] * item["spacing"][0]
                    best = min(best, (d, n)) if best else (d, n)
                facts.append(f"point {L} lies outside all segmented structures" +
                             (f" (nearest: {best[1].replace('_', ' ')}, {best[0]:.0f} mm)" if best else ""))
        txt += "\nCode-computed position of each point: " + "; ".join(facts) + "."
    return txt


def run(model, workers, conds=("C0", "C1", "C2")):
    from ctgraph.llm import LLM, user_msg, extract_json
    from ctgraph.navi5 import anatomy_lookup
    items = json.load(open(OUT / "items.json"))
    llm = LLM(model, tag="structure_probe")
    od = OUT / model; od.mkdir(parents=True, exist_ok=True)

    def job(arg):
        it, cond = arg
        f = od / f"{it['id']}__{cond}.json"
        if f.exists():
            return json.load(open(f))
        name = STRUCTS[it["structure"]]
        base = cond.rstrip("r")          # "C0r" / "C1r": same as C0 / C1 but the reply also carries a short reason
        if base in ("C0", "C1", "C2"):
            c1 = C1.format(name=name) if base in ("C1", "C2") else ""
        elif base == "C1n":              # overlay, but without the sentence "the X itself is not segmented"
            c1 = "\nSegmented structures are outlined in colour (automatic segmentation, can be imperfect; colour key below the image)."
        else:                            # C3 / C4 / C5: no overlay, anatomy map as text
            c1 = text_map(it, points=base in ("C4", "C5"))
        c2 = ("\nAnatomy reference: " + anatomy_lookup(name, k=1)[:1500]) if base in ("C2", "C5") else ""
        txt = PROMPT.format(name=name, c1=c1, c2=c2)
        if cond.endswith("r"):
            txt = txt.replace('{"answer": "<A|B|C|D>"', '{"reason": "<1-2 sentences: what you see at the points>", "answer": "<A|B|C|D>"')
        img_cond = base if base in ("C1", "C2", "C1n") else "C0"
        buf = io.BytesIO(); render(it, "C1" if img_cond == "C1n" else img_cond).save(buf, format="PNG")
        r = llm.chat([user_msg(txt, [base64.b64encode(buf.getvalue()).decode()])], max_tokens=300 if cond.endswith("r") else 100,
                     temperature=0.0)
        j = extract_json(r["text"]) or {}
        pred = str(j.get("answer", "")).strip().upper()[:1]
        rec = dict(id=it["id"], structure=it["structure"], cond=cond, pred=pred, answer=it["answer"], correct=pred == it["answer"],
                   conf=j.get("confidence"), raw=r["text"][:200])
        json.dump(rec, open(f, "w"))
        return rec
    jobs = [(it, c) for it in items for c in conds]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(job, jobs))
    print(model, len(res), "done")


def score():
    from math import comb
    mcn = lambda b, c: 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
    for md in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "img"):
        recs = [json.load(open(f)) for f in md.glob("*.json")]
        by = {}
        for r in recs:
            by.setdefault((r["structure"], r["cond"]), []).append(r["correct"])
        print(f"== {md.name}")
        for s in list(STRUCTS) + ["ALL"]:
            row = []
            for c in ("C0", "C1", "C2", "C1n", "C3", "C4", "C5", "C0r", "C1r"):
                v = [x for (ss, cc), xs in by.items() if cc == c and (s == "ALL" or ss == s) for x in xs]
                row.append(f"{c} {np.mean(v):.2f} ({len(v)})" if v else f"{c} -")
            print(f"  {s:28} " + "  ".join(row))
        D = {(r["id"], r["cond"]): r["correct"] for r in recs}
        ids = sorted({r["id"] for r in recs})
        for a, b in (("C1", "C0"), ("C2", "C1"), ("C2", "C0"), ("C1n", "C1"), ("C3", "C0"), ("C4", "C0"), ("C5", "C4"), ("C0r", "C0"), ("C1r", "C1")):
            x = sum(D.get((i, a), False) and not D.get((i, b), False) for i in ids)
            y = sum(D.get((i, b), False) and not D.get((i, a), False) for i in ids)
            print(f"  {a} vs {b}: +{x}/-{y} p={mcn(x, y):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        OUT.mkdir(parents=True, exist_ok=True)
        cs = cases(); rng = random.Random(2026)
        items = []
        for sname in STRUCTS:
            order = cs[:]; rng.shuffle(order)
            got = []
            with Pool(4) as p:
                for chunk in range(0, len(order), 12):
                    for it in p.map(build_one, [(c, sname, 2026 + k) for k, c in enumerate(order[chunk:chunk + 12])]):
                        if it and len(got) < N_PER:
                            got.append(it)
                    if len(got) >= N_PER:
                        break
            print(sname, len(got)); items += got
        json.dump(items, open(OUT / "items.json", "w"))
        (OUT / "img").mkdir(exist_ok=True)
        for it in items[::25]:
            for c in ("C0", "C1"):
                render(it, c).save(OUT / "img" / f"{it['id']}__{c}.png")
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4,
            tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("C0", "C1", "C2"))
    else:
        score()
