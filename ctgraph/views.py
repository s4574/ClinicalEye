"""Agent-facing views: organ-cropped axial crops, montages, adjacent-slice triplets.

All images follow the radiological convention of render.py (axial: image top = anterior,
image left = patient RIGHT). Every image carries its slice index z so the VLM can refer to it.
Boxes passed around between the agent and the code are in voxel space (x0, x1, y0, y1).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import render as R
from .volume import Case

ORGAN_ALIASES = {
    "liver": ["liver"], "spleen": ["spleen"], "colon": ["colon"],
    "pancreas": ["pancreas"], "pancreatic": ["pancreas"],
    "kidney": ["kidney_left", "kidney_right"],
    "left kidney": ["kidney_left"], "right kidney": ["kidney_right"],
}
DEFAULT_WINDOW = {"liver": "liver"}
IMAGE_DIR = {"right": "image-left", "left": "image-right", "anterior": "image-up", "posterior": "image-down"}


def organ_structures(organ_field: str) -> list[str]:
    out = []
    for o in organ_field.split(","):
        out += ORGAN_ALIASES.get(o.strip(), [o.strip()])
    return list(dict.fromkeys(out))


def union_bbox(g: dict, names: list[str]) -> list[int]:
    bs = np.array([g["nodes"][n]["bbox_vox"] for n in names if n in g["nodes"]])
    return [int(bs[:, 0].min()), int(bs[:, 1].max()), int(bs[:, 2].min()), int(bs[:, 3].max()),
            int(bs[:, 4].min()), int(bs[:, 5].max())]


def pad_box(box, shape, margin_mm: float, spacing, min_mm: float = 60.0):
    """Pad an (x0, x1, y0, y1) voxel box by margin_mm and to at least min_mm per side; clip to volume."""
    out = []
    for ax in range(2):
        lo, hi = box[2 * ax], box[2 * ax + 1]
        m = int(round(margin_mm / spacing[ax]))
        lo, hi = lo - m, hi + m
        need = int(round(min_mm / spacing[ax])) - (hi - lo)
        if need > 0:
            lo, hi = lo - need // 2, hi + need - need // 2
        out += [max(lo, 0), min(hi, shape[ax] - 1)]
    return out


def box_to_rc(box, shape):
    """Voxel (x0, x1, y0, y1) -> axial pixel (r0, c0, r1, c1)."""
    X, Y = shape[0], shape[1]
    x0, x1, y0, y1 = box
    return (Y - 1 - y1, X - 1 - x1, Y - 1 - y0, X - 1 - x0)


def crop_frac_to_box(crop_box, frac):
    """A box given by the VLM as image fractions [left, top, right, bottom] of a crop -> voxel box."""
    x0, x1, y0, y1 = crop_box
    l, t, r, b = [float(np.clip(v, 0, 1)) for v in frac]
    l, r = min(l, r), max(l, r)
    t, b = min(t, b), max(t, b)
    # image left = high x, image top = high y
    nx0, nx1 = x1 - r * (x1 - x0), x1 - l * (x1 - x0)
    ny0, ny1 = y1 - b * (y1 - y0), y1 - t * (y1 - y0)
    return [int(round(nx0)), int(round(nx1)), int(round(ny0)), int(round(ny1))]


def label(img: Image.Image, text: str, size: int = 22) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = R._font(size)
    w = int(size * 0.62 * len(text)) + 8
    d.rectangle([0, 0, w, size + 6], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0), font=f)
    return img


def axial_crop(case: Case, z: int, box, win: str = "soft_tissue", size: int = 384,
               mark_box=None, mark_label: str = "") -> Image.Image:
    """Crop of axial slice z to voxel box (x0, x1, y0, y1), upscaled; optional marked sub-box."""
    z = int(np.clip(z, 0, case.shape[2] - 1))
    r0, c0, r1, c1 = box_to_rc(box, case.shape)
    full = R.to_rgb(R.axial(case.ct, z, win))
    img = full.crop((c0, r0, c1 + 1, r1 + 1))
    img, s = R.upscale(img, size)
    if mark_box is not None:
        mr0, mc0, mr1, mc1 = box_to_rc(mark_box, case.shape)
        img = R.draw_boxes(img, [(mr0 - r0, mc0 - c0, mr1 - r0, mc1 - c0)], [mark_label], s, width=2)
    return img


def locator(case: Case, z: int, box, size: int = 256) -> Image.Image:
    """Full axial slice, small, with the crop region outlined (where-am-I view)."""
    img, s = R.upscale(R.to_rgb(R.axial(case.ct, z)), size)
    return R.draw_boxes(img, [box_to_rc(box, case.shape)], [""], s, width=2)


def montage(imgs: list[Image.Image], ncol: int, bg=(40, 40, 40), gap: int = 4) -> Image.Image:
    w, h = max(i.size[0] for i in imgs), max(i.size[1] for i in imgs)
    nrow = (len(imgs) + ncol - 1) // ncol
    out = Image.new("RGB", (ncol * w + (ncol - 1) * gap, nrow * h + (nrow - 1) * gap), bg)
    for k, im in enumerate(imgs):
        out.paste(im, ((k % ncol) * (w + gap), (k // ncol) * (h + gap)))
    return out


def coronal_with_levels(case: Case, y: int, levels: list[int], size: int = 512) -> Image.Image:
    """Coronal slice (image top = superior, image left = patient right), resampled to physical aspect,
    with horizontal ticks + labels at the given axial levels."""
    y = int(np.clip(y, 0, case.shape[1] - 1))
    img = R.to_rgb(R.coronal(case.ct, y))
    sx, _, sz = case.spacing
    w_mm, h_mm = img.size[0] * sx, img.size[1] * sz
    s = size / max(w_mm, h_mm)
    img = img.resize((max(1, round(w_mm * s)), max(1, round(h_mm * s))), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    f = R._font(13)
    Z = case.shape[2]
    for i, z in enumerate(levels):
        r = (Z - 1 - z + 0.5) * sz * s
        x_txt = img.size[0] * (0.07 if i % 2 == 0 else 0.20)
        d.line([(0, r), (x_txt - 2, r)], fill=(255, 255, 0), width=2)
        d.text((x_txt, r - 8), f"z={z}", fill=(255, 255, 0), font=f)
    return img


def triplet(case: Case, z: int, box, d: int, win: str, size: int = 384, mark_box=None, mark_label="") -> Image.Image:
    """[z+d (superior), z, z-d (inferior)] side by side - the adjacent-slice context."""
    ims = []
    for zz, tag in ((z + d, "above"), (z, "CENTER"), (z - d, "below")):
        zz = int(np.clip(zz, 0, case.shape[2] - 1))
        im = axial_crop(case, zz, box, win, size, mark_box, mark_label)
        ims.append(label(im, f"z={zz} ({tag})"))
    return montage(ims, 3)


def image_frame_phrase(delta_mm_xy) -> str:
    """Patient-frame in-plane offset (x=right+, y=anterior+) -> phrase in axial image directions."""
    parts = []
    dx, dy = delta_mm_xy
    if abs(dy) >= 5:
        parts.append(f"{IMAGE_DIR['anterior' if dy > 0 else 'posterior']} {abs(dy):.0f} mm")
    if abs(dx) >= 5:
        parts.append(f"{IMAGE_DIR['right' if dx > 0 else 'left']} {abs(dx):.0f} mm")
    return ", ".join(parts) or "same position"
