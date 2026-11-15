"""Rendering in radiological display convention.

Axial:   image top = anterior,  image left = patient RIGHT.
Coronal: image top = superior,  image left = patient RIGHT.
Sagittal:image top = superior,  image left = anterior.
All helpers return the image plus a function mapping voxel coords -> pixel coords.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WINDOWS = {  # (level, width) HU
    "soft_tissue": (40, 400),
    "liver": (60, 160),
    "lung": (-600, 1500),
    "bone": (400, 1800),
    "vessel": (100, 600),
}

COLORS = [(255, 60, 60), (60, 200, 255), (255, 210, 0), (80, 255, 80), (255, 80, 255), (255, 140, 0)]


def window(hu: np.ndarray, name="soft_tissue") -> np.ndarray:
    """name: a WINDOWS key or an explicit (level, width) tuple."""
    level, width = name if isinstance(name, (tuple, list)) else WINDOWS[name]
    return (np.clip((hu - (level - width / 2)) / width, 0, 1) * 255).astype(np.uint8)


def axial(ct: np.ndarray, z: int, win: str = "soft_tissue") -> np.ndarray:
    """(Y, X) uint8 image; pixel (r, c) <- voxel (X-1-c, Y-1-r, z)."""
    return window(ct[::-1, ::-1, z].T, win)


def axial_xy_to_rc(x, y, shape_xyz):
    X, Y, _ = shape_xyz
    return (Y - 1 - y, X - 1 - x)


def coronal(ct: np.ndarray, y: int, win: str = "soft_tissue") -> np.ndarray:
    return window(ct[::-1, y, ::-1].T, win)


def sagittal(ct: np.ndarray, x: int, win: str = "soft_tissue") -> np.ndarray:
    return window(ct[x, ::-1, ::-1].T, win)


def to_rgb(g: np.ndarray) -> Image.Image:
    return Image.fromarray(np.stack([g] * 3, -1))


def upscale(img: Image.Image, longest: int = 768) -> tuple[Image.Image, float]:
    s = longest / max(img.size)
    return img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BILINEAR), s


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default(size=size)


def draw_boxes(img: Image.Image, boxes_rc: list[tuple[float, float, float, float]], labels: list[str],
               scale: float = 1.0, width: int = 3) -> Image.Image:
    """boxes in unscaled pixel coords (r0, c0, r1, c1)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = _font(26)
    for i, ((r0, c0, r1, c1), lab) in enumerate(zip(boxes_rc, labels)):
        col = COLORS[i % len(COLORS)]
        d.rectangle([c0 * scale, r0 * scale, c1 * scale, r1 * scale], outline=col, width=width)
        if not lab:
            continue
        tx, ty = c0 * scale + 2, max(r0 * scale - 32, 0)
        d.rectangle([tx - 2, ty, tx + 16 * len(lab) + 6, ty + 30], fill=(0, 0, 0))
        d.text((tx + 1, ty + 1), lab, fill=col, font=f)
    return img


def mask_bbox_rc(mask2d_rc: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.nonzero(mask2d_rc)
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def axial_mask(labels: np.ndarray, z: int, k: int) -> np.ndarray:
    return (labels[::-1, ::-1, z].T == k)


def overlay_mask(img: Image.Image, mask_rc: np.ndarray, color=(255, 60, 60), alpha: float = 0.35,
                 scale: float = 1.0) -> Image.Image:
    """Semi-transparent fill + 1px contour of a 2D mask (unscaled pixel grid) on an upscaled image."""
    from scipy import ndimage
    m = Image.fromarray((mask_rc * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
    m = np.asarray(m) > 0
    edge = m & ~ndimage.binary_erosion(m, iterations=2)
    a = np.asarray(img).astype(np.float32).copy()
    a[m] = a[m] * (1 - alpha) + np.array(color) * alpha
    a[edge] = color
    return Image.fromarray(a.astype(np.uint8))


def b64png(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()
