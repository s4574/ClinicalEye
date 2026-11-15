"""Adult normal reference ranges for segmented structures (node annotations of the anatomical graph).

Values are typical adult CT ranges from radiology references; they vary with body size, so they are
reported as soft flags ("above typical range"), never as diagnoses.
"""
from __future__ import annotations

# name: (volume_ml (lo, hi), max_diameter_mm hi, note)
NORMAL = {
    "liver": ((1000, 1900), 210, "craniocaudal span usually < 16-17 cm; larger with body size"),
    "spleen": ((80, 350), 140, "length usually < 12-13 cm; volume > ~500 mL suggests splenomegaly"),
    "kidney_left": ((100, 260), 135, "length usually 10-13 cm; left often slightly larger than right"),
    "kidney_right": ((100, 260), 135, "length usually 10-13 cm"),
    "pancreas": ((40, 130), 200, "head <= ~3 cm, body/tail <= ~2.5 cm thick; atrophies with age"),
    "gall_bladder": ((5, 80), 110, "distended when fasting; wall should be thin"),
}


def flags(g: dict, name: str) -> str:
    n, ref = g["nodes"].get(name), NORMAL.get(name)
    if n is None or ref is None:
        return ""
    (lo, hi), dmax, note = ref
    out = []
    if n["volume_ml"] > hi:
        out.append(f"volume ABOVE typical adult range {lo}-{hi} mL")
    elif n["volume_ml"] < lo:
        out.append(f"volume below typical adult range {lo}-{hi} mL")
    else:
        out.append(f"volume within typical adult range {lo}-{hi} mL")
    if n["max_diameter_mm"] > dmax:
        out.append(f"max diameter above typical (<= {dmax} mm)")
    if name.startswith("kidney_"):
        other = "kidney_right" if name == "kidney_left" else "kidney_left"
        if other in g["nodes"]:
            r = n["volume_ml"] / max(g["nodes"][other]["volume_ml"], 1e-6)
            if r > 1.3:
                out.append(f"{r:.1f}x the volume of the {other.replace('_', ' ')} (asymmetric)")
    return "; ".join(out) + f" [{note}]"
