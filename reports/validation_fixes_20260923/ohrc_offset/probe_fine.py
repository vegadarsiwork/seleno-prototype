"""Instrumented control run: what the fine stage's correspondences are made of.

Wraps register._fine_stage to capture the coarse model it prewarps with and the
native correspondences it returns, then compares coarse, final and TRUE models.
register.py itself is not modified.

    OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/ohrc_offset/probe_fine.py [control_name ...]
"""
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
import importlib
RG = importlib.import_module("seleno.tool.register")   # seleno.tool re-exports a function of this name
from seleno.tool import warp_model as WM
from seleno.tool.coordinates import grid_to_reference, project

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "diagnostics"
OUT = ROOT / "outputs/validation_fixes_20260923/probe"
OPTIONS = {"max_side": 2048, "grid": (12, 12), "segments": 6}


def probe(c):
    cap = {}
    original = RG._fine_stage

    def fine(*a, **k):
        cap["Hcoarse"] = np.asarray(k.get("Hcoarse"), float)
        out = original(*a, **k)
        cap["fcorr"], cap["info"] = out
        return out
    RG._fine_stage = fine
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = RG.register(str(ROOT / c["source"]), str(ROOT / c["reference"]),
                              str(OUT / c["name"]), **OPTIONS)
    finally:
        RG._fine_stage = original
    model = json.loads((Path(res.out_dir) / "transform.json").read_text())
    frame = model["frame"]
    G = grid_to_reference(frame)
    step = frame["reference_decimation"]
    Hc = np.eye(3)
    Hc[:2, :] = cap["Hcoarse"][:2, :]
    Hc_native = G @ Hc @ np.linalg.inv(G)
    P = np.linalg.inv(Hc_native)                 # the fine stage's prewarp, native px
    f = cap["fcorr"]
    window_src = project(np.linalg.inv(P), f.src)   # where the matcher put the source point
    d = window_src - f.ref                         # the matcher's own measured displacement
    zero = np.hypot(*d.T) < 1e-9
    integer = np.all(np.abs(d - np.round(d)) < 1e-9, axis=1)
    Tinv = np.asarray(c["truth"]["T_inverse_native_ref_px"], float)
    Hf_native = np.asarray(model["matrix_reference_px"], float)
    # compare models over the fine points' own reference positions
    pts = f.ref

    def gap(H1, H2):
        v = np.hypot(*(project(H1, pts) - project(H2, pts)).T)
        return {"median": float(np.median(v)), "max": float(v.max())}
    ev = json.loads((Path(res.out_dir) / "evaluation.json").read_text())
    e = WM.residuals(model, np.array(ev["source"]), np.array(ev["reference"]))
    return {"name": c["name"], "status": res.status, "fine_method": cap["info"].get("method"),
            "fine_points": int(len(f)), "reference_decimation": step,
            "matcher_displacement_exactly_zero": float(zero.mean()),
            "matcher_displacement_whole_pixels": float(integer.mean()),
            "coarse_vs_truth_ref_px": gap(Hc_native, Tinv),
            "final_vs_truth_ref_px": gap(Hf_native, Tinv),
            "final_vs_coarse_ref_px": gap(Hf_native, Hc_native),
            "held_out_exactly_on_model": float(np.mean(e < 1e-6))}


def main():
    names = sys.argv[1:] or ["control_ohrc_nac", "control_ohrc_nac_km", "control_tmc2_selene"]
    controls = {c["name"]: c for c in json.loads((DIAG / "controls.json").read_text())}
    out = []
    for n in names:
        print("PROBE", n, flush=True)
        out.append(probe(controls[n]))
        print(json.dumps(out[-1], indent=1), flush=True)
    (HERE / "probe_fine.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
