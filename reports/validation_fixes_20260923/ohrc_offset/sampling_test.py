"""Where does the OHRC control's ~0.4-0.6 reference px offset come from?

Hypothesis under test: the source block-averaging path (active at ~4.17x source
px per reference px) samples at the block CORNER instead of its centre, which
would shift every sample by (f - 1) / 2 = 1.58 source px = 0.38 reference px.

Test A (sampling alone): the real source geometry, but the pixel values replaced
by a linear ramp (value = sample index + 1000, or line index + 1000). prealign()
puts that onto a native reference window; if sampling is centred, every output
value equals the source coordinate recorded for that pixel in back_x / back_y.

Test B (placement before any matching): prealign the real source through the
KNOWN control transform (prewarp = T) and measure its shift against the synthetic
reference with the calibrated estimator from diagnostics/analyse.py. Zero means
sampling, placement and the synthetic reference agree, and the offset arises
later (matching or fitting).

    OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
      reports/validation_fixes_20260923/ohrc_offset/sampling_test.py
"""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool.register import prealign
from seleno.tool.scene import load

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "diagnostics"
spec = importlib.util.spec_from_file_location("analyse", DIAG / "analyse.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
SIDE = 512


def windows(path, n=5, k=16):
    """n native SIDE x SIDE windows fully inside the synthetic reference's valid
    area, spread evenly along it, as (r0, c0, r1, c1)."""
    import cv2
    import rasterio
    with rasterio.open(path) as ds:
        small = ds.read(1, out_shape=(ds.height // k, ds.width // k)) > 0
        H, W = ds.height, ds.width
    inner = cv2.erode(small.astype(np.uint8), np.ones((SIDE // k + 3,) * 2, np.uint8)).astype(bool)
    ys, xs = np.nonzero(inner)
    order = np.argsort(ys * inner.shape[1] + xs)
    pick = order[np.linspace(0, len(order) - 1, n).round().astype(int)]
    out = []
    for i in pick:                       # kept inside the grid, so every window is SIDE x SIDE
        r0 = int(np.clip(ys[i] * k - SIDE // 2, 0, H - SIDE))
        c0 = int(np.clip(xs[i] * k - SIDE // 2, 0, W - SIDE))
        out.append((r0, c0, r0 + SIDE, c0 + SIDE))
    return out


def ramp_test(S, R, wins):
    rows = []
    h, w = S.array.shape
    for axis, name in ((1, "sample"), (0, "line")):
        idx = (np.arange(w) if axis == 1 else np.arange(h)[:, None]).astype(np.float32) + 1000
        ramp = dataclasses.replace(S, array=np.broadcast_to(idx, (h, w)), nodata=None)
        for win in wins:
            a, am, _, _, bx, by, info = prealign(ramp, R, max_side=2048, window=win)
            if info["reference_decimation"] != 1 or am.sum() < 1000:
                continue
            truth = (bx if axis == 1 else by) + 1000
            d = (a - truth)[am & np.isfinite(truth)]
            rows.append({"axis": name, "window": list(win), "pixels": int(d.size),
                         "source_oversample": info.get("source_oversample"),
                         "source_taps": info.get("source_taps"),
                         "mean_bias_source_px": float(d.mean()),
                         "median_bias_source_px": float(np.median(d)),
                         "mean_abs_source_px": float(np.abs(d).mean())})
    return rows


def placement_test(S, Rsyn, T, wins):
    rows = []
    for win in wins:
        a, am, b, bm, *_ , info = prealign(S, Rsyn, max_side=2048, window=win, prewarp=T)
        if info["reference_decimation"] != 1:
            continue
        ga, gav = A.gradmag(a, am)
        gb, gbv = A.gradmag(b, bm)
        both = gav & gbv
        shifts, agree = [], []
        H, W = a.shape
        for y in range(0, H - 64 + 1, 64):
            for x in range(0, W - 64 + 1, 64):
                if both[y:y + 64, x:x + 64].all() and gb[y:y + 64, x:x + 64].std() > 1e-6:
                    p = A.shift_phase(gb[y:y + 64, x:x + 64], ga[y:y + 64, x:x + 64])[0]
                    n, _, edge = A.shift_ncc(gb[y:y + 64, x:x + 64], ga[y:y + 64, x:x + 64])
                    shifts.append(p)
                    agree.append(bool(np.hypot(*(p - n)) <= A.AGREE_PX and not edge))
        if len(shifts) < 4:
            continue
        s, g = np.array(shifts), np.array(agree)
        rows.append({"window": list(win), "subwindows": int(len(s)),
                     "median_shift_ref_px": np.median(s, axis=0).tolist(),
                     "mean_shift_ref_px": s.mean(axis=0).tolist(),
                     "methods_agree": int(g.sum()),
                     "median_shift_where_agree_ref_px": (np.median(s[g], axis=0).tolist() if g.any() else None),
                     "source_oversample": info.get("source_oversample"),
                     "source_taps": info.get("source_taps")})
    return rows


def main():
    cal = A.calibrate_signs()
    controls = {c["name"]: c for c in json.loads((DIAG / "controls.json").read_text())}
    out = {"estimator_calibration": cal, "cases": []}
    for name in ("control_ohrc_nac", "control_ohrc_nac_km", "control_tmc2_selene"):
        c = controls[name]
        print("CASE", name, flush=True)
        S, R, Rsyn = load(str(ROOT / c["source"])), load(str(ROOT / c["grid_of"])), load(str(ROOT / c["reference"]))
        wins = windows(ROOT / c["reference"])
        T = np.asarray(c["truth"]["T_native_ref_px"], float)
        case = {"name": name, "source_px_per_ref_px": c["truth"]["source_px_per_ref_px"],
                "corner_hypothesis_predicts_source_px": (c["truth"]["source_px_per_ref_px"] - 1) / 2,
                "ramp": ramp_test(S, R, wins), "placement": placement_test(S, Rsyn, T, wins)}
        out["cases"].append(case)
        (HERE / "sampling_test.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
        for r in case["ramp"]:
            print("  ramp %-6s window %s: f=%.3f taps=%s mean bias %+.4f source px (median %+.4f)" % (
                r["axis"], r["window"][:2], r["source_oversample"], r["source_taps"],
                r["mean_bias_source_px"], r["median_bias_source_px"]), flush=True)
        for r in case["placement"]:
            print("  placement window %s: median shift (%+.3f, %+.3f) ref px over %d subwindows; "
                  "phase/NCC agree in %d, median there %s" % (
                      r["window"][:2], *r["median_shift_ref_px"], r["subwindows"], r["methods_agree"],
                      None if r["median_shift_where_agree_ref_px"] is None else
                      "(%+.3f, %+.3f)" % tuple(r["median_shift_where_agree_ref_px"])), flush=True)


if __name__ == "__main__":
    main()
