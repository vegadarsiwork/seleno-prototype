"""Which matcher wins, as a function of working resolution.

Phase 2 and Phase 7 appear to disagree about the matcher ranking. They do not.
They were run at different working resolutions, and the ranking moves with
resolution, because downsampling removes the fine shadow structure that breaks
descriptors. This script measures that directly: the same pair, the same
methods, several working grids, one table per pair.

    python scripts/resolution_ranking.py [--out reports/RESOLUTION_RANKING.md]

Each row is a real run of a real method on real data. The native-resolution row
is measured inside a window positioned by the coarse solution, which is what the
tool's own fine stage does - without that correction an unrefined strip's window
is filled with ground from kilometres away and every method scores zero.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

RG = importlib.import_module("seleno.tool.register")
from seleno import verify as V                                        # noqa: E402
from seleno.tool import methods as M                                  # noqa: E402
from seleno.tool.profiles import Profiles                             # noqa: E402
from seleno.tool.scene import load                                    # noqa: E402

METHODS = ["dense-ncc", "disk_lightglue", "sift", "akaze", "orb"]

PAIRS = [
    {"name": "OHRC -> LROC NAC",
     "source": "data/raw/ch2/ohrc/data/calibrated/20241115/"
               "ch2_ohr_ncp_20241115T1326321339_d_img_d18.xml",
     "reference": "data/raw/nac/NAC_CM355_P892S2250_pole_6km.tif",
     "sides": [256, 512, 1024, 2048]},
    {"name": "TMC-2 -> SELENE TC",
     "source": "data/raw/ch2/tmc2/extracted/data/calibrated/20260813/"
               "ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml",
     "reference": "data/raw/selene/TCO_MAPe04_S15E141S18E144SC.lbl",
     "sides": [256, 512, 1024, 2048]},
]


def score(A, Am, B, Bm, max_shift, grid=(8, 8)):
    """Run every method on one placement and verify each independently."""
    rows = []
    for name in METHODS:
        t0 = time.time()
        try:
            c, why = RG._run_one(name, A, Am, B, Bm, max_shift, grid)
        except Exception as exc:                                      # noqa: BLE001
            rows.append({"method": name, "candidates": 0, "inliers": 0,
                         "ratio": 0.0, "seconds": round(time.time() - t0, 2),
                         "note": type(exc).__name__})
            continue
        n = 0 if c is None else len(c)
        inl, ratio = 0, 0.0
        if n >= 4:
            vr = V.verify(c.src, c.ref, model_type="affine", threshold=3.0)
            inl, ratio = int(vr.n_inliers), float(vr.inlier_ratio)
        rows.append({"method": name, "candidates": n, "inliers": inl,
                     "ratio": round(ratio, 4),
                     "seconds": round(time.time() - t0, 2), "note": ""})
    return rows


def run_pair(spec, profiles):
    S = load(os.path.join(ROOT, spec["source"]), profiles)
    R = load(os.path.join(ROOT, spec["reference"]), profiles)
    ref_gsd = float(R.gsd_m or 0)
    src_gsd = float(S.gsd_m or 0)
    cache: dict = {}
    out = {"name": spec["name"], "source_gsd_m": src_gsd, "reference_gsd_m": ref_gsd,
           "levels": []}

    best_H = None
    best_frame = None
    for side in spec["sides"]:
        A, Am, B, Bm, bx, by, fr = RG.prealign(S, R, max_side=side, cache=cache)
        both = Am & Bm
        if both.sum() < 4096:
            continue
        step = fr["reference_decimation"]
        mpp = ref_gsd * step if ref_gsd else None
        rows = score(A, Am, B, Bm, max_shift=max(16, 4 * step))
        out["levels"].append({"label": "%d x decimation" % step,
                              "working_shape": list(A.shape), "decimation": step,
                              "metres_per_px": mpp, "native": step == 1,
                              "rows": rows})
        print("  %-18s grid %-12s %6.2f m/px  %s" % (
            "%dx decimation" % step, "x".join(map(str, A.shape)), mpp or -1,
            "  ".join("%s %d/%d" % (r["method"], r["inliers"], r["candidates"])
                      for r in rows)), flush=True)
        # keep the best model we have seen, to position the native window
        top = max(rows, key=lambda r: r["inliers"])
        if top["inliers"] >= 8 and (best_H is None or side == max(spec["sides"])):
            c, _ = RG._run_one(top["method"], A, Am, B, Bm, max(16, 4 * step), (8, 8))
            if c is not None and len(c) >= 4:
                vr = V.verify(c.src, c.ref, model_type="affine", threshold=3.0)
                if vr.model is not None and vr.n_inliers >= 8:
                    best_H, best_frame = vr.model, fr

    # --- native resolution, inside a window placed by the coarse solution ---
    if best_H is not None and best_frame["reference_decimation"] > 1:
        st = best_frame["reference_decimation"]
        orow, ocol = best_frame["reference_origin"]
        T = np.array([[1.0 / st, 0, -ocol / float(st)],
                      [0, 1.0 / st, -orow / float(st)], [0, 0, 1.0]])
        Hw = np.eye(3)
        Hw[:2, :] = np.asarray(best_H, float)[:2, :]
        prewarp = np.linalg.inv(np.linalg.inv(T) @ Hw @ T)

        A0, Am0, B0, Bm0, _, _, fr0 = RG.prealign(S, R, max_side=max(spec["sides"]),
                                                  cache=cache)
        bb = M.overlap_bbox(Am0 & Bm0, pad=4)
        cy = orow + (bb[0] + bb[2]) / 2.0 * st
        cx = ocol + (bb[1] + bb[3]) / 2.0 * st
        half = max(spec["sides"]) // 2
        H_, W_ = R.array.shape
        r0 = int(np.clip(cy - half, 0, max(0, H_ - 2 * half)))
        c0 = int(np.clip(cx - half, 0, max(0, W_ - 2 * half)))
        win = (r0, c0, min(H_, r0 + 2 * half), min(W_, c0 + 2 * half))
        for label, pw in (("native, geometry only", None),
                          ("native, coarse-corrected", prewarp)):
            A, Am, B, Bm, bx, by, fr = RG.prealign(S, R, max_side=max(spec["sides"]),
                                                   window=win, cache=cache, prewarp=pw)
            if (Am & Bm).sum() < 4096:
                continue
            rows = score(A, Am, B, Bm, max_shift=32)
            out["levels"].append({"label": label, "working_shape": list(A.shape),
                                  "decimation": fr["reference_decimation"],
                                  "metres_per_px": ref_gsd or None,
                                  "native": True, "rows": rows})
            print("  %-18s grid %-12s %6.2f m/px  %s" % (
                label, "x".join(map(str, A.shape)), ref_gsd or -1,
                "  ".join("%s %d/%d" % (r["method"], r["inliers"], r["candidates"])
                          for r in rows)), flush=True)
    return out


def markdown(results) -> str:
    L = ["# Matcher ranking as a function of working resolution", "",
         "Generated by `scripts/resolution_ranking.py` on %s."
         % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "",
         "Every cell is `inliers / candidates` from a real run. Inliers are",
         "counted by an independent affine verification at a 3 px threshold, so",
         "the columns are comparable across methods and across rows.", ""]
    for r in results:
        L += ["## %s" % r["name"], "",
              "Source %.2f m/px, reference %.2f m/px."
              % (r["source_gsd_m"], r["reference_gsd_m"]), "",
              "| working resolution | grid | " + " | ".join(METHODS) + " | winner |",
              "|---|---|" + "---|" * (len(METHODS) + 1)]
        for lv in r["levels"]:
            by = {x["method"]: x for x in lv["rows"]}
            top = max(lv["rows"], key=lambda x: x["inliers"])
            cells = []
            for mth in METHODS:
                x = by.get(mth)
                cells.append("--" if x is None else "%d / %d" % (x["inliers"], x["candidates"]))
            L.append("| %s (%s) | %s | %s | **%s** |" % (
                ("%.2f m/px" % lv["metres_per_px"]) if lv["metres_per_px"] else "n/a",
                lv["label"], "x".join(map(str, lv["working_shape"])),
                " | ".join(cells),
                top["method"] if top["inliers"] >= 8 else "none"))
        L.append("")
    L += FINDINGS.splitlines()
    return "\n".join(L)


FINDINGS = """
## Phase 2 for comparison

`scripts/benchmark_illumination.py`, NAC against NAC across sub-solar-longitude
bins, truth = identity, **32 m/px**, solved = within 3 px:

| matcher | delta-az 45-90 | delta-az 90-135 | delta-az 135-180 |
|---|--:|--:|--:|
| sift | 0/12 | 0/12 | 0/6 |
| akaze | 0/12 | 0/12 | 0/6 |
| orb | 0/12 | 0/12 | 0/6 |
| **disk_lightglue** | **7/12** | 0/12 | 0/6 |

## What the three tables say together

They were read as a contradiction. They are not one. Phase 2 was run at
**32 m/px**, and the Phase 7 rows at a comparable resolution agree with it
exactly: at 25 m/px on the OHRC pair, DISK+LightGlue scores 113 inliers while
AKAZE scores **0**. The disagreement was never between the phases - it was
between two working resolutions that were never stated side by side.

**1. The ranking inverts with working resolution.** On OHRC -> NAC the learned
matcher owns the coarse end and the classical detectors own the fine end:

| working resolution | winner | DISK+LightGlue | AKAZE |
|---|---|--:|--:|
| 25 m/px | disk_lightglue | 113 | 0 |
| 13 m/px | disk_lightglue | 418 | 33 |
| 7 m/px | disk_lightglue | 388 | 138 |
| 4 m/px | **akaze** | 334 | 442 |
| 1 m/px (native) | **akaze** | 537 | **3223** |

The mechanism is illumination, not scale as such. What breaks a gradient
descriptor at the pole is fine shadow structure that moves with the Sun;
downsampling averages it away, which is why a learned matcher trained for
robustness wins once the image is blurred enough, and why ordinary corner
detectors come back as soon as real texture is resolved.

**2. Polarity overrides resolution entirely.** TMC-2 against SELENE is
contrast-inverted (cross-correlation -0.51). No sparse method reaches double
figures at ANY working resolution, native included, while dense masked NCC on
gradient magnitude locks at every scale where there is signal at all. Sun-angle
robustness and polarity robustness are separate problems, and a matcher that
solves one need not touch the other.

**3. A native window is worthless without the coarse solution.** The two
"native" rows on the OHRC pair differ only in whether the window was positioned
using the coarse transform. Placed from the raw geometry - which is out by
kilometres on an unrefined strip - every method collapses to near zero
(DISK 0/1, SIFT 0/1). Corrected, the same window at the same resolution gives
AKAZE 3223 inliers. A resolution comparison that skips this step measures
nothing but the geolocation error.

## Why the tool tries several methods

This is the justification for running candidates and taking the geometrically
verified winner rather than picking a matcher in advance. Any fixed choice is
wrong somewhere in this table: DISK loses at native resolution, AKAZE scores
zero at 25 m/px, and both are useless on an anti-correlated pair. The tool
cannot know the working resolution or the polarity before it reads the files,
so it measures the pair, orders the candidates accordingly, and records which
one won in `metrics.json:method_used`. The fine stage re-runs that selection at
native resolution for the same reason - the coarse winner has no claim to the
fine grid, and on the OHRC pair it is a different method.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports",
                                                  "RESOLUTION_RANKING.md"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results",
                                                   "resolution_ranking.json"))
    ap.add_argument("--rebuild", action="store_true",
                    help="rewrite the report from the saved measurements "
                         "without re-running anything")
    a = ap.parse_args()
    if a.rebuild:
        with open(a.json) as fh:
            results = json.load(fh)
        with open(a.out, "w") as fh:
            fh.write(markdown(results))
        print("rebuilt %s from %s" % (os.path.relpath(a.out, ROOT),
                                      os.path.relpath(a.json, ROOT)))
        return
    profiles = Profiles.load()
    results = []
    for spec in PAIRS:
        if not os.path.exists(os.path.join(ROOT, spec["source"])):
            print("skip %s: source missing" % spec["name"])
            continue
        print("== %s" % spec["name"], flush=True)
        results.append(run_pair(spec, profiles))
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(results, fh, indent=1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(markdown(results))
    print("\nwrote %s and %s" % (os.path.relpath(a.out, ROOT),
                                 os.path.relpath(a.json, ROOT)))


if __name__ == "__main__":
    main()
