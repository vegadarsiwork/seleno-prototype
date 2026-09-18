"""Matcher benchmark against real Sun-azimuth change, with exact ground truth.

    python scripts/benchmark_illumination.py

Why this dataset
----------------
The LROC controlled south-polar mosaics are one ISIS jigsaw solution (18 323 NAC
images, LOLA-tied, 0.8 px at 2 sigma) resampled onto **one** 1 m polar
stereographic grid, then split into bins by sub-solar longitude. So two bins of
the same tile are:

* the same ground, to about a pixel, **without any matcher being involved** - the
  ground truth is the identity transform, by construction; and
* genuinely differently illuminated: different Sun azimuth, different cast
  shadows, different underlying NAC frames.

That is the combination this project has been missing. The legacy `data/pairs/`
fixtures are an image matched against a warped copy of itself, so their
illumination change is a gamma edit; here it is real.

What is measured
----------------
For each ordered pair of bins and each matcher: the residual translation the
matcher implies, against a truth of zero. Reported per matcher and stratified by
Sun-azimuth difference, which Phase 1 established is the variable that actually
moves at the pole (incidence is pinned near 90 degrees there).

Failures are outcomes. A matcher that returns nothing, or that returns a
confident wrong answer, appears in the table as such rather than being dropped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2                                                       # noqa: E402
from seleno import matchers, nac, verify                          # noqa: E402

CACHE = os.path.join(ROOT, "results", "nac_cache")
OUT = os.path.join(ROOT, "results", "illumination_bench")
TILE = "P892S2250"
BINS = ["005", "035", "065", "095", "125", "155", "185", "215", "245", "275", "305", "335"]
PATCH = 512
MIN_VALID = 0.45
PASS_PX = 3.0            # a translation within 3 px of truth counts as solved
MATCHERS = ["sift", "akaze", "orb", "rift", "disk_lightglue"]


def d_az(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def pick_patches(imgs, n_patches, rng):
    """Boxes where *every* bin has data and texture, so no pair is handed a blank."""
    stack = np.stack([np.isfinite(v) for v in imgs.values()])
    common = stack.all(axis=0)
    ref = np.nan_to_num(list(imgs.values())[0])
    k = PATCH
    h, w = common.shape
    cand = []
    for j in range(0, h - k, k // 2):
        for i in range(0, w - k, k // 2):
            sub = common[j:j + k, i:i + k]
            if sub.mean() < MIN_VALID:
                continue
            tex = float(np.std(ref[j:j + k, i:i + k][sub]))
            cand.append((tex, j, i))
    cand.sort(reverse=True)
    return [(j, i) for _, j, i in cand[:n_patches]]


def norm_u8(z):
    m = np.isfinite(z)
    o = np.zeros(z.shape, np.uint8)
    if m.sum() < 50:
        return o, m
    lo, hi = np.nanpercentile(z[m], (2, 98))
    o[m] = np.clip((z[m] - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    return o, m


def evaluate(res, truth_shift=(0.0, 0.0)):
    """Residual of a matcher's implied translation against the known truth."""
    if res is None or len(res.kp_src) < 4:
        return {"status": "insufficient_matches", "n": 0 if res is None else len(res.kp_src)}
    d = res.kp_ref - res.kp_src
    # MAGSAC on a translation-only model: fit as a 2-DoF problem by voting, then
    # report the inlier set. Fitting a homography to a pair that must be identity
    # is how this project previously manufactured scale factors of 0.157.
    med = np.median(d, axis=0)
    resid = np.linalg.norm(d - med, axis=1)
    inl = resid < 3.0
    if inl.sum() < 4:
        return {"status": "no_consensus", "n": int(len(d)), "inliers": int(inl.sum())}
    est = d[inl].mean(axis=0)
    err = math.hypot(est[0] - truth_shift[0], est[1] - truth_shift[1])
    return {"status": "ok", "n": int(len(d)), "inliers": int(inl.sum()),
            "inlier_ratio": round(float(inl.mean()), 3),
            "dx_px": round(float(est[0]), 2), "dy_px": round(float(est[1]), 2),
            "error_px": round(err, 2), "solved": bool(err <= PASS_PX)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=3)
    ap.add_argument("--bins", nargs="*", default=BINS)
    ap.add_argument("--matchers", nargs="*", default=MATCHERS)
    ap.add_argument("--device", default=None,
                    help="cuda to run learned matchers on a GPU; sweeps over the "
                         "144-pair matrix are impractically slow on CPU")
    args = ap.parse_args()
    if args.device:
        os.environ["SELENO_DEVICE"] = args.device
    os.makedirs(OUT, exist_ok=True)

    print("loading %d illumination bins of tile %s ..." % (len(args.bins), TILE))
    imgs, res_m = {}, None
    for b in args.bins:
        img, gx, gy, r = nac.load_browse(b, TILE, CACHE)
        imgs[b] = img
        res_m = r
    print("  %.2f m/px, %s" % (res_m, list(imgs.values())[0].shape))

    rng = np.random.default_rng(0)
    boxes = pick_patches(imgs, args.patches, rng)
    print("  %d patches of %d px where every bin has >=%.0f%% coverage"
          % (len(boxes), PATCH, 100 * MIN_VALID))

    rows = []
    pairs = list(itertools.combinations(args.bins, 2))
    total = len(pairs) * len(boxes) * len(args.matchers)
    done = 0
    for (b1, b2) in pairs:
        daz = d_az(b1, b2)
        for (j, i) in boxes:
            A, ma = norm_u8(imgs[b1][j:j + PATCH, i:i + PATCH])
            B, mb = norm_u8(imgs[b2][j:j + PATCH, i:i + PATCH])
            for name in args.matchers:
                done += 1
                t0 = time.time()
                try:
                    kw = {"device": args.device} if name in ("disk_lightglue", "loftr") \
                        and args.device else {}
                    r = matchers.run_matcher(name, A, B, **kw)
                    ev = evaluate(r)
                except Exception as exc:
                    ev = {"status": "error", "reason": "%s: %s" % (type(exc).__name__, exc)}
                ev.update({"matcher": name, "bin_a": b1, "bin_b": b2, "d_az_deg": daz,
                           "patch": [j, i], "runtime_s": round(time.time() - t0, 2)})
                rows.append(ev)
            if done % 50 < len(args.matchers):
                print("  %d/%d ..." % (done, total), flush=True)

    json.dump(rows, open(os.path.join(OUT, "raw.json"), "w"), indent=1)

    # ---------------- summary ----------------
    def bucket(d):
        return "0-45" if d < 45 else "45-90" if d < 90 else "90-135" if d < 135 else "135-180"

    print("\n%-16s %7s %7s %8s %9s %9s" % ("matcher", "runs", "solved", "rate",
                                           "med err", "med inl"))
    print("-" * 62)
    summary = {}
    for name in args.matchers:
        sub = [r for r in rows if r["matcher"] == name]
        ok = [r for r in sub if r.get("status") == "ok"]
        solved = [r for r in ok if r.get("solved")]
        errs = [r["error_px"] for r in solved]
        inls = [r["inliers"] for r in ok]
        summary[name] = {"runs": len(sub), "solved": len(solved),
                         "rate": round(len(solved) / max(len(sub), 1), 3),
                         "median_error_px": round(float(np.median(errs)), 2) if errs else None,
                         "median_inliers": int(np.median(inls)) if inls else 0}
        print("%-16s %7d %7d %7.1f%% %9s %9s"
              % (name, len(sub), len(solved), 100 * len(solved) / max(len(sub), 1),
                 summary[name]["median_error_px"], summary[name]["median_inliers"]))

    print("\nsolve rate by Sun-azimuth difference")
    buckets = ["0-45", "45-90", "90-135", "135-180"]
    print("%-16s %s" % ("matcher", "".join("%12s" % b for b in buckets)))
    print("-" * (16 + 12 * len(buckets)))
    strat = {}
    for name in args.matchers:
        cells = []
        strat[name] = {}
        for bk in buckets:
            sub = [r for r in rows if r["matcher"] == name and bucket(r["d_az_deg"]) == bk]
            s = sum(1 for r in sub if r.get("solved"))
            strat[name][bk] = {"runs": len(sub), "solved": s}
            cells.append("%12s" % ("%d/%d" % (s, len(sub)) if sub else "-"))
        print("%-16s %s" % (name, "".join(cells)))

    json.dump({"summary": summary, "stratified": strat, "res_m": res_m,
               "device": args.device or "cpu",
               "tile": TILE, "patch_px": PATCH, "pass_px": PASS_PX,
               "bins": args.bins, "n_patches": len(boxes)},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
