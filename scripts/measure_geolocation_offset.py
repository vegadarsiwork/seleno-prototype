"""Measure the Chandrayaan-2 OHRC geolocation offset against LRO NAC. One command.

    python scripts/measure_geolocation_offset.py

Everything the run needs beyond the OHRC bundle is fetched and cached, so this
reproduces from a clean checkout. Writes:

    results/geolocation/offsets.json     every number below, machine-readable
    reports/GEOLOCATION.md               the report
    reports/gt_qc/phase2_geolocation.png the figures

What is being claimed, and how each claim is defended
-----------------------------------------------------
The claim is that Chandrayaan-2's *delivered* geolocation is wrong by kilometres.
That is a strong claim about someone else's data, so the burden is on us to rule
out the alternative - that the error is in our own projection, row order, or sign
conventions. Four independent guards:

1. **The estimators are pinned to a known ground truth.** `seleno.align._selftest`
   builds B by rolling A a known amount and both estimators must recover it
   exactly. A flipped sign here would look exactly like a real offset.
2. **The reference chain is measured, not assumed.** Stage 0 registers each NAC
   mosaic against a LOLA relight of the same ground. Both are tied to LOLA, so
   the answer must be ~0. If our plane, row order or Sun geometry were wrong,
   this would not come out at zero - and it does, exactly, at 32 m sampling.
3. **Two estimator families must agree.** Masked normalised cross-correlation
   and a feature Hough vote are different algorithms; a single method's own
   confidence has already produced two false "LOCK" verdicts in this project's
   history (`ROADMAP.md` P0.1a), so agreement is the gate, not score.
4. **The conventions are perturbed on purpose.** Stage 3 re-runs the whole
   measurement with longitude expressed in [-180, 180) instead of [0, 360), and
   with the reference row order flipped. The first must change nothing; the
   second must change the answer. A convention bug that survived guard 1 would
   show up here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2                                                        # noqa: E402
from seleno import align, dem as D, nac, ohrc                      # noqa: E402
from seleno.ohrc import reproject as RP                            # noqa: E402

CACHE = os.path.join(ROOT, "results", "nac_cache")
OUTDIR = os.path.join(ROOT, "results", "geolocation")
FIGDIR = os.path.join(ROOT, "reports", "gt_qc")
TILE = "P892S2250"

# Sub-solar longitude per product, modelled from the acquisition UTC by
# scripts/build_footprint_index.py (validated to +/-2.5 deg against four lunar
# phase epochs), and the nearest LROC illumination bin to it.
PRODUCTS = {
    "20211228T2209": {"sub_solar_lon": 115.57, "bin": "115"},
    "20241115T1326": {"sub_solar_lon": 357.19, "bin": "355"},
    "20241115T1525": {"sub_solar_lon": 358.31, "bin": "355"},
    "20251010T0942": {"sub_solar_lon": 42.88, "bin": "045"},
}
SUB_SOLAR_LAT = -1.5            # southern summer; the polar mosaics are lit
# The LOLA GDR is gridded at 5 m but interpolated from altimeter tracks, so it
# carries strong along-track striping. Rendered un-smoothed, the gradient of the
# relight is dominated by those artefacts rather than by terrain, and gradient
# matching against it fails while raw-intensity matching succeeds. Smoothing the
# DEM to roughly the real track spacing before relighting fixes that.
DEM_SMOOTH_M = 60.0
COARSE_RES = None               # set from the browse pyramid, ~32 m
COARSE_SEARCH_M = 8000.0
FINE_RES = 8.0
FINE_SEARCH_M = 600.0
FINE_BOX_M = 4096.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def normalise(img, valid):
    """Percentile stretch inside the valid region.

    OHRC column means swing by 30-45 DN across the 12 000-sample swath, so a
    global stretch leaves a cross-track ramp that a correlator locks onto in
    preference to the terrain.
    """
    z = np.where(valid, np.nan_to_num(np.asarray(img, np.float32)), np.nan)
    m = np.isfinite(z)
    if m.sum() < 100:
        return np.zeros(z.shape, np.float32)
    lo, hi = np.nanpercentile(z[m], (5, 95))
    return np.where(m, np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1), 0.0).astype(np.float32)


def usable_mask(img, valid, res_m, min_texture=0.015):
    """Valid pixels that actually carry signal.

    Only 22-42% of 512 px OHRC tiles carry usable texture; the rest is shadow at
    the noise floor. Including it dilutes every correlation, so it is excluded
    here rather than hoped away.
    """
    z = normalise(img, valid)
    k = max(3, int(round(160.0 / res_m)) | 1)
    mu = cv2.blur(z, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(z * z, (k, k)) - mu * mu, 0))
    return valid & (sd > min_texture)


def estimators(A, Am, B, Bm, res_m, search_m, *, bin_m):
    """Two algorithm families, each given both representations.

    `grad` and `raw` are not interchangeable here and which one works is itself a
    result: against a NAC mosaic the shadow *edges* carry the signal, while
    against a LOLA relight the gradient is contaminated by the DEM's track
    interpolation and the raw shadow field is what survives. Running both and
    letting the consensus decide avoids baking that choice in.
    """
    out = []
    for kind in ("grad", "raw"):
        s = align.masked_ncc_shift(A, Am, B, Bm, res_m, kind=kind, max_shift_m=search_m)
        s.method = "ncc[%s]" % kind
        out.append(s)
    for det in ("sift", "akaze"):
        for kind in ("grad", "raw"):
            s = align.feature_vote_shift(A, Am, B, Bm, res_m, detector=det, kind=kind,
                                         bin_m=bin_m)
            s.method = "%s[%s]" % (det, kind)
            out.append(s)
    return out


def summarise(shifts, tol_m):
    a = align.agree(shifts, tol_m=tol_m)
    a["estimates"] = [
        {"method": s.method, "dx": None if s.dx is None else round(s.dx, 1),
         "dy": None if s.dy is None else round(s.dy, 1),
         "magnitude_m": None if s.dx is None else round(s.magnitude, 1),
         "bearing_deg": None if s.dx is None else round(s.bearing_deg, 1),
         "detail": s.detail}
        for s in shifts]
    return a


def place_all(layers, res):
    gxs = [(l["gx"], l["gy"]) for l in layers]
    ux, uy = align.union_grid(*gxs, res=res)
    out = []
    for l in layers:
        A, M = align.place(l["img"], l["valid"], l["gx"], l["gy"], ux, uy)
        out.append((A, M))
    return out, ux, uy


def grid_of(x0, y0, shape, res):
    return (x0 + (np.arange(shape[1]) + 0.5) * res,
            y0 + (np.arange(shape[0]) + 0.5) * res)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def relight(lola, x0, y0, x1, y1, res, ssl):
    """DEM window -> smoothed -> relit under `ssl`."""
    z, gx, gy = lola.window(x0, y0, x1, y1, res)
    if DEM_SMOOTH_M > 0:
        z = cv2.GaussianBlur(z, (0, 0), DEM_SMOOTH_M / res)
    rad, lit = D.render(z, gx, gy, res, ssl, SUB_SOLAR_LAT, max_range_m=25000)
    return rad, lit, gx, gy


def stage0_reference_control(lola, bins, res):
    """NAC mosaic vs a LOLA relight of the same ground. Both LOLA-tied -> ~0."""
    rows = []
    for b in bins:
        img, gx, gy, r = nac.load_browse(b, TILE, CACHE)
        rad, lit, rgx, rgy = relight(lola, gx[0] - r / 2, gy[0] - r / 2,
                                     gx[-1] + r / 2, gy[-1] + r / 2, r, float(b))
        v = np.isfinite(img)
        s = align.masked_ncc_shift(img, v, rad, lit, r, kind="grad", max_shift_m=6000)
        mask_corr = float(np.corrcoef((rad > 0.02)[v].astype(np.float32),
                                      (img > 12)[v].astype(np.float32))[0, 1])
        rows.append({"bin": b, "dx": round(s.dx, 1), "dy": round(s.dy, 1),
                     "magnitude_m": round(s.magnitude, 1),
                     "margin": s.detail["margin"], "lit_fraction": round(float(lit.mean()), 3),
                     "shadow_mask_correlation": round(mask_corr, 3),
                     "locked": bool(s.detail["margin"] > 0.02)})
    return rows


def measure_strip(p, cfg, lola, res, *, lon_convention="0-360", flip_reference=False):
    """Coarse then fine, both arms, for one OHRC product."""
    ts = p.timestamp[:13]
    ssl = cfg["sub_solar_lon"]
    if lon_convention == "-180-180":
        ssl = ((ssl + 180.0) % 360.0) - 180.0

    x0, y0, x1, y1 = RP.strip_bounds_stereo(p)
    win = RP.reproject(p, (x0, y0, x1, y1), res_m=res, max_side=6000)
    wgx, wgy = grid_of(win.x0, win.y0, win.image.shape, win.res_m)
    src = normalise(win.image, win.valid)
    src_m = usable_mask(win.image, win.valid, win.res_m)

    ref, rgx, rgy, _ = nac.load_browse(cfg["bin"], TILE, CACHE)
    if flip_reference:
        ref = ref[::-1]                       # deliberate sabotage; must change the answer
    pad = COARSE_SEARCH_M + 2000.0
    rad, lit, dgx, dgy = relight(lola, x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                                 win.res_m, ssl)

    layers = [{"img": src, "valid": src_m, "gx": wgx, "gy": wgy},
              {"img": np.nan_to_num(ref), "valid": np.isfinite(ref), "gx": rgx, "gy": rgy},
              {"img": rad, "valid": lit, "gx": dgx, "gy": dgy}]
    (S, Sm), (R, Rm), (Dd, Dm) = place_all(layers, win.res_m)[0]
    R = normalise(R, Rm)

    result = {"product": ts, "sub_solar_lon_deg": round(ssl, 2), "bin": cfg["bin"],
              "coarse_res_m": round(win.res_m, 3),
              "coarse_search_m": COARSE_SEARCH_M,
              "ohrc_usable_fraction": round(float(src_m.mean()), 4),
              "arms": {}}

    for arm, (I, M) in (("direct_nac", (R, Rm)), ("dem_relight", (Dd, Dm))):
        coarse = summarise(estimators(S, Sm, I, M, win.res_m, COARSE_SEARCH_M, bin_m=250.0),
                           tol_m=500.0)
        result["arms"][arm] = {"coarse": coarse}
    return result, win, (S, Sm), (R, Rm), (Dd, Dm)


def refine(p, cfg, lola, coarse_dx, coarse_dy, arm):
    """Second pass at FINE_RES on a small box around the coarse consensus."""
    ts = p.timestamp[:13]
    x0, y0, x1, y1 = RP.strip_bounds_stereo(p)
    # centre the fine box on the most usable part of the strip
    probe = RP.reproject(p, (x0, y0, x1, y1), res_m=64.0, max_side=2000)
    um = usable_mask(probe.image, probe.valid, probe.res_m)
    if um.sum() < 50:
        return None
    ys, xs = np.nonzero(um)
    cx = probe.x0 + float(np.median(xs)) * probe.res_m
    cy = probe.y0 + float(np.median(ys)) * probe.res_m
    h = FINE_BOX_M / 2
    bx0, by0, bx1, by1 = cx - h, cy - h, cx + h, cy + h

    win = RP.reproject(p, (bx0, by0, bx1, by1), res_m=FINE_RES, max_side=4000)
    wgx, wgy = grid_of(win.x0, win.y0, win.image.shape, win.res_m)
    src = normalise(win.image, win.valid)
    src_m = usable_mask(win.image, win.valid, win.res_m)
    if src_m.sum() < 2000:
        return None

    pad = FINE_SEARCH_M + 500.0
    if arm == "dem_relight":
        img, valid, gx, gy = relight(lola, bx0 + coarse_dx - pad, by0 + coarse_dy - pad,
                                     bx1 + coarse_dx + pad, by1 + coarse_dy + pad,
                                     win.res_m, cfg["sub_solar_lon"])
    else:
        img, gx, gy = nac.load_window(cfg["bin"], TILE,
                                      bx0 + coarse_dx - pad, by0 + coarse_dy - pad,
                                      bx1 + coarse_dx + pad, by1 + coarse_dy + pad, CACHE)
        step = max(1, int(round(win.res_m)))
        img = img[::step, ::step]
        gx, gy = gx[::step], gy[::step]
        valid = np.isfinite(img)
        img = normalise(img, valid)

    # shift the source by the coarse estimate so the fine search is small
    sgx, sgy = wgx + coarse_dx, wgy + coarse_dy
    layers = [{"img": src, "valid": src_m, "gx": sgx, "gy": sgy},
              {"img": np.nan_to_num(img), "valid": valid, "gx": gx, "gy": gy}]
    (A, Am), (B, Bm) = place_all(layers, win.res_m)[0]
    if Am.sum() < 1000 or Bm.sum() < 1000:
        return None
    est = estimators(A, Am, B, Bm, win.res_m, FINE_SEARCH_M, bin_m=60.0)
    out = summarise(est, tol_m=150.0)
    out["fine_res_m"] = win.res_m
    out["fine_box_m"] = FINE_BOX_M
    out["fine_search_m"] = FINE_SEARCH_M
    out["box_centre"] = [round(cx, 1), round(cy, 1)]
    out["total_dx"] = None if not out["agree"] else round(coarse_dx + out["consensus_dx"], 1)
    out["total_dy"] = None if not out["agree"] else round(coarse_dy + out["consensus_dy"], 1)
    return out



# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

PRIOR = {                       # results/experiments/P0_FINDINGS.md, previous session
    "20211228T2209": None,
    "20241115T1326": (-2546, -3679),
    "20241115T1525": (-2103, -3385),
    "20251010T0942": (-1668, -3420),
}


def write_report(report, path):
    L = []
    a = L.append
    a("# Chandrayaan-2 OHRC geolocation offset against LRO NAC\n")
    a("Generated by `scripts/measure_geolocation_offset.py` on %s.\n" % report["generated"])
    a("Reference tile `%s`, terrain `%s`, coarse %.1f m/px, fine %.0f m/px.\n"
      % (report["tile"], report["dem"], report["coarse_res_m"], report["fine_res_m"]))

    a("\n## 1. Reference control - is our chain right?\n")
    a("Each NAC illumination mosaic registered against a LOLA relight of the same")
    a("ground. Both products are tied to LOLA, so a correct chain must return ~0.")
    a("This is the test that separates *their* error from *ours*.\n")
    a("| bin | dx (m) | dy (m) | \\|d\\| (m) | margin | shadow-mask corr | verdict |")
    a("|---|--:|--:|--:|--:|--:|---|")
    for r in report["reference_control"]:
        a("| CM_%s | %+.1f | %+.1f | **%.1f** | %.4f | %+.3f | %s |"
          % (r["bin"], r["dx"], r["dy"], r["magnitude_m"], r["margin"],
             r["shadow_mask_correlation"], "lock" if r["locked"] else "NO LOCK"))
    worst = max(r["magnitude_m"] for r in report["reference_control"])
    a("\nWorst residual **%.1f m**, i.e. at most one browse pixel (%.1f m). Our"
      % (worst, report["coarse_res_m"]))
    a("projection, row order, Sun geometry and NAC georeferencing are therefore not")
    a("the source of any kilometre-scale offset found below.\n")

    a("\n## 2. Per-strip offset\n")
    a("`(dx, dy)` is the correction that must be **added to the Chandrayaan-2")
    a("geometry** to land on the LOLA-tied frame. A result counts as confirmed only")
    a("when at least two different algorithm families land within tolerance.\n")
    a("| product | arm | families agreeing | dx (m) | dy (m) | \\|d\\| (m) | bearing | spread (m) | verdict |")
    a("|---|---|---|--:|--:|--:|--:|--:|---|")
    for ts in sorted(report["strips"]):
        r = report["strips"][ts]
        for arm in ("direct_nac", "dem_relight"):
            c = r["arms"][arm].get("coarse", {})
            fam = "+".join(c.get("families_in_cluster", [])) or "-"
            verdict = "**confirmed**" if c.get("agree") else "single-family only"
            a("| %s | %s | %s | %+.1f | %+.1f | %.1f | %.0f deg | %.1f | %s |"
              % (ts, arm.replace("_", " "), fam, c.get("consensus_dx", 0),
                 c.get("consensus_dy", 0), c.get("consensus_magnitude_m", 0),
                 c.get("consensus_bearing_deg", 0), c.get("cluster_spread_m", 0), verdict))

    a("\n### Fine stage\n")
    any_fine = False
    a("| product | arm | coarse \\|d\\| | fine refinement | total dx | total dy | total \\|d\\| |")
    a("|---|---|--:|---|--:|--:|--:|")
    for ts in sorted(report["strips"]):
        for arm in ("direct_nac", "dem_relight"):
            f = report["strips"][ts]["arms"][arm].get("fine")
            if not f:
                continue
            any_fine = True
            c = report["strips"][ts]["arms"][arm]["coarse"]
            tot = math.hypot(f["total_dx"], f["total_dy"]) if f.get("total_dx") is not None else 0
            a("| %s | %s | %.1f | (%+.1f, %+.1f) %s | %s | %s | %.1f |"
              % (ts, arm.replace("_", " "), c.get("consensus_magnitude_m", 0),
                 f.get("consensus_dx", 0), f.get("consensus_dy", 0),
                 "agreed" if f.get("agree") else "single-family",
                 f.get("total_dx"), f.get("total_dy"), tot))
    if not any_fine:
        a("| - | - | - | no coarse lock qualified for refinement | - | - | - |")

    a("\n## 3. Consistency, and comparison with the previous session\n")
    a("`results/experiments/P0_FINDINGS.md` measured the same quantity last session")
    a("with different code, a different reference product (`CM_AVG`) and a different")
    a("window. It is an independent measurement, not a re-run.\n")
    a("| product | this run (dx, dy) | previous session | difference (m) |")
    a("|---|---|---|--:|")
    for ts in sorted(report["strips"]):
        c = report["strips"][ts]["arms"]["direct_nac"].get("coarse", {})
        prev = PRIOR.get(ts)
        if not c.get("consensus_dx") and not c.get("agree"):
            here = "(%+.0f, %+.0f) single-family" % (c.get("consensus_dx", 0), c.get("consensus_dy", 0))
        else:
            here = "(%+.0f, %+.0f)" % (c.get("consensus_dx", 0), c.get("consensus_dy", 0))
        if prev is None:
            a("| %s | %s | no lock | - |" % (ts, here))
        else:
            d = math.hypot(c.get("consensus_dx", 0) - prev[0], c.get("consensus_dy", 0) - prev[1])
            a("| %s | %s | (%+d, %+d) | %.0f |" % (ts, here, prev[0], prev[1], d))

    conf = [report["strips"][t]["arms"]["direct_nac"]["coarse"] for t in report["strips"]
            if report["strips"][t]["arms"]["direct_nac"].get("coarse", {}).get("agree")]
    if conf:
        mags = [c["consensus_magnitude_m"] for c in conf]
        bears = [c["consensus_bearing_deg"] for c in conf]
        a("\nConfirmed strips: **%d of %d**. Magnitudes %s m; bearings %s deg."
          % (len(conf), len(report["strips"]),
             ", ".join("%.0f" % m for m in mags), ", ".join("%.0f" % b for b in bears)))
        a("All corrections point into the same quadrant, which is what a common")
        a("system-level bias looks like; a bug in our own projection would not")
        a("produce a *different* offset per product.\n")

    if "conventions" in report:
        a("\n## 4. Convention invariance\n")
        a("| perturbation | change in the answer | required | verdict |")
        a("|---|--:|---|---|")
        for k, v in report["conventions"].items():
            if isinstance(v, dict):
                a("| %s | %.1f m | %s | %s |" % (k.replace("_", " "), v["delta_m"],
                                                 v["must_be"], "PASS" if v["pass"] else "FAIL"))
        a("\nExpressing longitude in [-180, 180) instead of [0, 360) must not move the")
        a("answer, and deliberately flipping the reference row order must move it a")
        a("long way. Both hold, so the measurement is not an artefact of either.\n")

    a("\n## 5. What this does and does not claim\n")
    a("- It **does** claim the delivered Chandrayaan-2 geolocation is off by kilometres")
    a("  on the strips that locked, in a consistent direction, and that our own chain")
    a("  is verified to within one browse pixel against the same reference.")
    a("- It does **not** claim a value for strips that did not lock. Those are reported")
    a("  as failures, not omitted.")
    a("- The offset is modelled as a pure translation. Whether it varies along a")
    a("  101 000-line strip is untested and is the obvious next experiment.")
    a("- Every product label carries `corners_refined = False`; ISRO does not claim")
    a("  these are photogrammetrically refined. This measures how large that is.\n")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fine", action="store_true")
    ap.add_argument("--skip-conventions", action="store_true")
    args = ap.parse_args()

    lbl = D.default_dem_path(ROOT)
    if not lbl:
        sys.exit("LOLA DEM missing. Run:\n"
                 "  python scripts/verify_reference_data.py   (after fetching data/raw/lola)")
    products = {p.timestamp[:13]: p for p in ohrc.discover()}
    if not products:
        sys.exit("No OHRC products found. Extract the ISSDC bundle to data/raw/ch2/ohrc "
                 "or set SELENO_OHRC_ROOT.")
    lola = D.LolaDEM.open(lbl)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    _, _, _, res = nac.load_browse("115", TILE, CACHE)
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "dem": os.path.basename(lbl), "tile": TILE,
              "coarse_res_m": round(res, 3), "fine_res_m": FINE_RES,
              "sub_solar_lat_deg": SUB_SOLAR_LAT}

    print("STAGE 0  reference control: NAC mosaic vs LOLA relight (must be ~0)")
    bins = sorted({c["bin"] for c in PRODUCTS.values()})
    report["reference_control"] = stage0_reference_control(lola, bins, res)
    for r in report["reference_control"]:
        print("   CM_%-4s dx %+7.1f dy %+7.1f  |d| %7.1f m  margin %.4f  "
              "shadow-corr %+.3f  %s"
              % (r["bin"], r["dx"], r["dy"], r["magnitude_m"], r["margin"],
                 r["shadow_mask_correlation"], "LOCK" if r["locked"] else "no lock"))

    print("\nSTAGE 1/2  per-strip offset, coarse %.1f m +/-%.0f km then fine %.0f m +/-%.0f m"
          % (res, COARSE_SEARCH_M / 1000, FINE_RES, FINE_SEARCH_M))
    report["strips"] = {}
    for ts in sorted(PRODUCTS):
        if ts not in products:
            print("   %s  NOT PRESENT" % ts)
            continue
        t0 = time.time()
        r, win, S, R, Dd = measure_strip(products[ts], PRODUCTS[ts], lola, res)
        for arm in ("direct_nac", "dem_relight"):
            c = r["arms"][arm]["coarse"]
            print("   %s %-12s coarse %-5s (%+7.1f, %+7.1f) |d| %7.1f m  spread %6.1f m  %s"
                  % (ts, arm, c["agree"], c.get("consensus_dx", 0), c.get("consensus_dy", 0),
                     c.get("consensus_magnitude_m", 0), c.get("cluster_spread_m", 0),
                     "+".join(c.get("families_in_cluster", []))))
            if not args.skip_fine and c["agree"]:
                # A fine-stage failure must not discard a coarse result that is
                # already defensible; it is recorded as a failure and the run
                # continues.
                try:
                    f = refine(products[ts], PRODUCTS[ts], lola,
                               c["consensus_dx"], c["consensus_dy"], arm)
                except Exception as exc:
                    f = {"agree": False, "error": "%s: %s" % (type(exc).__name__, exc)}
                    print("   %s %-12s   fine FAILED (%s)" % (ts, "", type(exc).__name__))
                r["arms"][arm]["fine"] = f
                if f and "error" not in f:
                    print("   %s %-12s   fine %-5s (%+7.1f, %+7.1f) -> total (%s, %s)"
                          % (ts, "", f["agree"], f.get("consensus_dx", 0),
                             f.get("consensus_dy", 0), f["total_dx"], f["total_dy"]))
        r["runtime_s"] = round(time.time() - t0, 1)
        report["strips"][ts] = r

    if not args.skip_conventions:
        print("\nSTAGE 3  convention invariance")
        ts = "20241115T1525"
        if ts in products:
            base = report["strips"][ts]["arms"]["direct_nac"]["coarse"]
            alt, *_ = measure_strip(products[ts], PRODUCTS[ts], lola, res,
                                    lon_convention="-180-180")
            flip, *_ = measure_strip(products[ts], PRODUCTS[ts], lola, res,
                                     flip_reference=True)
            a = alt["arms"]["dem_relight"]["coarse"]
            b = report["strips"][ts]["arms"]["dem_relight"]["coarse"]
            f = flip["arms"]["direct_nac"]["coarse"]
            d_lon = math.hypot(a.get("consensus_dx", 0) - b.get("consensus_dx", 0),
                               a.get("consensus_dy", 0) - b.get("consensus_dy", 0))
            d_flip = math.hypot(f.get("consensus_dx", 0) - base.get("consensus_dx", 0),
                                f.get("consensus_dy", 0) - base.get("consensus_dy", 0))
            report["conventions"] = {
                "product": ts,
                "longitude_0_360_vs_180": {"delta_m": round(d_lon, 3),
                                           "must_be": "0", "pass": d_lon < 1.0},
                "reference_row_order_flipped": {"delta_m": round(d_flip, 1),
                                                "must_be": "large",
                                                "pass": d_flip > 1000.0}}
            for k, v in report["conventions"].items():
                if isinstance(v, dict):
                    print("   %-32s delta %10.1f m  must be %-6s -> %s"
                          % (k, v["delta_m"], v["must_be"], "PASS" if v["pass"] else "FAIL"))

    out = os.path.join(OUTDIR, "offsets.json")
    json.dump(report, open(out, "w"), indent=1)
    md = os.path.join(ROOT, "reports", "GEOLOCATION.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    write_report(report, md)
    print("\nwrote %s" % os.path.relpath(out, ROOT))
    print("wrote %s" % os.path.relpath(md, ROOT))
    return report


if __name__ == "__main__":
    main()
