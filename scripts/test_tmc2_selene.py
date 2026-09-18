"""TMC-2 against SELENE TC: the first genuinely cross-mission pair in this project.

    python scripts/test_tmc2_selene.py

Why this pair matters
---------------------
Everything the project has registered so far has been Chandrayaan-2 against LRO,
at the south pole, where the Sun never rises more than a degree above the
horizon. This is different on every axis:

* **cross-mission and cross-sensor**: ISRO TMC-2 (5.48 m) against JAXA SELENE TC
  (7.403 m). A scale ratio of 1.35, not the 31x of OHRC against TC.
* **non-polar**: latitude -15 to -18, solar incidence 39 deg rather than 90 deg.
* **real illumination change, twice over**: the SELENE morning and evening maps
  are the same ground under opposite Sun azimuths, both photometrically
  normalised to `STANDARD_GEOMETRY = (30, 0, 30)`. So the brightness difference
  between them is largely removed while the topographic shading and cast shadows
  from the actual acquisitions remain - which isolates the geometric part of the
  illumination problem from the radiometric part.

Both products are map-projected, so the comparison is done on one plate-carree
grid at SELENE's own sampling and the truth between TMC-2 and TC is a small
translation, not an unknown 8-DoF warp.

What is and is not claimed
--------------------------
There is **no independent ground truth** for this pair. SELENE TC is the
reference ISRO themselves used (`reference_data_used = SELENE` in the TMC-2
label), so a small residual is expected but its true value is unknown. This
script therefore reports *agreement between independent matchers*, not accuracy,
and says so. It is a feasibility test for PS pair type B, not an accuracy
measurement.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2                                                        # noqa: E402
from scipy.interpolate import LinearNDInterpolator                # noqa: E402
from seleno import align, matchers, ohrc                          # noqa: E402
from seleno.ohrc.label import parse_label                         # noqa: E402

TMC_ROOT = os.path.join(ROOT, "data/raw/ch2/tmc2/extracted")
SELENE = os.path.join(ROOT, "data/raw/selene")
OUT = os.path.join(ROOT, "results", "tmc2_selene")


def read_pds3_label(path):
    out = {}
    for line in open(path, errors="replace"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().strip('"')] = v.split("<")[0].strip().strip('"').strip()
    return out


class SeleneTile:
    """A SELENE TC map tile on its own plate-carree grid."""

    def __init__(self, lbl_path):
        L = read_pds3_label(lbl_path)
        self.path = lbl_path[:-4] + ".img"
        self.lines = int(L["LINES"])
        self.samples = int(L["LINE_SAMPLES"])
        self.res = float(L["MAP_RESOLUTION"])          # pixels per degree
        self.scale = float(L["SCALING_FACTOR"])
        self.ul_lat = float(L["UPPER_LEFT_LATITUDE"])
        self.ul_lon = float(L["UPPER_LEFT_LONGITUDE"])
        self.kind = "morning" if "MORNING" in L["DATA_SET_ID"] else "evening"
        want = self.lines * self.samples * 2
        got = os.path.getsize(self.path)
        if got != want:
            raise ValueError("%s is %d bytes, label implies %d" % (self.path, got, want))

    def window(self, lon0, lat0, lon1, lat1):
        """Crop by lon/lat. Returns (img float32, lons, lats)."""
        c0 = int(round((lon0 - self.ul_lon) * self.res))
        c1 = int(round((lon1 - self.ul_lon) * self.res))
        r0 = int(round((self.ul_lat - lat1) * self.res))
        r1 = int(round((self.ul_lat - lat0) * self.res))
        c0, c1 = max(0, c0), min(self.samples, c1)
        r0, r1 = max(0, r0), min(self.lines, r1)
        m = np.memmap(self.path, dtype=">u2", mode="r", shape=(self.lines, self.samples))
        a = np.asarray(m[r0:r1, c0:c1], np.float32) * self.scale
        lons = self.ul_lon + (np.arange(c0, c1) + 0.5) / self.res
        lats = self.ul_lat - (np.arange(r0, r1) + 0.5) / self.res
        return a, lons, lats


def tmc2_on_grid(product, lons, lats, node_step=1):
    """Resample a TMC-2 strip onto a plate-carree grid via its geometry lattice."""
    g = product.geometry
    pts = np.column_stack([g.lon[::node_step, ::node_step].ravel(),
                           g.lat[::node_step, ::node_step].ravel()])
    S, L = np.meshgrid(g.pixels[::node_step], g.scans[::node_step])
    f_s = LinearNDInterpolator(pts, S.ravel())
    f_l = LinearNDInterpolator(pts, L.ravel())
    LO, LA = np.meshgrid(lons, lats)
    SS, LL = f_s(LO, LA), f_l(LO, LA)
    valid = (np.isfinite(SS) & np.isfinite(LL)
             & (SS >= 0) & (SS < product.samples) & (LL >= 0) & (LL < product.lines))
    out = np.zeros(LO.shape, np.float32)
    if valid.any():
        s_lo, s_hi = int(np.nanmin(SS[valid])), int(np.nanmax(SS[valid])) + 1
        l_lo, l_hi = int(np.nanmin(LL[valid])), int(np.nanmax(LL[valid])) + 1
        block = product.read_tile(s_lo, l_lo, s_hi - s_lo, l_hi - l_lo).astype(np.float32)
        si = np.clip(np.nan_to_num(SS - s_lo).astype(np.int64), 0, block.shape[1] - 1)
        li = np.clip(np.nan_to_num(LL - l_lo).astype(np.int64), 0, block.shape[0] - 1)
        out[valid] = block[li[valid], si[valid]]
    return out, valid


def norm_u8(z, m):
    o = np.zeros(z.shape, np.uint8)
    if m.sum() < 100:
        return o
    lo, hi = np.percentile(z[m], (2, 98))
    o[m] = np.clip((z[m] - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    return o


def implied_shift(res, deg_per_px):
    """Reduce a matcher's correspondences to a translation, in metres."""
    if res is None or len(res.kp_src) < 4:
        return None
    d = res.kp_ref - res.kp_src
    med = np.median(d, axis=0)
    inl = np.linalg.norm(d - med, axis=1) < 3.0
    if inl.sum() < 4:
        return {"status": "no_consensus", "n": int(len(d)), "inliers": int(inl.sum())}
    est = d[inl].mean(axis=0)
    m_per_px = deg_per_px * math.pi / 180.0 * 1737400.0
    return {"status": "ok", "n": int(len(d)), "inliers": int(inl.sum()),
            "inlier_ratio": round(float(inl.mean()), 3),
            "dx_px": round(float(est[0]), 2), "dy_px": round(float(est[1]), 2),
            "dx_m": round(float(est[0]) * m_per_px, 1),
            "dy_m": round(float(est[1]) * m_per_px, 1),
            "magnitude_m": round(float(np.hypot(*est)) * m_per_px, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon0", type=float, default=141.30)
    ap.add_argument("--lat0", type=float, default=-16.60)
    ap.add_argument("--span", type=float, default=0.30, help="degrees")
    ap.add_argument("--matchers", nargs="*",
                    default=["sift", "akaze", "orb", "disk_lightglue"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device:
        os.environ["SELENO_DEVICE"] = args.device
    os.makedirs(OUT, exist_ok=True)

    tiles = {}
    for lbl in sorted(glob.glob(os.path.join(SELENE, "*.lbl"))):
        t = SeleneTile(lbl)
        tiles[t.kind] = t
    if "morning" not in tiles:
        sys.exit("no SELENE tile in %s" % SELENE)
    print("SELENE tiles: %s" % ", ".join("%s (%s)" % (k, os.path.basename(v.path))
                                         for k, v in sorted(tiles.items())))

    prods = [p for p in ohrc.discover(TMC_ROOT) if p.instrument == "TMC-2" and p.has_geometry]
    if not prods:
        sys.exit("no TMC-2 product with geometry under %s" % TMC_ROOT)

    ref = tiles["morning"]
    lon1, lat1 = args.lon0 + args.span, args.lat0 + args.span
    deg_per_px = 1.0 / ref.res
    print("box: lon %.3f..%.3f  lat %.3f..%.3f  at %.4f m/px"
          % (args.lon0, lon1, args.lat0, lat1,
             deg_per_px * math.pi / 180.0 * 1737400.0))

    # which TMC-2 strip covers the box?
    chosen = None
    for p in prods:
        L = p.label
        blk = L.corners_refined_block or L.corners_system
        la = [v for k, v in blk.items() if "lat" in k.lower()]
        lo = [v for k, v in blk.items() if "lon" in k.lower()]
        if min(la) <= args.lat0 and max(la) >= lat1 and min(lo) <= args.lon0 and max(lo) >= lon1:
            chosen = p
            break
    if chosen is None:
        sys.exit("no TMC-2 strip covers that box; strips are at %s"
                 % [(p.timestamp[:13]) for p in prods])
    print("TMC-2 strip: %s  (%d x %d, %.2f m/px, incidence %.2f deg)"
          % (chosen.timestamp[:13], chosen.lines, chosen.samples, chosen.gsd_m,
             chosen.label.solar_incidence_deg))

    sm, lons, lats = ref.window(args.lon0, args.lat0, lon1, lat1)
    print("SELENE window %s" % (sm.shape,))
    tm, tv = tmc2_on_grid(chosen, lons, lats)
    print("TMC-2 resampled onto the same grid: %.1f%% covered" % (100 * tv.mean()))
    if tv.mean() < 0.15:
        sys.exit("TMC-2 covers too little of the box; move it along the strip")

    A = norm_u8(tm, tv)
    results = {"box": [args.lon0, args.lat0, lon1, lat1],
               "tmc2": chosen.product_id, "arms": {}}

    for kind, tile in sorted(tiles.items()):
        B_raw, _, _ = tile.window(args.lon0, args.lat0, lon1, lat1)
        bv = B_raw > 0
        B = norm_u8(B_raw, bv)
        print("\n--- TMC-2 vs SELENE TC %s map ---" % kind)
        arm = {}
        for name in args.matchers:
            try:
                r = matchers.run_matcher(name, A, B)
                s = implied_shift(r, deg_per_px)
            except Exception as exc:
                s = {"status": "error", "reason": "%s: %s" % (type(exc).__name__, exc)}
            arm[name] = s
            if s and s.get("status") == "ok":
                print("  %-16s %5d matches, %4d inliers (%.0f%%)  ->  (%+.1f, %+.1f) m, |d| %.1f m"
                      % (name, s["n"], s["inliers"], 100 * s["inlier_ratio"],
                         s["dx_m"], s["dy_m"], s["magnitude_m"]))
            else:
                print("  %-16s %s" % (name, (s or {}).get("status", "insufficient_matches")))
        ok = [v for v in arm.values() if v and v.get("status") == "ok"]
        if len(ok) >= 2:
            xs = np.array([v["dx_m"] for v in ok])
            ys = np.array([v["dy_m"] for v in ok])
            spread = float(np.hypot(xs - xs.mean(), ys - ys.mean()).max())
            print("  AGREEMENT across %d matchers: consensus (%+.1f, %+.1f) m, "
                  "max deviation %.1f m" % (len(ok), xs.mean(), ys.mean(), spread))
            arm["_agreement"] = {"n": len(ok), "consensus_dx_m": round(float(xs.mean()), 1),
                                 "consensus_dy_m": round(float(ys.mean()), 1),
                                 "max_deviation_m": round(spread, 1)}
        results["arms"][kind] = arm

    json.dump(results, open(os.path.join(OUT, "tmc2_selene.json"), "w"), indent=1)
    print("\nwrote %s" % os.path.relpath(os.path.join(OUT, "tmc2_selene.json"), ROOT))
    print("\nNote: SELENE TC is the reference ISRO used for this product's own")
    print("geolocation, so these residuals are agreement, not independent accuracy.")


if __name__ == "__main__":
    main()
