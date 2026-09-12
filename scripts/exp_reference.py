"""Experiment harness: OHRC against an external LRO NAC reference.

Design rule for this whole phase: BASELINE -> ONE CHANGE -> MEASURE ->
KEEP / MODIFY / DELETE. Every method here is run on the *same* pair, on the
*same* grid, so the only variable is the correspondence method.

Why both images end up on one grid
----------------------------------
OHRC is a pushbroom ribbon in its own pixel frame; the NAC mosaic is north-up
polar stereographic. Comparing them in either native frame mixes a real ground
offset with a rotation of tens of degrees. Both are therefore resampled onto the
*same* south polar stereographic grid at the NAC scale, after which **the true
transform between them is close to a pure translation**. That makes the methods
directly comparable: correlation returns a translation, and a feature matcher's
fitted transform can be reduced to one.

The metric that matters
-----------------------
Not any single method's confidence - the CM_AVG spike showed a margin test
producing two false "LOCK" verdicts. What counts is **agreement between
independent methods**. If correlation and SIFT and AKAZE land on the same
translation, the terrain correspondence is real. If they scatter, it is not,
regardless of how good any one peak looked.

    python scripts/exp_reference.py --list
    python scripts/exp_reference.py --product 20241115T1525 --mosaic CM_235
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno import ohrc, verify                       # noqa: E402
from seleno.ohrc import reproject as R                # noqa: E402
from seleno.ohrc import tiles as T                    # noqa: E402

PDS_BASE = ("https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/"
            "DATA/BDR/NAC_POLE")
NS = NL = 45488
RECORD = 181952
HDR = RECORD
SAMP_OFF = 45487.5
CACHE = os.path.join(ROOT, "results", "nac_cache")
OUT = os.path.join(ROOT, "results", "experiments")

# Illumination-binned mosaics are named by sub-solar longitude bin; CM_AVG has
# no coherent illumination at all (its README: "does not accurately represent
# any realistic lighting condition").
MOSAICS = {"CM_AVG": "NAC_POLE_SOUTH_CM_AVG", "CM_235": "NAC_POLE_SOUTH_CM_235"}


def product_name(mosaic: str, tile: str) -> str:
    return "%s_%s" % (MOSAICS[mosaic], tile)


def url_for(mosaic: str, tile: str) -> str:
    d = MOSAICS[mosaic]
    return "%s/%s/%s_%s.IMG" % (PDS_BASE, d, d, tile)


def resolve(mosaic: str, tile: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "_url_%s_%s.txt" % (mosaic, tile))
    if os.path.exists(p):
        return open(p).read().strip()
    u = subprocess.run(["curl", "-sIL", "--max-time", "90", "-o", os.devnull,
                        "-w", "%{url_effective}", url_for(mosaic, tile)],
                       capture_output=True, text=True, check=True).stdout.strip()
    open(p, "w").write(u)
    return u


def stereo_to_nac(x, y):
    """x = col - 45487.5 ; y = -row - 0.5   (PDS3 offsets, 0-based, 1 m/px)."""
    return x + SAMP_OFF, -y - 0.5


def fetch_region(mosaic: str, tile: str, col0: int, row0: int, n: int,
                 ds: int = 1, block: int = 512):
    col0, row0 = int(col0), int(row0)
    if not (0 <= col0 and col0 + n <= NS and 0 <= row0 and row0 + n <= NL):
        raise ValueError("region (col %d row %d n %d) runs off tile %s; the pole "
                         "is the corner where four tiles meet" % (col0, row0, n, tile))
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, "%s_%s_c%d_r%d_n%d.npy" % (mosaic, tile, col0, row0, n))
    if os.path.exists(out):
        return np.load(out)
    # Any already-cached region that CONTAINS this box can serve it. Opened with
    # mmap so a 2048 crop out of a 6144 cache costs 16 MB, not 151 MB.
    pats = [("%s_%s_c" % (mosaic, tile), "%s_%s_c%%d_r%%d_n%%d.npy" % (mosaic, tile))]
    if mosaic == "CM_AVG":                      # written by the earlier spike script
        pats.append(("nac_%s_c" % tile, "nac_%s_c%%d_r%%d_n%%d.npy" % tile))
    for prefix, _ in pats:
        for f in sorted(os.listdir(CACHE)):
            if not (f.startswith(prefix) and f.endswith(".npy")):
                continue
            m = re.search(r"_c(\d+)_r(\d+)_n(\d+)\.npy$", f)
            if not m:
                continue
            c, r, N = (int(g) for g in m.groups())
            if c <= col0 and col0 + n <= c + N and r <= row0 and row0 + n <= r + N:
                print("    cropping %dx%d (ds %d) out of cached %s (no download)"
                      % (n, n, ds, f))
                big = np.load(os.path.join(CACHE, f), mmap_mode="r")
                rr, cc = row0 - r, col0 - c
                if ds == 1:
                    a = np.array(big[rr:rr + n, cc:cc + n])
                    return a, np.isfinite(a) & (a > -1e30)
                # Block-wise so peak memory is one block, not the whole region.
                # Nodata is held out of the average rather than poisoning it.
                m = n // ds
                out = np.empty((m, m), np.float32)
                okd = np.empty((m, m), np.float32)
                step = max(ds, (block // ds) * ds)
                for i in range(0, n, step):
                    h = min(step, n - i) // ds * ds
                    if h == 0:
                        break
                    blk = np.array(big[rr + i:rr + i + h, cc:cc + n], np.float32)
                    good = np.isfinite(blk) & (blk > -1e30)
                    blk[~good] = 0.0
                    sh = (m, h // ds)
                    num = cv2.resize(blk, sh, interpolation=cv2.INTER_AREA)
                    den = cv2.resize(good.astype(np.float32), sh, interpolation=cv2.INTER_AREA)
                    out[i // ds:i // ds + h // ds] = num / np.maximum(den, 1e-6)
                    okd[i // ds:i // ds + h // ds] = den
                    del blk, good, num, den
                return out, okd > 0.5
    lo, hi = HDR + row0 * RECORD, HDR + (row0 + n) * RECORD - 1
    tmp = os.path.join(CACHE, "_chunk_%s.bin" % mosaic)
    print("    fetching %.0f MB (one range request) ..." % ((hi - lo + 1) / 1e6))
    t0 = time.time()
    subprocess.run(["curl", "-s", "--max-time", "2400", "-r", "%d-%d" % (lo, hi),
                    "-o", tmp, resolve(mosaic, tile)], check=True)
    if os.path.getsize(tmp) < (hi - lo + 1):
        os.remove(tmp)
        raise IOError("short read")
    buf = np.fromfile(tmp, dtype="<f4")[: n * (RECORD // 4)].reshape(n, RECORD // 4)
    arr = np.ascontiguousarray(buf[:, col0:col0 + n])
    del buf
    os.remove(tmp)
    np.save(out, arr)
    print("    %.0fs cached" % (time.time() - t0))
    ok = np.isfinite(arr) & (arr > -1e30)
    if ds > 1:
        m = n // ds
        g = ok.astype(np.float32)
        a = np.where(ok, arr, 0.0).astype(np.float32)
        num = cv2.resize(a, (m, m), interpolation=cv2.INTER_AREA)
        den = cv2.resize(g, (m, m), interpolation=cv2.INTER_AREA)
        return num / np.maximum(den, 1e-6), den > 0.5
    return arr, ok


# --------------------------------------------------------------------------- #
# representations
# --------------------------------------------------------------------------- #

def to_u8(z, mask=None, lo=2, hi=98):
    z = np.nan_to_num(np.asarray(z, np.float32))
    m = mask if mask is not None else np.ones(z.shape, bool)
    if m.sum() < 10:
        return np.zeros(z.shape, np.uint8)
    a, b = np.percentile(z[m], (lo, hi))
    o = np.clip((z - a) * (255.0 / max(b - a, 1e-6)), 0, 255).astype(np.uint8)
    if mask is not None:
        o[~mask] = 0
    return o


def rep(img_u8, kind):
    """Representations under test. `raw` is the baseline; the rest are one change each."""
    z = img_u8.astype(np.float32)
    if kind == "raw":
        return img_u8
    if kind == "grad":
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, 3)
        g = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)
        return to_u8(g)
    if kind == "clahe":
        return cv2.createCLAHE(2.5, (8, 8)).apply(img_u8)
    if kind == "hipass":
        return to_u8(z - cv2.GaussianBlur(z, (0, 0), 12.0))
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# correspondence methods, all returning a translation estimate
# --------------------------------------------------------------------------- #


def translation_vote(p1, p2, bin_m=20.0):
    """Hough vote for a pure translation.

    Both images sit on the same projected grid, so a real correspondence must be
    a translation with scale 1 and rotation 0. Fitting a 4-DoF similarity to 2-3
    points instead lets RANSAC "explain" random matches with absurd scales, which
    is exactly what the first run produced. Voting on p2-p1 has a ONE point
    minimal model, so random matches cannot conspire.

    Significance is measured against the null of matches scattered uniformly over
    the span they actually occupy: a peak worth believing must hold far more
    matches than that null predicts for a cell of the same size.
    """
    d = np.asarray(p2, np.float64) - np.asarray(p1, np.float64)
    if len(d) < 4:
        return None
    key = np.round(d / bin_m).astype(np.int64)
    from collections import Counter
    c = Counter(map(tuple, key))
    best, bestn = None, -1
    for k in c:
        n = sum(c.get((k[0] + i, k[1] + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1))
        if n > bestn:
            bestn, best = n, k
    m = (np.abs(key[:, 0] - best[0]) <= 1) & (np.abs(key[:, 1] - best[1]) <= 1)
    span_x = max(d[:, 0].max() - d[:, 0].min(), bin_m)
    span_y = max(d[:, 1].max() - d[:, 1].min(), bin_m)
    expected = len(d) * (3 * bin_m) ** 2 / (span_x * span_y)
    return {"dx": float(d[m, 0].mean()), "dy": float(d[m, 1].mean()),
            "votes": int(bestn), "total": int(len(d)),
            "expected_by_chance": round(float(expected), 2),
            "significance": round(float(bestn / max(expected, 1e-6)), 2),
            "scatter_m": round(float(np.hypot(d[m, 0].std(), d[m, 1].std())), 1)}

def method_correlation(a_u8, b_u8, valid_a, search_px, kind, res_m=1.0):
    """Baseline. NCC template match; template is the valid core of the source."""
    A, B = rep(a_u8, kind), rep(b_u8, kind)
    ys, xs = np.nonzero(valid_a)
    if len(ys) < 5000:
        return None
    cy, cx = int(np.median(ys)), int(np.median(xs))
    n = a_u8.shape[0]
    half = min(cy, cx, n - cy, n - cx) - search_px - 4
    half = min(half, 1200)
    if half < 150:
        return None
    tmpl = A[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
    win = B[cy - half - search_px:cy + half + search_px,
            cx - half - search_px:cx + half + search_px].astype(np.float32)
    if tmpl.std() < 1e-6 or win.std() < 1e-6:
        return None
    resp = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(resp)
    m = resp.copy()
    r = max(5, min(resp.shape) // 12)
    m[max(0, loc[1] - r):loc[1] + r + 1, max(0, loc[0] - r):loc[0] + r + 1] = -1
    ox, oy = loc[0] - search_px, loc[1] - search_px
    edge = min(search_px - abs(ox), search_px - abs(oy)) <= 2
    return {"method": "corr-%s" % kind, "dx": ox * res_m, "dy": oy * res_m,
            "score": round(float(peak), 4), "second": round(float(m.max()), 4),
            "margin": round(float(peak - m.max()), 4),
            "template_m": 2 * half * res_m, "search_m": search_px * res_m,
            "at_search_edge": bool(edge), "n": None}


def method_features(a_u8, b_u8, valid_a, detector, kind, ratio=0.8, nfeat=20000,
                    edge_guard=24):
    """SIFT / AKAZE / ORB + mutual ratio test + MAGSAC++.

    Keypoints within `edge_guard` px of the source's validity boundary are
    dropped: the OHRC ribbon's edge against empty grid is a hard synthetic edge
    that detectors fire on enthusiastically and that has no counterpart in the
    reference.
    """
    A, B = rep(a_u8, kind), rep(b_u8, kind)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_guard + 1,) * 2)
    core = cv2.erode(valid_a.astype(np.uint8), ker).astype(bool)
    if core.sum() < 50000:
        return None
    if detector == "sift":
        det = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.02, edgeThreshold=12)
        norm = cv2.NORM_L2
    elif detector == "akaze":
        det = cv2.AKAZE_create(threshold=0.0008)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.ORB_create(nfeatures=nfeat, fastThreshold=8, nlevels=10)
        norm = cv2.NORM_HAMMING
    kp1, d1 = det.detectAndCompute(A, core.astype(np.uint8) * 255)
    kp2, d2 = det.detectAndCompute(B, None)
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return None
    if norm == cv2.NORM_HAMMING:
        d1, d2 = d1.astype(np.uint8), d2.astype(np.uint8)
    bf = cv2.BFMatcher(norm)

    def one_way(x, y):
        keep = {}
        for p in bf.knnMatch(x, y, k=2):
            if len(p) == 2 and p[0].distance < ratio * p[1].distance:
                keep[p[0].queryIdx] = p[0].trainIdx
        return keep

    fwd, bwd = one_way(d1, d2), one_way(d2, d1)
    pairs = [(i, j) for i, j in fwd.items() if bwd.get(j) == i]
    if len(pairs) < 8:
        return {"method": "%s-%s" % (detector, kind), "dx": None, "dy": None,
                "score": 0.0, "n": len(pairs), "inliers": 0, "note": "too few candidates"}
    p1 = np.float32([kp1[i].pt for i, _ in pairs])
    p2 = np.float32([kp2[j].pt for _, j in pairs])
    vote = translation_vote(p1, p2)
    M, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0,
                                          confidence=0.999, maxIters=20000)
    if M is None:
        return {"method": "%s-%s" % (detector, kind), "dx": None, "dy": None,
                "score": 0.0, "n": len(pairs), "inliers": 0, "note": "no model",
                "vote": vote}
    mask = mask.ravel().astype(bool)
    H = np.eye(3); H[:2, :] = M
    res = verify.transfer_error(H, p1.astype(np.float64), p2.astype(np.float64))
    d = verify.decompose(H)
    # On a shared grid anything but scale~1 / rotation~0 is a fitting artefact.
    bogus = abs(d["est_scale_x"] - 1.0) > 0.05 or abs(d["est_rotation_deg"]) > 5.0
    return {"method": "%s-%s" % (detector, kind), "vote": vote,
            "similarity_rejected": bool(bogus),
            "dx": None if bogus else float(M[0, 2]),
            "dy": None if bogus else float(M[1, 2]),
            "score": round(float(mask.mean()), 4), "n": int(len(pairs)),
            "inliers": int(mask.sum()),
            "rmse_px": round(float(np.sqrt((res[mask] ** 2).mean())), 3) if mask.any() else None,
            "scale": d["est_scale_x"], "rot_deg": d["est_rotation_deg"]}


# --------------------------------------------------------------------------- #

def agreement(rows, tol_m=150.0):
    """Cross-method agreement, reported as every cluster rather than the best.

    Three correlation variants agreeing is NOT three independent methods - they
    are one algorithm run on three renderings of the same pixels, so they fail
    together and produce a confident false lock. Clusters are therefore also
    scored by how many distinct ALGORITHM FAMILIES they span (corr / sift /
    akaze), which is the thing that actually makes agreement evidence.
    """
    pts = [(r["estimate"][0], r["estimate"][1], r["method"]) for r in rows
           if r and r.get("estimate") is not None]
    clusters, used = [], set()
    for x0, y0, _ in sorted(pts):
        near = [(x, y, m) for x, y, m in pts if np.hypot(x - x0, y - y0) <= tol_m]
        key = tuple(sorted(m for _, _, m in near))
        if key in used:
            continue
        used.add(key)
        xs = np.array([q[0] for q in near]); ys = np.array([q[1] for q in near])
        fams = sorted({m.split("-")[0] for _, _, m in near})
        clusters.append({
            "members": [q[2] for q in near], "families": fams,
            "n": len(near), "n_families": len(fams),
            "centre": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "offset_m": round(float(np.hypot(xs.mean(), ys.mean())), 1),
            "spread_m": round(float(max(np.hypot(xs - xs.mean(), ys - ys.mean())))
                              if len(near) > 1 else 0.0, 1)})
    clusters.sort(key=lambda c: (-c["n_families"], -c["n"]))
    return {"n": len(pts), "clusters": clusters,
            "best": clusters[0] if clusters else None}


def pick_window(a, min_km=6.0, want_m=6144):
    """OHRC windows inside tile P892S2250 with room for a large template."""
    gsd = a.gsd_m or 0.24
    idx = T.StripIndex.cached(a, os.path.join(ROOT, "results", "dataset"),
                              tile=512, stride=1024, step=8)
    out = []
    for st in idx.usable():
        cx, cy = st.sample0 + 512, st.line0 + 512
        x, y = a.geometry.stereo(cx, cy)
        d = float(np.hypot(x, y)) / 1000.0
        if not (x < -4000 and y < -4000 and d > min_km):
            continue
        col, row = stereo_to_nac(x, y)
        if not (want_m / 2 < col < NS - want_m / 2 and want_m / 2 < row < NL - want_m / 2):
            continue
        out.append({"sample0": st.sample0, "line0": st.line0, "std": st.std,
                    "x": float(x), "y": float(y), "km": round(d, 2),
                    "col": float(col), "row": float(row),
                    "max_tmpl_m": round(2 * min(cx, a.samples - cx, cy, a.lines - cy) * gsd)})
    out.sort(key=lambda w: (-min(w["max_tmpl_m"], 2400), -w["std"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default=None)
    ap.add_argument("--mosaic", default="CM_235", choices=sorted(MOSAICS))
    ap.add_argument("--tile", default="P892S2250")
    ap.add_argument("--region", type=int, default=3072)
    ap.add_argument("--search", type=int, default=700, help="correlation search, px = m")
    ap.add_argument("--reps", default="raw,grad,clahe")
    ap.add_argument("--align", default=None,
                    help="dx,dy metres to pre-apply to OHRC; a correct "
                         "offset must make every method relock at ~0")
    ap.add_argument("--coarse", type=int, default=1,
                    help="downsample factor; 4 buys a wide search cheaply")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--col0", type=int, default=None,
                    help="pin the NAC region origin (use an already-cached region)")
    ap.add_argument("--row0", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    a = None

    if args.col0 is not None and args.row0 is not None:
        # Region pinned to bytes already on disk: pick whichever OHRC product
        # actually covers the box rather than choosing a window and downloading.
        n = args.region
        xc = args.col0 + n / 2.0 - SAMP_OFF
        yc = -(args.row0 + n / 2.0) - 0.5
        best = None
        prods = ohrc.discover()
        if args.product:                      # explicit choice wins over proximity
            prods = [ohrc.by_timestamp(prods, args.product)]
        for p in prods:
            if not p.has_geometry:
                continue
            g = p.geometry
            d = float(np.min(np.hypot(g.x - xc, g.y - yc)))
            if best is None or d < best[0]:
                best = (d, p)
        if best[0] > n:
            print("no OHRC product has geometry within %.0f m of the pinned box "
                  "(nearest %.0f m)" % (n, best[0]))
            return 1
        a = best[1]
        sm, ln, _ok = a.geometry.stereo_to_pixel(xc, yc)
        sm, ln = float(np.ravel(sm)[0]), float(np.ravel(ln)[0])
        w = {"sample0": int(sm), "line0": int(ln), "std": 0.0,
             "km": round(float(np.hypot(xc, yc)) / 1000.0, 2),
             "col": args.col0 + n / 2.0, "row": args.row0 + n / 2.0,
             "pinned": True}
        print("pinned box -> nearest product %s, lattice %.0f m away, "
              "sample %.0f line %.0f" % (a.timestamp[:13], best[0], sm, ln))
        wins = [w]
    else:
        a = ohrc.by_timestamp(ohrc.discover(), args.product or "20241115T1525")
        wins = pick_window(a, want_m=args.region)
    if not wins:
        print("no usable window with room for a %d m region" % args.region)
        return 1
    if args.list:
        for w in wins[:10]:
            print("  s%-6d l%-7d sd %5.1f  %5.1f km  NAC col %.0f row %.0f  max tmpl %d m"
                  % (w["sample0"], w["line0"], w["std"], w["km"], w["col"], w["row"],
                     w["max_tmpl_m"]))
        return 0

    w = wins[0]
    sun = a.sun_at_line(max(0, min(a.lines - 1, w["line0"] + 512)))
    n = args.region
    col0, row0 = int(round(w["col"] - n / 2)), int(round(w["row"] - n / 2))
    print("EXPERIMENT  OHRC %s  x  NAC %s/%s" % (a.timestamp, args.mosaic, args.tile))
    print("  OHRC window s%d l%d (sd %.1f), %.1f km from pole" %
          (w["sample0"], w["line0"], w["std"], w["km"]))
    print("  OHRC sun at that line: az %.1f deg, el %+.3f deg" %
          (sun["azimuth_deg"], sun["elevation_deg"]))
    print("  NAC region %d x %d at col0 %d row0 %d" % (n, n, col0, row0))

    ds = args.coarse
    res = float(ds)
    nac, nac_ok = fetch_region(args.mosaic, args.tile, col0, row0, n, ds=ds)
    nac, nac_ok = nac[::-1], nac_ok[::-1]            # rows -> +y
    x0 = col0 - SAMP_OFF
    y_top = -row0 - 0.5
    adx, ady = (0.0, 0.0)
    if args.align:
        adx, ady = (float(v) for v in args.align.split(","))
        print("  pre-applying offset (%+.0f, %+.0f) m to the OHRC side; if that "
              "offset is right every method must now return ~0" % (adx, ady))
    bounds = (x0 - adx, y_top - n - ady, x0 + n - adx, y_top - ady)
    t0 = time.time()
    proj = R.reproject(a, bounds, res_m=res, max_side=n // ds)
    print("  OHRC reprojected onto the same grid at %.0f m/px in %.0fs, valid %.1f%%"
          % (res, time.time() - t0, 100 * proj.valid.mean()))

    nac_u8 = to_u8(nac, nac_ok)
    oh_u8 = to_u8(proj.image.astype(np.float32), proj.valid)
    valid = proj.valid & nac_ok[:proj.valid.shape[0], :proj.valid.shape[1]]
    dark = float((proj.image[proj.valid] <= 10).mean()) if proj.valid.any() else 1.0
    print("  OHRC pixels <= DN10 within coverage: %.1f%%" % (100 * dark))
    print("  correlation search +/-%d m; feature consensus is a translation vote"
          % args.search)

    search_px = max(8, int(round(args.search / res)))
    reps = [k.strip() for k in args.reps.split(",")]
    rows = []
    print("")
    print("  %-14s %8s %8s %9s %6s  %s"
          % ("method", "dx m", "dy m", "score", "cand", "consensus / note"))
    for kind in reps:
        r = method_correlation(oh_u8, nac_u8, valid, search_px, kind, res_m=res)
        gc.collect()
        if not r:
            continue
        r["estimate"] = None if r["at_search_edge"] else (r["dx"], r["dy"])
        rows.append(r)
        print("  %-14s %8.0f %8.0f %9.4f %6s  margin %.4f, tmpl %.0f m%s"
              % (r["method"], r["dx"], r["dy"], r["score"], "-", r["margin"],
                 r["template_m"],
                 "  << PINNED AT SEARCH EDGE" if r["at_search_edge"] else ""))
    for det in ("sift", "akaze"):
        for kind in reps:
            r = method_features(oh_u8, nac_u8, valid, det, kind)
            gc.collect()
            if not r:
                continue
            v = r.get("vote")
            ok = bool(v and v["votes"] >= 3 and v["significance"] >= 20.0)
            r["estimate"] = (v["dx"] * res, v["dy"] * res) if ok else None
            rows.append(r)
            if v:
                print("  %-14s %8.0f %8.0f %9s %6d  vote %d (chance %.1f) = %.1fx%s"
                      % (r["method"], v["dx"] * res, v["dy"] * res, "-", v["total"],
                         v["votes"], v["expected_by_chance"], v["significance"],
                         "  ACCEPTED" if ok else "  not significant"))
            else:
                print("  %-14s %8s %8s %9s %6d  %s"
                      % (r["method"], "-", "-", "-", r.get("n", 0),
                         r.get("note", "too few candidates")))

    ag = agreement(rows)
    print("")
    print("  CROSS-METHOD AGREEMENT (clusters within 150 m)")
    print("    methods returning an estimate : %d" % ag["n"])
    for c in ag["clusters"]:
        print("    (%+.0f, %+.0f) m  |%.0f m|  %d methods / %d families %s  spread %.0f m"
              % (c["centre"][0], c["centre"][1], c["offset_m"], c["n"],
                 c["n_families"], c["families"], c["spread_m"]))
    b = ag["best"]
    if b and b["n_families"] >= 2 and b["n"] >= 3:
        print("    -> CORRESPONDENCE at (%+.0f, %+.0f) m: %d methods spanning %s"
              % (b["centre"][0], b["centre"][1], b["n"], b["families"]))
    else:
        print("    -> NO CROSS-FAMILY AGREEMENT. Nothing here is trustworthy.")

    rec = {"ohrc": a.product_id, "mosaic": args.mosaic, "tile": args.tile,
           "window": w, "sun": sun, "region": n, "search_m": args.search, "coarse": args.coarse,
           "align_applied": [adx, ady],
           "ohrc_dark_fraction": round(dark, 4),
           "ohrc_valid_fraction": round(float(proj.valid.mean()), 4),
           "rows": rows, "agreement": ag}
    tag = "%s__%s_%s_s%d_l%d" % (a.timestamp[:13], args.mosaic, args.tile,
                                 w["sample0"], w["line0"])
    with open(os.path.join(OUT, "exp_%s.json" % tag), "w") as fh:
        json.dump(rec, fh, indent=2, default=str)

    cells = []
    for im, tag2 in ((nac_u8, "NAC %s" % args.mosaic),
                     (oh_u8, "OHRC %s reprojected" % a.timestamp[:13])):
        v = cv2.cvtColor(cv2.resize(im, (430, 430)), cv2.COLOR_GRAY2BGR)
        cv2.putText(v, tag2, (8, 20), 0, 0.42, (0, 255, 255), 1)
        cells.append(v)
    ov = np.zeros((430, 430, 3), np.uint8)
    ov[..., 1] = cv2.resize(nac_u8, (430, 430))
    ov[..., 0] = ov[..., 2] = cv2.resize(oh_u8, (430, 430))
    cv2.putText(ov, "overlay G=NAC M=OHRC", (8, 20), 0, 0.42, (0, 255, 255), 1)
    cells.append(ov)
    cv2.imwrite(os.path.join(OUT, "exp_%s.png" % tag), np.hstack(cells))
    print("\n  wrote results/experiments/exp_%s.{json,png}" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
