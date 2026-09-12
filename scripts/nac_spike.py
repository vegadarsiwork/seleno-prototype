"""P0.1 spike - can Chandrayaan-2 OHRC be locked onto the LRO NAC controlled mosaic?

The reference is the LROC **South Pole Controlled NAC Average Mosaic**
(`NAC_POLE_SOUTH_CM_AVG`), 1 m/px, polar stereographic, centred 90 S, radius
1737.4 km - the same projection and radius as `seleno.ohrc.geometry`, so no
reprojection is needed on the reference side.

Two facts about it that shape this test:

* It is **controlled**: ISIS jigsaw over 18 323 NAC images tied to LOLA tracks,
  bundle residuals 0.8 px at 2-sigma. That is ~0.8 m absolute, against
  Chandrayaan-2's measured 500-900 m inter-product disagreement. It is a real
  geodetic reference.
* It is an **average** mosaic. Its own README says it "does not accurately
  represent any realistic lighting condition". So it has no coherent shadow
  structure, which is the hardest possible thing to correlate a deeply-shadowed
  OHRC window against. If this fails, `NAC_POLE_SOUTH_CM_235` (single
  illumination) is the fallback before concluding anything about matchability.

Two traps this script exists to avoid, both hit during the spike:

* **Window smaller than the geolocation error.** A 384 m window placed by CH-2
  geometry that is 500-900 m wrong shows *different ground*. Any comparison at
  that scale is meaningless; you must search.
* **The pole is the corner where all four tiles meet.** Our strips pass within
  2 km of it, so a naive box around a near-pole window runs off the tile edge
  into negative rows. Every window here is bounds-checked.

    python scripts/nac_spike.py                 # cache + sweep template sizes
    python scripts/nac_spike.py --list          # show candidate windows only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from seleno import ohrc                                    # noqa: E402
from seleno.ohrc import tiles as T                         # noqa: E402

# --- the reference product -------------------------------------------------
PDS = ("https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/"
       "NAC_POLE/NAC_POLE_SOUTH_CM_AVG/NAC_POLE_SOUTH_CM_AVG_P892S2250.IMG")
TILE = "P892S2250"          # lon 180-270 E, lat -90..-88.5  => x<=0, y<=0
NS = NL = 45488             # samples, lines
RECORD = 181952             # bytes per record = 45488 samples x 4 bytes
HDR = RECORD                # LABEL_RECORDS = 1, ^IMAGE = 2
# PDS3: SAMPLE_PROJECTION_OFFSET 45487.5, LINE_PROJECTION_OFFSET -0.5, 1 m/px
SAMP_OFF, LINE_OFF = 45487.5, -0.5

CACHE = os.path.join(ROOT, "results", "nac_cache")


def stereo_to_nac(x, y):
    """South polar stereographic metres -> (col, row), 0-based, in this tile.

    PDS3 convention: the projection origin sits at
    line = 1 + LINE_PROJECTION_OFFSET, sample = 1 + SAMPLE_PROJECTION_OFFSET,
    with line increasing downward (y decreasing) and sample rightward. In
    0-based array indices (col = sample-1, row = line-1) and at 1 m/px that is

        x = col - 45487.5          y = -row - 0.5

    which inverts to the expressions below. Getting the half-pixel wrong here
    biases everything downstream by 0.5 m - immaterial for correlation, fatal
    for a sub-pixel accuracy claim.
    """
    return x + SAMP_OFF, -y - 0.5


def nac_to_stereo(col, row):
    return col - SAMP_OFF, -row - 0.5


def resolve_url() -> str:
    """The PDS host 302s to a cloud store; resolve once and reuse."""
    p = os.path.join(CACHE, "_final_url.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    out = subprocess.run(["curl", "-sIL", "--max-time", "90", "-o", os.devnull,
                          "-w", "%{url_effective}", PDS],
                         capture_output=True, text=True, check=True).stdout.strip()
    os.makedirs(CACHE, exist_ok=True)
    open(p, "w").write(out)
    return out


def fetch_region(col0: int, row0: int, n: int, url: str) -> np.ndarray:
    """Cache an n x n NAC window as float32.

    ONE contiguous range request covering whole records, then slice columns.
    Per-row requests were tried first and earned an HTTP 429 - the server rate
    limits, and every request also went through a 302.
    """
    col0, row0 = int(col0), int(row0)
    if not (0 <= col0 and col0 + n <= NS and 0 <= row0 and row0 + n <= NL):
        raise ValueError("window (col %d, row %d, n %d) runs off tile %s "
                         "(%d x %d). The pole is the corner where four tiles "
                         "meet - pick a window further from it."
                         % (col0, row0, n, TILE, NS, NL))
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, "nac_%s_c%d_r%d_n%d.npy" % (TILE, col0, row0, n))
    if os.path.exists(out):
        print("  cached: %s" % os.path.basename(out))
        return np.load(out)

    lo = HDR + row0 * RECORD
    hi = HDR + (row0 + n) * RECORD - 1
    tmp = os.path.join(CACHE, "_chunk.bin")
    print("  one range request, %.0f MB ..." % ((hi - lo + 1) / 1e6))
    t0 = time.time()
    subprocess.run(["curl", "-s", "--max-time", "1800", "-r", "%d-%d" % (lo, hi),
                    "-o", tmp, url], check=True)
    got = os.path.getsize(tmp)
    if got < (hi - lo + 1):
        os.remove(tmp)
        raise IOError("short read: %d of %d bytes" % (got, hi - lo + 1))
    buf = np.fromfile(tmp, dtype="<f4")[: n * (RECORD // 4)].reshape(n, RECORD // 4)
    arr = np.ascontiguousarray(buf[:, col0:col0 + n])
    del buf
    os.remove(tmp)
    np.save(out, arr)
    json.dump({"tile": TILE, "col0": col0, "row0": row0, "n": n, "res_m": 1.0,
               "stereo_x0": col0 - SAMP_OFF, "stereo_y0": -(row0 + 0.5)},
              open(out.replace(".npy", ".json"), "w"), indent=2)
    print("  %.0fs -> %s" % (time.time() - t0, os.path.basename(out)))
    return arr


def gradmag(z):
    z = np.nan_to_num(np.asarray(z, np.float32))
    s = z.std() or 1.0
    z = (z - z.mean()) / s
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, 3)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)


def search(nac, tmpl):
    """NCC template match; returns offset, peak, runner-up outside an exclusion zone."""
    resp = cv2.matchTemplate(nac.astype(np.float32), tmpl.astype(np.float32),
                             cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(resp)
    cx = (nac.shape[1] - tmpl.shape[1]) // 2
    cy = (nac.shape[0] - tmpl.shape[0]) // 2
    m = resp.copy()
    r = max(5, min(resp.shape) // 12)
    m[max(0, loc[1] - r):loc[1] + r + 1, max(0, loc[0] - r):loc[0] + r + 1] = -1
    return loc[0] - cx, loc[1] - cy, float(peak), float(m.max()), loc


def candidates(a, min_km=6.0, limit=8, want_template_m=2304):
    """Usable OHRC windows inside tile P892S2250 that can host a large template.

    Two constraints, and the second is easy to forget: an OHRC strip is only
    12 000 samples = 2.88 km wide, so a window near the cross-track edge cannot
    host a template of any size. The largest square centred at sample s is
    2*min(s, samples-s) pixels, which collapses to nothing at the edges. Since
    correlation against a 1 m/px reference needs a template of order a kilometre
    to produce a decisive peak, windows are ranked by the template they can
    actually support, not by texture alone.
    """
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
        half_px = min(cx, a.samples - cx, cy, a.lines - cy)
        max_tm = 2 * half_px * gsd                      # metres
        out.append({"sample0": st.sample0, "line0": st.line0, "std": st.std,
                    "x": float(x), "y": float(y), "km_from_pole": round(d, 2),
                    "max_template_m": round(max_tm)})
    # prefer windows that can host the template we want, then by texture
    out.sort(key=lambda w: (-min(w["max_template_m"], want_template_m), -w["std"]))
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="20241115T1326")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--region", type=int, default=4096, help="NAC region side, px = m")
    ap.add_argument("--templates", default="384,768,1536,2304")
    args = ap.parse_args()

    a = ohrc.by_timestamp(ohrc.discover(), args.product)
    cands = candidates(a)
    if not cands:
        print("no candidate window sits safely inside tile %s" % TILE)
        return 1
    print("candidate OHRC windows inside %s:" % TILE)
    for c in cands:
        print("   s%-6d l%-7d sd %5.1f  %5.1f km from pole  max template %5d m" %
              (c["sample0"], c["line0"], c["std"], c["km_from_pole"],
               c["max_template_m"]))
    if args.list:
        return 0

    w = cands[0]
    cx, cy = w["sample0"] + 512, w["line0"] + 512
    x, y = a.geometry.stereo(cx, cy)
    col, row = stereo_to_nac(x, y)
    N = args.region
    col0, row0 = int(round(col - N / 2)), int(round(row - N / 2))
    print("\nusing s%d l%d -> stereo (%.0f, %.0f) -> NAC col %.0f row %.0f"
          % (w["sample0"], w["line0"], x, y, col, row))
    print("NAC region %d x %d at col0 %d row0 %d" % (N, N, col0, row0))

    nac = fetch_region(col0, row0, N, resolve_url())
    valid = np.isfinite(nac) & (nac > -1e30)
    print("  valid %.1f%%   reflectance %.3f .. %.3f"
          % (100 * valid.mean(), np.nanmin(nac[valid]), np.nanmax(nac[valid])))

    gsd = a.gsd_m or 0.24
    nac_g = gradmag(nac)
    print("\ntemplate sweep (OHRC resampled 0.24 -> 1 m/px, %.1fx):" % (1.0 / gsd))
    print("  %-9s %-8s %8s %8s %8s   %-22s %s"
          % ("tmpl m", "search m", "peak", "2nd", "margin", "offset (m)", "verdict"))
    rows = []
    for tm in [int(v) for v in args.templates.split(",")]:
        if tm + 200 > N:
            print("  %-9d skipped - larger than the cached region" % tm)
            continue
        half = int(tm / 2 / gsd)
        s0, l0 = int(cx) - half, int(cy) - half
        if s0 < 0 or l0 < 0 or s0 + 2 * half > a.samples or l0 + 2 * half > a.lines:
            print("  %-9d skipped - OHRC window runs off the strip" % tm)
            continue
        oh = a.read_tile(s0, l0, 2 * half, 2 * half)
        tmpl = cv2.resize(oh, (tm, tm), interpolation=cv2.INTER_AREA).astype(np.float32)
        dx, dy, peak, second, loc = search(nac_g, gradmag(tmpl))
        conf = peak > 0.10 and (peak - second) > 0.04
        rows.append((tm, dx, dy, peak, second, conf, loc, tmpl))
        print("  %-9d %-8d %8.4f %8.4f %8.4f   (%+6d, %+6d) %6.0f   %s"
              % (tm, (N - tm) // 2, peak, second, peak - second, dx, dy,
                 np.hypot(dx, dy), "LOCK" if conf else "no lock"))

    good = [r for r in rows if r[5]]
    print()
    if good:
        tm, dx, dy, peak, second, _, loc, tmpl = max(good, key=lambda r: r[3] - r[4])
        print("BEST LOCK: %d m template, offset (%+d, %+d) m = %.0f m, peak %.3f margin %.3f"
              % (tm, dx, dy, np.hypot(dx, dy), peak, peak - second))
        print("  -> Chandrayaan-2 geolocation is %.0f m from the LOLA-controlled reference"
              % np.hypot(dx, dy))
        cut = nac[loc[1]:loc[1] + tm, loc[0]:loc[0] + tm]
        pc = (N - tm) // 2
        pred = nac[pc:pc + tm, pc:pc + tm]
        cells = []
        for im, tag in ((pred, "NAC at CH-2 predicted spot"),
                        (cut, "NAC at correlation LOCK"),
                        (tmpl, "CH-2 OHRC 0.24->1 m/px")):
            z = np.nan_to_num(np.asarray(im, np.float32))
            m = z > 0
            lo_, hi_ = np.percentile(z[m], (1, 99)) if m.any() else (0, 1)
            d8 = np.clip((z - lo_) * (255.0 / max(hi_ - lo_, 1e-6)), 0, 255).astype(np.uint8)
            v = cv2.cvtColor(cv2.resize(d8, (400, 400)), cv2.COLOR_GRAY2BGR)
            cv2.putText(v, tag, (8, 20), 0, 0.42, (0, 255, 255), 1)
            cells.append(v)
        out = os.path.join(ROOT, "results", "dataset", "_nac_lock.png")
        cv2.imwrite(out, np.hstack(cells))
        print("  wrote %s" % out)
    else:
        print("NO LOCK at any template size.")
        print("  Next: try NAC_POLE_SOUTH_CM_235 (single-illumination controlled")
        print("  mosaic). CM_AVG averages away coherent shading by construction,")
        print("  which is the hardest possible reference to correlate against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
