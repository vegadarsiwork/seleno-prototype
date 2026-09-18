"""OHRC <-> NAC validation pairs. Test-only; never reachable from the training set.

The cross-sensor pairs cannot be built the way the NAC<->NAC training pairs are,
because Chandrayaan-2's delivered geolocation is wrong by kilometres. But Phase 2
*measured* that error for two strips, with three algorithm families agreeing to
under 52 m and an independent prior measurement agreeing to under 83 m
(`reports/GEOLOCATION.md`). Applying the measured correction puts the OHRC strip
on the LOLA-tied grid, after which the truth relating it to a NAC mosaic of the
same ground is again the identity.

So this validation set is only as good as that measurement, and it inherits its
caveats:

* only strips whose offset is **confirmed** are used - the provisional and
  failed ones are excluded, not guessed at;
* the residual after correction is not zero, it is whatever the fine stage left
  (tens of metres, i.e. a pixel or two at 32 m sampling). A model is not expected
  to beat that, and `tolerance_px` reflects it.

This is the set to early-stop on. Improving on held-out NAC tiles while getting
worse here is precisely the overfitting-to-NAC-radiometry failure the amendment
called out, and it is invisible if you only watch in-domain validation.
"""
from __future__ import annotations

import json
import os

import numpy as np

# sub-solar longitude per OHRC product and the LROC bin matched to it in Phase 1
PRODUCT_BINS = {
    "20211228T2209": "115",
    "20241115T1326": "355",
    "20241115T1525": "355",
    "20251010T0942": "045",
}
TILE = "P892S2250"


def load_confirmed_offsets(path):
    """Only strips where two or more algorithm families agreed."""
    if not os.path.exists(path):
        return {}
    rep = json.load(open(path))
    out = {}
    for ts, r in rep.get("strips", {}).items():
        arm = r.get("arms", {}).get("direct_nac", {})
        c = arm.get("coarse", {})
        if not c.get("agree"):
            continue
        f = arm.get("fine") or {}
        if f.get("total_dx") is not None:
            out[ts] = (float(f["total_dx"]), float(f["total_dy"]), "coarse+fine")
        else:
            out[ts] = (float(c["consensus_dx"]), float(c["consensus_dy"]), "coarse")
    return out


def build_pairs(offsets_json, cache_dir, chip=512, res_m=None,
                min_valid_src=0.10, min_valid_ref=0.35, max_per_strip=8, seed=0):
    """OHRC chips and the co-located NAC chip, after the measured correction.

    Returns a list of ``(ohrc_u8, nac_u8, valid_o, valid_n, meta)``.
    """
    import cv2
    from .. import nac, ohrc
    from ..ohrc import reproject as RP

    offs = load_confirmed_offsets(offsets_json)
    if not offs:
        return []
    products = {p.timestamp[:13]: p for p in ohrc.discover()}
    rng = np.random.default_rng(seed)
    pairs = []
    for ts, (dx, dy, stage) in sorted(offs.items()):
        if ts not in products or ts not in PRODUCT_BINS:
            continue
        b = PRODUCT_BINS[ts]
        ref, rgx, rgy, r = nac.load_browse(b, TILE, cache_dir)
        res = res_m or r
        p = products[ts]
        x0, y0, x1, y1 = RP.strip_bounds_stereo(p)
        win = RP.reproject(p, (x0, y0, x1, y1), res_m=res, max_side=4000)

        # OHRC pixel (col,row) -> plane metres, corrected -> NAC pixel
        for _ in range(max_per_strip * 12):
            if len([q for q in pairs if q[4]["product"] == ts]) >= max_per_strip:
                break
            c = int(rng.integers(0, max(1, win.image.shape[1] - chip)))
            rr = int(rng.integers(0, max(1, win.image.shape[0] - chip)))
            vo = win.valid[rr:rr + chip, c:c + chip]
            # An OHRC strip is 12 000 samples = 2.88 km wide, so at ~32 m/px a
            # 512 px (16 km) chip can cover at most ~18% of itself. Holding the
            # source to the same coverage threshold as the reference is not
            # strict, it is unsatisfiable - it returned zero pairs.
            if vo.mean() < min_valid_src:
                continue
            gx0 = win.x0 + c * win.res_m + dx
            gy0 = win.y0 + rr * win.res_m + dy
            ci = int(round((gx0 - rgx[0]) / r))
            ri = int(round((gy0 - rgy[0]) / r))
            if ci < 0 or ri < 0 or ci + chip > ref.shape[1] or ri + chip > ref.shape[0]:
                continue
            nref = ref[ri:ri + chip, ci:ci + chip]
            vn = np.isfinite(nref)
            if vn.mean() < min_valid_ref:
                continue

            def u8(z, m):
                o = np.zeros(z.shape, np.uint8)
                if m.sum() < 50:
                    return o
                lo, hi = np.nanpercentile(np.asarray(z, np.float32)[m], (2, 98))
                o[m] = np.clip((np.asarray(z, np.float32)[m] - lo) * (255.0 / max(hi - lo, 1e-6)),
                               0, 255).astype(np.uint8)
                return o

            ochip = u8(win.image[rr:rr + chip, c:c + chip], vo)
            nchip = u8(nref, vn)
            pairs.append((ochip, nchip, vo, vn,
                          {"product": ts, "bin": b, "offset_m": [dx, dy],
                           "offset_stage": stage, "row": rr, "col": c,
                           "res_m": res}))
    return pairs
