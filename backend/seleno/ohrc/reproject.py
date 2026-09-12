"""Reprojecting OHRC strips onto a common south polar stereographic grid.

This is the "reference and overlap estimation" stage's real machinery, and it is
also the only honest way to ask whether two strips share ground: both are
pushbroom acquisitions with different roll angles, so comparing them in either
one's pixel grid mixes a genuine terrain offset with a geometric distortion.

The mapping is built by interpolating the delivered geometry lattice. Each
product ships ~122 000 nodes of (Pixel, Scan) -> (lon, lat); those are converted
to projected metres once, and a Delaunay interpolator then gives
(x, y) -> (sample, line) for an arbitrary target grid. That is far cheaper than
inverting the forward mapping per pixel, and it is exactly as accurate as the
lattice it comes from - which is system-level, unrefined, and good to
metres-to-decametres, not to a pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator


@dataclass
class StereoWindow:
    """A resampled view of one strip on a projected grid."""

    image: np.ndarray                 # uint8, (h, w)
    valid: np.ndarray                 # bool, where the strip actually covers
    x0: float
    y0: float
    res_m: float
    product_id: str

    @property
    def extent(self) -> tuple[float, float, float, float]:
        h, w = self.image.shape[:2]
        return (self.x0, self.y0, self.x0 + w * self.res_m, self.y0 + h * self.res_m)

    def pixel_to_stereo(self, col, row):
        return self.x0 + np.asarray(col) * self.res_m, self.y0 + np.asarray(row) * self.res_m

    def stereo_to_pixel(self, x, y):
        return (np.asarray(x) - self.x0) / self.res_m, (np.asarray(y) - self.y0) / self.res_m


def _lattice_interpolators(product, node_step: int = 1):
    """(x, y) -> (sample, line), built from the delivered geometry lattice."""
    g = product.geometry
    X = g.x[::node_step, ::node_step].ravel()
    Y = g.y[::node_step, ::node_step].ravel()
    S, L = np.meshgrid(g.pixels[::node_step], g.scans[::node_step])
    pts = np.column_stack([X, Y])
    f_s = LinearNDInterpolator(pts, S.ravel())
    f_l = LinearNDInterpolator(pts, L.ravel())
    return f_s, f_l


def strip_bounds_stereo(product) -> tuple[float, float, float, float]:
    g = product.geometry
    return float(g.x.min()), float(g.y.min()), float(g.x.max()), float(g.y.max())


def common_bounds(a, b) -> tuple[float, float, float, float] | None:
    ax0, ay0, ax1, ay1 = strip_bounds_stereo(a)
    bx0, by0, bx1, by1 = strip_bounds_stereo(b)
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def reproject(product, bounds, res_m: float = 2.0, *,
              node_step: int = 1, max_side: int = 4000,
              read_step: int | None = None) -> StereoWindow:
    """Resample a strip onto a projected grid covering `bounds`.

    `res_m` is the output ground sampling. Sampling is nearest-neighbour from a
    decimated read of the source, which is appropriate here: we are looking at
    ~2 m/px output from ~0.24 m/px input, so the limiting factor is the geometry
    lattice, not interpolation quality.
    """
    x0, y0, x1, y1 = bounds
    w = int(min(max_side, max(2, round((x1 - x0) / res_m))))
    h = int(min(max_side, max(2, round((y1 - y0) / res_m))))
    res_x = (x1 - x0) / w
    res_y = (y1 - y0) / h

    gx = x0 + (np.arange(w) + 0.5) * res_x
    gy = y0 + (np.arange(h) + 0.5) * res_y
    GX, GY = np.meshgrid(gx, gy)

    f_s, f_l = _lattice_interpolators(product, node_step=node_step)
    S = f_s(GX, GY)
    L = f_l(GX, GY)
    valid = np.isfinite(S) & np.isfinite(L)
    valid &= (S >= 0) & (S < product.samples) & (L >= 0) & (L < product.lines)

    step = read_step or max(1, int(round(res_m / max(product.gsd_m or 0.24, 1e-6) / 2)))
    out = np.zeros((h, w), np.uint8)
    if valid.any():
        # Read the covering source block once (decimated) and index into it,
        # instead of a per-pixel memmap hit.
        s_lo, s_hi = int(np.nanmin(S[valid])), int(np.nanmax(S[valid])) + 1
        l_lo, l_hi = int(np.nanmin(L[valid])), int(np.nanmax(L[valid])) + 1
        block = product.read_tile(s_lo, l_lo, s_hi - s_lo, l_hi - l_lo, step=step)
        bh, bw = block.shape[:2]
        si = np.clip((np.nan_to_num((S - s_lo) / step, nan=0.0)).astype(np.int64), 0, bw - 1)
        li = np.clip((np.nan_to_num((L - l_lo) / step, nan=0.0)).astype(np.int64), 0, bh - 1)
        out[valid] = block[li[valid], si[valid]]

    return StereoWindow(out, valid, x0, y0, float(res_x), product.product_id)


def coarse_align(a, b, *, res_m: float = 4.0, template_m: float = 1500.0,
                 search_m: float = 900.0, lit_dn: int = 25,
                 use_gradient: bool = True) -> dict:
    """Measure the bulk ground offset between two strips' delivered geometry.

    **Why this stage exists.** On the two 2024-11-15 products the delivered
    system-level geolocation disagrees between strips by several hundred metres.
    At 0.24 m/px that is a couple of thousand pixels, so a window cut from one
    strip and its geometry-predicted counterpart in the other barely overlap.
    Feeding those to a matcher produces a handful of spurious correspondences
    and a refusal - and it would be easy, and wrong, to read that as "the
    illumination change defeats SIFT". It is a mislocation, and it has to be
    measured and removed before any statement about matching can be made.

    Method: reproject both strips onto one projected grid, restrict to the
    region where both are valid *and* both are lit (a correlation over the
    shadowed majority measures nothing), and search translations by normalised
    cross-correlation of gradient magnitude. Gradient magnitude because it
    survives a change of illumination direction far better than raw DN.

    The returned `confident` flag requires a real peak *and* a margin over the
    runner-up. A translation is only a translation: with different roll angles
    the residual varies along the strip, so this is a starting prior for the
    per-window stage, not a registration.
    """
    bounds = common_bounds(a, b)
    if bounds is None:
        return {"ok": False, "reason": "footprints do not intersect"}
    wa = reproject(a, bounds, res_m=res_m)
    wb = reproject(b, bounds, res_m=res_m)

    both = wa.valid & wb.valid
    lit = both & (wa.image > lit_dn) & (wb.image > lit_dn)
    if lit.sum() < 20000:
        return {"ok": False, "reason": "too little mutually lit ground to correlate",
                "lit_pixels": int(lit.sum()), "common_pixels": int(both.sum())}

    ys, xs = np.nonzero(lit)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    half = max(40, int(round(template_m / wa.res_m / 2)))
    pad = max(10, int(round(search_m / wa.res_m)))

    H, W = wa.image.shape
    half = min(half, (min(H, W) // 2) - pad - 4)
    if half < 30:
        return {"ok": False, "reason": "grid too small for the requested template"}
    cy = int(np.clip(cy, half + pad + 1, H - half - pad - 2))
    cx = int(np.clip(cx, half + pad + 1, W - half - pad - 2))

    def prep(z):
        z = z.astype(np.float32)
        if use_gradient:
            gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
            z = np.sqrt(gx * gx + gy * gy)
        return cv2.GaussianBlur(z, (0, 0), 1.2)

    Ap, Bp = prep(wa.image), prep(wb.image)
    tmpl = Ap[cy - half:cy + half, cx - half:cx + half]
    win = Bp[cy - half - pad:cy + half + pad, cx - half - pad:cx + half + pad]
    if float(tmpl.std()) < 1e-6 or float(win.std()) < 1e-6:
        return {"ok": False, "reason": "no texture in the correlation window"}

    resp = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(resp)
    dx, dy = loc[0] - pad, loc[1] - pad
    masked = resp.copy()
    r = max(3, pad // 6)
    masked[max(0, loc[1] - r):loc[1] + r + 1, max(0, loc[0] - r):loc[0] + r + 1] = -1.0
    second = float(masked.max())

    confident = bool(peak > 0.10 and (peak - second) > 0.04)
    return {
        "ok": True,
        "a": a.timestamp, "b": b.timestamp,
        "offset_stereo_m": [round(dx * wa.res_m, 2), round(dy * wa.res_m, 2)],
        "offset_m": round(float(np.hypot(dx, dy) * wa.res_m), 2),
        "offset_px_source_gsd": round(float(np.hypot(dx, dy) * wa.res_m
                                            / max(a.gsd_m or 0.24, 1e-6)), 1),
        "dx_grid_px": int(dx), "dy_grid_px": int(dy),
        "grid_res_m": round(wa.res_m, 4),
        "template_m": round(2 * half * wa.res_m, 1),
        "search_m": round(pad * wa.res_m, 1),
        "peak_ncc": round(float(peak), 4),
        "second_peak_ncc": round(second, 4),
        "peak_margin": round(float(peak - second), 4),
        "representation": "gradient magnitude" if use_gradient else "raw DN",
        "lit_pixels": int(lit.sum()), "common_pixels": int(both.sum()),
        "confident": confident,
        "note": ("bulk translation only; with differing roll angles the residual "
                 "varies along the strip, so this is a prior for the per-window "
                 "stage and not itself a registration"),
    }


def align_offset(a_win: StereoWindow, b_win: StereoWindow, *,
                 search_m: float = 600.0, blur: float = 1.5) -> dict:
    """Search for the translation that best aligns two projected views.

    Uses normalised cross-correlation over an explicit search range, which -
    unlike phase correlation - returns a response surface that can be inspected.
    That matters here: on strips this different in illumination the correlation
    peak may be genuinely absent, and a method that always returns *a* shift
    would hide that.

    Reports the peak, its margin over the second-best peak, and whether the
    result should be believed.
    """
    A = a_win.image.astype(np.float32)
    B = b_win.image.astype(np.float32)
    va, vb = a_win.valid, b_win.valid
    both = va & vb
    if both.sum() < 5000:
        return {"ok": False, "reason": "insufficient common coverage",
                "common_pixels": int(both.sum())}

    def prep(z, v):
        z = z.copy()
        m = float(z[v].mean()) if v.any() else 0.0
        z[~v] = m
        z = cv2.GaussianBlur(z, (0, 0), blur)
        s = float(z[v].std()) or 1.0
        return (z - m) / s

    Ap, Bp = prep(A, va), prep(B, vb)
    pad = int(round(search_m / a_win.res_m))
    h, w = Ap.shape
    if pad * 2 + 8 >= min(h, w):
        pad = max(2, min(h, w) // 3)
    tmpl = Ap[pad:h - pad, pad:w - pad]
    if tmpl.size < 1000:
        return {"ok": False, "reason": "search range leaves too small a template"}

    resp = cv2.matchTemplate(Bp, tmpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(resp)
    dx = loc[0] - pad
    dy = loc[1] - pad

    # second-best peak outside a small exclusion zone around the best
    masked = resp.copy()
    r = max(3, pad // 8)
    y0 = max(0, loc[1] - r); y1 = min(resp.shape[0], loc[1] + r + 1)
    x0 = max(0, loc[0] - r); x1 = min(resp.shape[1], loc[0] + r + 1)
    masked[y0:y1, x0:x1] = -1.0
    second = float(masked.max())

    return {
        "ok": True,
        "dx_px": int(dx), "dy_px": int(dy),
        "dx_m": round(dx * a_win.res_m, 2), "dy_m": round(dy * a_win.res_m, 2),
        "offset_m": round(float(np.hypot(dx, dy) * a_win.res_m), 2),
        "peak_ncc": round(float(peak), 4),
        "second_peak_ncc": round(second, 4),
        "peak_margin": round(float(peak - second), 4),
        "search_m": search_m,
        "res_m": a_win.res_m,
        "common_pixels": int(both.sum()),
        "confident": bool(peak > 0.25 and (peak - second) > 0.05),
    }
