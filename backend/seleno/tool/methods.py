"""Candidate correspondence methods, and the rule for choosing between them.

This module exists because of a measured result, not a preference. Phase 2
benchmarked matchers on real illumination change against controlled common-grid geometry and
found that *which method works depends on the regime*:

| regime | evidence | what works |
|---|---|---|
| same sensor, illumination change (NAC vs NAC) | see reconciled Phase 2 corner-error table; learned matching can survive moderate azimuth change | **sparse learned** |
| cross-sensor, anti-correlated (TMC-2 vs SELENE, r = -0.410) | every sparse matcher failed in all three representations tried; dense gradient NCC locked with margins to 0.37 | **dense NCC** |

So the tool runs candidates and takes the geometrically verified winner, and
records which one won and why. Hardcoding either would fail half the cases.

Why gradient magnitude is the dense representation
--------------------------------------------------
It is polarity-blind. Under a Sun-azimuth reversal a slope's lit and shadowed
sides swap, so intensity and signed gradient both invert while gradient
*magnitude* does not. That is why the dense method survives a pair whose raw
correlation is -0.41 and the sparse ones do not: a descriptor sees only its own
reversed patch, while the dense method integrates over the whole overlap.

Sub-pixel
---------
A raw correlation peak is quantised to whole reference pixels - the TMC-2 run
returned dy = 8, 8, 8, 8, 7, 9 pixels exactly, which is the quantisation, not the
measurement. Every dense result here is refined by a parabolic fit through the
correlation peak, and the per-point tie points are refined the same way, so the
reported transform is genuinely sub-pixel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Correspondences:
    """What a method produced, before geometric verification."""

    src: np.ndarray                      # (N, 2) float32, source pixel coords
    ref: np.ndarray                      # (N, 2) float32, reference pixel coords
    confidence: np.ndarray               # (N,) higher is better
    method: str
    kind: str                            # "dense" | "sparse"
    detail: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.src)


# --------------------------------------------------------------------------- #
# representations
# --------------------------------------------------------------------------- #

def gradient_magnitude(a: np.ndarray, valid: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    z = np.where(valid, np.nan_to_num(a.astype(np.float32)), 0.0)
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, 3)
    g = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigma)
    return np.where(valid, g, 0.0)


def to_u8(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    o = np.zeros(a.shape, np.uint8)
    if valid.sum() < 16:
        return o
    lo, hi = np.percentile(a[valid], (2, 98))
    o[valid] = np.clip((a[valid] - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    return o


# --------------------------------------------------------------------------- #
# sub-pixel
# --------------------------------------------------------------------------- #

def parabolic_peak(surface: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Sub-pixel offset of a correlation peak from its integer cell.

    Independent 1-D parabolas through the three samples in each axis. Returns
    (dr, dc) in [-0.5, 0.5]; a degenerate neighbourhood returns 0 rather than a
    division blow-up.
    """
    def one(m, l, rr):
        den = (l - 2.0 * m + rr)
        if abs(den) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (l - rr) / den, -0.5, 0.5))

    h, w = surface.shape
    dr = dc = 0.0
    if 0 < r < h - 1:
        dr = one(surface[r, c], surface[r - 1, c], surface[r + 1, c])
    if 0 < c < w - 1:
        dc = one(surface[r, c], surface[r, c - 1], surface[r, c + 1])
    return dr, dc


# --------------------------------------------------------------------------- #
# dense: global translation then a grid of refined tie points
# --------------------------------------------------------------------------- #

def dense_translation(A, Am, B, Bm, *, max_shift_px=None, representation="grad"):
    """Masked NCC over the whole overlap. Sub-pixel by parabolic fit.

    `A` and `B` must be on one canvas of equal shape.
    """
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

    X = gradient_magnitude(A, Am) if representation == "grad" else np.where(Am, A, 0.0)
    Y = gradient_magnitude(B, Bm) if representation == "grad" else np.where(Bm, B, 0.0)
    X = X - (X[Am].mean() if Am.any() else 0.0)
    Y = Y - (Y[Bm].mean() if Bm.any() else 0.0)
    X, Y = np.where(Am, X, 0.0), np.where(Bm, Y, 0.0)

    xc = cross_correlate_masked(Y, X, Bm.astype(bool), Am.astype(bool),
                                mode="full", overlap_ratio=0.2)
    off_r, off_c = X.shape[0] - 1, X.shape[1] - 1
    rr, cc = np.indices(xc.shape)
    dr, dc = rr - off_r, cc - off_c
    search = np.ones(xc.shape, bool)
    if max_shift_px is not None:
        search = (np.abs(dr) <= max_shift_px) & (np.abs(dc) <= max_shift_px)
    if not search.any():
        return None
    masked = np.where(search, xc, -np.inf)
    pr, pc = np.unravel_index(int(np.argmax(masked)), xc.shape)
    peak = float(xc[pr, pc])
    second = masked.copy()
    second[max(0, pr - 6):pr + 7, max(0, pc - 6):pc + 7] = -np.inf
    runner = float(second.max()) if np.isfinite(second).any() else 0.0

    sub_r, sub_c = parabolic_peak(xc, pr, pc)
    return {"dx": float(dc[pr, pc]) + sub_c, "dy": float(dr[pr, pc]) + sub_r,
            "peak": round(peak, 4), "margin": round(peak - runner, 4),
            "subpixel_dx": round(sub_c, 3), "subpixel_dy": round(sub_r, 3),
            "at_search_edge": bool(max_shift_px is not None
                                   and (abs(dr[pr, pc]) >= max_shift_px - 1
                                        or abs(dc[pr, pc]) >= max_shift_px - 1))}


def overlap_bbox(mask, pad=0):
    """Bounding box of a boolean mask, or None."""
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    h, w = mask.shape
    return (max(0, int(rows[0]) - pad), max(0, int(cols[0]) - pad),
            min(h, int(rows[-1]) + 1 + pad), min(w, int(cols[-1]) + 1 + pad))


def grid_tiepoints(A, Am, B, Bm, dx0, dy0, *, grid=(8, 8), patch=48, search=12,
                   min_texture=0.01, min_peak=0.25, bbox=None):
    """Local NCC tie points on a uniform grid, each refined to sub-pixel.

    This is what turns a dense lock into the *match points* the problem statement
    asks for, and it delivers the uniform distribution it also asks for: the
    seeds are a grid by construction, not wherever a detector happened to fire.

    `bbox` confines the grid to the region the two images actually share. A strip
    crossing a quarter of a map tile has nothing to offer in the other three
    quarters, and spreading the seeds over the whole frame there just wastes most
    of them - the first TMC-2 run produced 12 tie points for exactly that reason.
    """
    h, w = A.shape
    gy, gx = grid
    half = patch // 2
    r0, c0, r1, c1 = bbox if bbox else (0, 0, h, w)
    src, ref, conf = [], [], []
    Ag = gradient_magnitude(A, Am)
    Bg = gradient_magnitude(B, Bm)
    for j in range(gy):
        for i in range(gx):
            cy = int(r0 + (j + 0.5) * (r1 - r0) / gy)
            cx = int(c0 + (i + 0.5) * (c1 - c0) / gx)
            if not (half + search <= cy < h - half - search
                    and half + search <= cx < w - half - search):
                continue
            t = Ag[cy - half:cy + half, cx - half:cx + half]
            tm = Am[cy - half:cy + half, cx - half:cx + half]
            if tm.mean() < 0.8 or t.std() < min_texture:
                continue
            ry, rx = int(round(cy + dy0)), int(round(cx + dx0))
            if not (half + search <= ry < h - half - search
                    and half + search <= rx < w - half - search):
                continue
            win = Bg[ry - half - search:ry + half + search,
                     rx - half - search:rx + half + search]
            wm = Bm[ry - half - search:ry + half + search,
                    rx - half - search:rx + half + search]
            if wm.mean() < 0.6 or win.std() < min_texture:
                continue
            try:
                resp = cv2.matchTemplate(win.astype(np.float32), t.astype(np.float32),
                                         cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            _, peak, _, loc = cv2.minMaxLoc(resp)
            if peak < min_peak:
                continue
            sr, sc = parabolic_peak(resp, loc[1], loc[0])
            off_x = loc[0] - search + sc
            off_y = loc[1] - search + sr
            src.append([cx, cy])
            ref.append([rx + off_x, ry + off_y])
            conf.append(peak)
    if not src:
        return None
    return Correspondences(np.asarray(src, np.float32), np.asarray(ref, np.float32),
                           np.asarray(conf, np.float32), "dense-ncc", "dense",
                           {"grid": list(grid), "patch_px": patch, "search_px": search,
                            "seed_dx": round(float(dx0), 3), "seed_dy": round(float(dy0), 3)})


# --------------------------------------------------------------------------- #
# sparse
# --------------------------------------------------------------------------- #

def sparse(A, Am, B, Bm, name: str, **kw) -> Correspondences | None:
    from .. import matchers
    Au, Bu = to_u8(A, Am), to_u8(B, Bm)
    try:
        r = matchers.run_matcher(name, Au, Bu, **kw)
    except Exception as exc:
        return Correspondences(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                               np.zeros((0,), np.float32), name, "sparse",
                               {"error": "%s: %s" % (type(exc).__name__, exc)})
    if len(r.kp_src) == 0:
        return Correspondences(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                               np.zeros((0,), np.float32), name, "sparse", dict(r.detail))
    # drop matches whose endpoints fall on nodata on either side
    si = np.clip(r.kp_src[:, ::-1].astype(int), 0, np.array(A.shape) - 1)
    ri = np.clip(r.kp_ref[:, ::-1].astype(int), 0, np.array(B.shape) - 1)
    keep = Am[si[:, 0], si[:, 1]] & Bm[ri[:, 0], ri[:, 1]]
    return Correspondences(r.kp_src[keep], r.kp_ref[keep], r.score[keep],
                           name, "sparse", dict(r.detail))


# --------------------------------------------------------------------------- #
# pair character - measured, not assumed
# --------------------------------------------------------------------------- #

def pair_character(A, Am, B, Bm, src_profile, ref_profile, gsd_ratio) -> dict:
    """Measurable properties that predict which regime a pair is in."""
    both = Am & Bm
    r = None
    if both.sum() > 500:
        a, b = A[both].astype(np.float64), B[both].astype(np.float64)
        if a.std() > 1e-9 and b.std() > 1e-9:
            r = float(np.corrcoef(a, b)[0, 1])
    return {"cross_correlation": None if r is None else round(r, 4),
            "anti_correlated": bool(r is not None and r < -0.05),
            "same_sensor": src_profile == ref_profile,
            "sensor_pair": "%s-%s" % (src_profile, ref_profile),
            "scale_ratio": None if gsd_ratio is None else round(gsd_ratio, 4),
            "overlap_fraction": round(float(both.mean()), 4)}


def candidate_plan(character: dict, available: dict) -> list:
    """Which methods to try, most promising first.

    Ordering is from the Phase 2 measurements, but every candidate still runs and
    the winner is decided by geometric verification. The ordering only decides
    what gets tried first, never what gets believed.
    """
    dense = ["dense-ncc"]
    learned = [n for n in ("disk_lightglue",) if available.get(n)]
    classical = ["sift", "akaze", "orb"]
    if character.get("anti_correlated"):
        # measured: every sparse matcher failed on the anti-correlated TMC-2 pair
        return dense + learned + classical
    if character.get("same_sensor"):
        # Ordering informed by the reconciled Phase 2 corner-error benchmark.
        return learned + dense + classical
    return dense + learned + classical
