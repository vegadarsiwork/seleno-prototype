"""Masked translation estimation on a shared projected grid.

When two rasters already sit on the same map projection, the only free
parameter is a translation, and fitting anything richer is how a pipeline
manufactures confident nonsense: `results/experiments/P0_FINDINGS.md` records a
4-DoF similarity fitted to 2-3 points returning scales of 0.157-2.68 on images
that must be related by scale 1, rotation 0.

So: one-parameter models only, two independent estimators, and a hard
requirement that they agree.

* `masked_ncc_shift` - masked normalised cross-correlation over the whole
  overlap, via `skimage`'s tested FFT implementation. Handles the large nodata
  fraction on both sides (an OHRC ribbon is ~20% of its bounding box; a polar
  NAC mosaic is mostly shadow).
* `feature_vote_shift` - detector + mutual ratio test + a Hough vote on the
  per-match displacement. A one-point minimal model, so unrelated matches
  cannot conspire into a plausible transform.

Both report metres in the plane, both report a confidence that is measured
against a null rather than asserted, and `agree` is what callers should gate on.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Shift:
    """A displacement in plane metres.

    Contract, used identically by every estimator here: ``(dx, dy)`` is what must
    be **added to A's coordinates** to reach the matching location in B.  So if A
    is a source image placed by its own geometry and B is a geodetic reference,
    ``(dx, dy)`` is the correction that geometry needs.
    """

    dx: float | None
    dy: float | None
    method: str
    confidence: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def magnitude(self) -> float | None:
        if self.dx is None:
            return None
        return math.hypot(self.dx, self.dy)

    @property
    def bearing_deg(self) -> float | None:
        """Direction of the displacement in the plane, degrees CCW from +x."""
        if self.dx is None:
            return None
        return math.degrees(math.atan2(self.dy, self.dx)) % 360.0


def _prep(img, valid, kind="grad"):
    """Normalise to zero-mean float over the valid region.

    `grad` is the default because brightness is not the shared quantity between
    a Chandrayaan-2 strip and either a NAC mosaic or a synthetic relight - the
    *edges* of shadows are.
    """
    z = np.where(valid, np.nan_to_num(img.astype(np.float32)), 0.0)
    if kind == "grad":
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, 3)
        z = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.2)
    elif kind == "binary":
        m = valid & np.isfinite(img)
        thr = np.percentile(z[m], 60) if m.sum() > 50 else 0.0
        z = (z > thr).astype(np.float32)
    if valid.sum() > 10:
        z = z - z[valid].mean()
    return np.where(valid, z, 0.0).astype(np.float32)


def place(img, valid, gx, gy, out_gx, out_gy):
    """Resample an array onto another grid of the same resolution, by nearest.

    `cross_correlate_masked` needs both inputs on one canvas. Doing that here,
    from each array's own pixel-centre coordinates, means a caller can never
    silently compare two grids that differ by half a pixel or by an origin.
    """
    out = np.zeros((len(out_gy), len(out_gx)), np.float32)
    msk = np.zeros(out.shape, bool)
    res_x = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
    res_y = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
    ci = np.rint((out_gx - gx[0]) / res_x).astype(np.int64)
    ri = np.rint((out_gy - gy[0]) / res_y).astype(np.int64)
    cok = (ci >= 0) & (ci < img.shape[1])
    rok = (ri >= 0) & (ri < img.shape[0])
    if not (cok.any() and rok.any()):
        return out, msk
    sub = img[np.ix_(ri[rok], ci[cok])]
    sv = valid[np.ix_(ri[rok], ci[cok])]
    block = np.ix_(np.nonzero(rok)[0], np.nonzero(cok)[0])
    out[block] = np.nan_to_num(sub)
    msk[block] = sv & np.isfinite(sub)
    return out, msk


def union_grid(*grids, res):
    """Smallest grid at `res` covering every (gx, gy) passed in."""
    x0 = min(float(g[0][0]) for g in grids) - res / 2
    x1 = max(float(g[0][-1]) for g in grids) + res / 2
    y0 = min(float(g[1][0]) for g in grids) - res / 2
    y1 = max(float(g[1][-1]) for g in grids) + res / 2
    w = max(2, int(round((x1 - x0) / res)))
    h = max(2, int(round((y1 - y0) / res)))
    return x0 + (np.arange(w) + 0.5) * res, y0 + (np.arange(h) + 0.5) * res


def masked_ncc_shift(a, a_valid, b, b_valid, res_m, *, kind="grad",
                     max_shift_m=None, overlap_ratio=0.2, exclude_px=6) -> Shift:
    """Shift that best moves `a` onto `b`, in plane metres.

    `a` and `b` must be sampled on the same grid resolution; they need not be
    the same size or cover the same ground.
    """
    from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

    if a.shape != b.shape:
        raise ValueError("masked_ncc_shift needs both arrays on one canvas; got %s and %s. "
                         "Use place()/union_grid() first." % (a.shape, b.shape))
    A = _prep(a, a_valid, kind)
    B = _prep(b, b_valid, kind)
    xc = cross_correlate_masked(B, A, b_valid.astype(bool), a_valid.astype(bool),
                                mode="full", overlap_ratio=overlap_ratio)
    # index (i, j) of `full` corresponds to shifting A by (i - (hA-1), j - (wA-1))
    off_r, off_c = A.shape[0] - 1, A.shape[1] - 1
    rr, cc = np.indices(xc.shape)
    dr = rr - off_r
    dc = cc - off_c
    search = np.ones(xc.shape, bool)
    if max_shift_m is not None:
        lim = max_shift_m / res_m
        search = (np.abs(dr) <= lim) & (np.abs(dc) <= lim)
    if not search.any():
        return Shift(None, None, "masked-ncc", 0.0, {"reason": "empty search window"})

    masked = np.where(search, xc, -np.inf)
    k = int(np.argmax(masked))
    pr, pc = np.unravel_index(k, xc.shape)
    peak = float(xc[pr, pc])

    second = masked.copy()
    second[max(0, pr - exclude_px):pr + exclude_px + 1,
           max(0, pc - exclude_px):pc + exclude_px + 1] = -np.inf
    runner = float(second.max()) if np.isfinite(second).any() else 0.0

    # rows increase with +y, columns with +x. The sign is pinned by `_selftest`,
    # which builds B by rolling A by a known amount and must recover it with the
    # same sign that `feature_vote_shift` returns from p2 - p1. An inverted sign
    # here is indistinguishable from a real geolocation offset, which is exactly
    # the quantity this project exists to measure.
    dy = float(dr[pr, pc]) * res_m
    dx = float(dc[pr, pc]) * res_m
    at_edge = bool(max_shift_m is not None
                   and (abs(dr[pr, pc]) >= lim - 1 or abs(dc[pr, pc]) >= lim - 1))
    return Shift(dx, dy, "masked-ncc", round(peak - runner, 4),
                 {"peak": round(peak, 4), "second": round(runner, 4),
                  "margin": round(peak - runner, 4), "at_search_edge": at_edge,
                  "representation": kind})


def feature_vote_shift(a, a_valid, b, b_valid, res_m, *, detector="sift",
                       kind="grad", ratio=0.85, nfeat=40000, edge_guard_px=8,
                       bin_m=None, min_votes=4, min_vote_fraction=0.12) -> Shift:
    """Detector matches reduced to a translation by a Hough vote.

    `min_votes` exists because the tallest bin of a histogram always has *some*
    height: a peak of one match out of nineteen is the arg-max of noise, and
    returning it as an estimate is how a consensus check gets poisoned by a
    method that had nothing to say.
    """
    bin_m = bin_m or max(4.0 * res_m, 50.0)
    A = _prep(a, a_valid, kind)
    B = _prep(b, b_valid, kind)

    def u8(z, m):
        if m.sum() < 50:
            return np.zeros(z.shape, np.uint8)
        lo, hi = np.percentile(z[m], (2, 98))
        o = np.clip((z - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
        o[~m] = 0
        return o

    Au, Bu = u8(A, a_valid), u8(B, b_valid)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_guard_px + 1,) * 2)
    core_a = cv2.erode(a_valid.astype(np.uint8), ker)
    core_b = cv2.erode(b_valid.astype(np.uint8), ker)
    if core_a.sum() < 400 or core_b.sum() < 400:
        return Shift(None, None, "%s-vote" % detector, 0.0, {"reason": "valid area too small"})

    if detector == "sift":
        det = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.02, edgeThreshold=12)
        norm = cv2.NORM_L2
    elif detector == "akaze":
        det = cv2.AKAZE_create(threshold=0.0008)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.ORB_create(nfeatures=nfeat, fastThreshold=8, nlevels=10)
        norm = cv2.NORM_HAMMING
    k1, d1 = det.detectAndCompute(Au, core_a * 255)
    k2, d2 = det.detectAndCompute(Bu, core_b * 255)
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return Shift(None, None, "%s-vote" % detector, 0.0,
                     {"reason": "too few descriptors", "n_src": 0 if d1 is None else len(d1),
                      "n_ref": 0 if d2 is None else len(d2)})
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
    if len(pairs) < 5:
        return Shift(None, None, "%s-vote" % detector, 0.0,
                     {"reason": "too few mutual matches", "mutual": len(pairs),
                      "n_src": len(k1), "n_ref": len(k2)})

    p1 = np.float32([k1[i].pt for i, _ in pairs])
    p2 = np.float32([k2[j].pt for _, j in pairs])
    d = (p2 - p1) * res_m                       # same grid, so pixels -> metres directly
    key = np.round(d / bin_m).astype(np.int64)
    c = Counter(map(tuple, key))
    best, bestn = None, -1
    for k in c:
        n = sum(c.get((k[0] + i, k[1] + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1))
        if n > bestn:
            bestn, best = n, k
    sel = (np.abs(key[:, 0] - best[0]) <= 1) & (np.abs(key[:, 1] - best[1]) <= 1)
    span_x = max(d[:, 0].max() - d[:, 0].min(), bin_m)
    span_y = max(d[:, 1].max() - d[:, 1].min(), bin_m)
    expected = len(d) * (3 * bin_m) ** 2 / (span_x * span_y)
    sig = bestn / max(expected, 1e-6)
    if bestn < min_votes or bestn < min_vote_fraction * len(d):
        return Shift(None, None, "%s-vote" % detector, 0.0,
                     {"reason": "peak too weak (%d votes of %d, need >=%d and >=%.0f%%)"
                                % (bestn, len(d), min_votes, 100 * min_vote_fraction),
                      "votes": int(bestn), "mutual": int(len(d))})
    # columns are +x, rows are +y
    dx = float(d[sel, 0].mean())
    dy = float(d[sel, 1].mean())
    return Shift(dx, dy, "%s-vote" % detector, round(float(sig), 2),
                 {"votes": int(bestn), "mutual": int(len(d)),
                  "expected_by_chance": round(float(expected), 2),
                  "significance": round(float(sig), 2),
                  "scatter_m": round(float(np.hypot(d[sel, 0].std(), d[sel, 1].std())), 1),
                  "n_src": len(k1), "n_ref": len(k2), "representation": kind})


def agree(shifts, tol_m: float) -> dict:
    """Do independent estimators land on the same translation?

    A single estimator's own score is not evidence - the CM_AVG spike produced
    two confident false locks (`ROADMAP.md` P0.1a). Agreement between different
    algorithm families is the gate.

    The consensus is the centroid of the **largest mutually-consistent cluster**,
    not the median of every estimate. With four estimators, one of which is
    wandering, a median is dragged off the answer the other three agree on; and
    the spread that matters is the spread *within* the cluster, not the distance
    to the outlier that was rejected.
    """
    good = [s for s in shifts if s.dx is not None]
    dropped = [{"method": s.method, "reason": s.detail.get("reason", "no estimate")}
               for s in shifts if s.dx is None]
    if len(good) < 2:
        return {"agree": False, "n": len(good), "reason": "fewer than two estimates",
                "dropped": dropped}
    pts = np.array([[s.dx, s.dy] for s in good])
    best_idx: list[int] = []
    for i in range(len(good)):
        near = [j for j in range(len(good))
                if math.hypot(*(pts[j] - pts[i])) <= tol_m]
        if len(near) > len(best_idx):
            best_idx = near
    members = [good[j] for j in best_idx]
    cx, cy = float(pts[best_idx, 0].mean()), float(pts[best_idx, 1].mean())
    spread = float(np.hypot(pts[best_idx, 0] - cx, pts[best_idx, 1] - cy).max())
    families = {m.method.split("[")[0].split("-")[0] for m in members}
    return {"agree": bool(len(best_idx) >= 2 and len(families) >= 2),
            "n": len(good), "n_in_cluster": len(best_idx),
            "families_in_cluster": sorted(families),
            "consensus_dx": round(cx, 1), "consensus_dy": round(cy, 1),
            "consensus_magnitude_m": round(math.hypot(cx, cy), 1),
            "consensus_bearing_deg": round(math.degrees(math.atan2(cy, cx)) % 360.0, 1),
            "cluster_spread_m": round(spread, 1), "tol_m": tol_m,
            "cluster": {m.method: (round(m.dx, 1), round(m.dy, 1)) for m in members},
            "outliers": {s.method: (round(s.dx, 1), round(s.dy, 1))
                         for k, s in enumerate(good) if k not in best_idx},
            "dropped": dropped}


# --------------------------------------------------------------------------- #
# self-test: recover a known shift
# --------------------------------------------------------------------------- #

def _selftest(verbose=True):
    """B is A displaced by a known (+dx, +dy); both estimators must recover it.

    Built with an explicit roll so the contract is unambiguous: a feature at A
    pixel (col, row) sits at B pixel (col + dx, row + dy).
    """
    rng = np.random.default_rng(7)
    scene = cv2.GaussianBlur(rng.random((512, 512)).astype(np.float32), (0, 0), 2.0)
    worst_ncc, worst_vote = 0.0, 0.0
    for (dx, dy) in [(0, 0), (-23, 17), (11, -31), (40, 40)]:
        A = scene.copy()
        B = np.roll(np.roll(scene, dy, axis=0), dx, axis=1)
        va = np.ones(A.shape, bool); va[:40] = False
        vb = np.ones(B.shape, bool); vb[:, :40] = False
        n = masked_ncc_shift(A, va, B, vb, res_m=1.0, kind="raw", max_shift_m=80)
        v = feature_vote_shift(A, va, B, vb, res_m=1.0, detector="sift", kind="raw",
                               bin_m=3.0)
        e_n = math.hypot(n.dx - dx, n.dy - dy)
        e_v = math.hypot(v.dx - dx, v.dy - dy) if v.dx is not None else float("nan")
        worst_ncc = max(worst_ncc, e_n)
        if v.dx is not None:
            worst_vote = max(worst_vote, e_v)
        if verbose:
            print("   true (%+4d, %+4d)   ncc (%+7.1f, %+7.1f) err %5.2f   "
                  "vote (%+7.1f, %+7.1f) err %5.2f"
                  % (dx, dy, n.dx, n.dy, e_n,
                     v.dx if v.dx is not None else float("nan"),
                     v.dy if v.dy is not None else float("nan"), e_v))
    return worst_ncc, worst_vote


if __name__ == "__main__":
    print("sign contract: (dx, dy) added to A's coordinates reaches B\n")
    wn, wv = _selftest()
    print("\nworst masked-NCC error %.2f px, worst feature-vote error %.2f px  -> %s"
          % (wn, wv, "PASS" if (wn < 1e-6 and wv < 2.0) else "FAIL"))
