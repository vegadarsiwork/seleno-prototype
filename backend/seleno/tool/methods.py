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


def refine_quantised(c, A, Am, B, Bm, *, patch=32, search=4, min_texture=0.01,
                     min_peak=0.25):
    """Re-measure correspondences whose displacement is a whole number of pixels.

    A detector that puts keypoints on the pixel lattice (ORB) reports whole-pixel
    displacements, and inside a window already prewarped by a model most of them
    are exactly zero - 63-69% of ORB fine points on the OHRC controls. A zero-
    displacement point lies exactly ON the prewarp model instead of measuring
    anything: fits are pulled onto that model, and a held-out point built that way
    scores exactly 0 against it. Each such point is re-measured by the same
    gradient NCC and parabolic peak as the dense tie points (source seed kept,
    reference position refined), with the largest patch up to `patch` that fits
    inside both windows. A re-measured offset of exactly zero is kept: it is then a
    measurement (a same-file pair really is at zero). A point NCC cannot measure is
    dropped if it sits exactly on the prewarp (zero displacement) and otherwise
    kept as the matcher gave it. Dense tie points are already this NCC
    measurement, and sparse points already below a pixel are returned unchanged.
    """
    if c is None or len(c) == 0 or c.kind != "sparse":
        return c
    src = c.src.astype(np.float64).copy()
    ref = c.ref.astype(np.float64).copy()
    d = ref - src
    quantised = np.all(np.abs(d - np.round(d)) < 1e-6, axis=1)
    detail = dict(c.detail, quantised=int(quantised.sum()), refined=0, dropped=0)
    if not quantised.any():
        return Correspondences(c.src, c.ref, c.confidence, c.method, c.kind, detail)
    Ag, Bg = gradient_magnitude(A, Am), gradient_magnitude(B, Bm)
    keep = np.ones(len(src), bool)
    refined = np.zeros(len(src), bool)
    for k in np.flatnonzero(quantised):
        cx, cy = int(round(src[k, 0])), int(round(src[k, 1]))
        rx = int(round(ref[k, 0] + cx - src[k, 0]))
        ry = int(round(ref[k, 1] + cy - src[k, 1]))
        half = min(patch // 2, cy, A.shape[0] - cy, cx, A.shape[1] - cx,
                   ry - search, B.shape[0] - ry - search, rx - search, B.shape[1] - rx - search)
        ok = half >= 6
        if ok:
            t = Ag[cy - half:cy + half, cx - half:cx + half]
            win = Bg[ry - half - search:ry + half + search, rx - half - search:rx + half + search]
            ok = (Am[cy - half:cy + half, cx - half:cx + half].mean() >= 0.8
                  and Bm[ry - half - search:ry + half + search,
                         rx - half - search:rx + half + search].mean() >= 0.6
                  and t.std() >= min_texture and win.std() >= min_texture)
        if ok:
            resp = cv2.matchTemplate(win.astype(np.float32), t.astype(np.float32),
                                     cv2.TM_CCOEFF_NORMED)
            _, peak, _, loc = cv2.minMaxLoc(resp)
            ok = peak >= min_peak and 0 < loc[0] < 2 * search and 0 < loc[1] < 2 * search
        if ok:
            sr, sc = parabolic_peak(resp, loc[1], loc[0])
            src[k] = (cx, cy)
            ref[k] = (rx + loc[0] - search + sc, ry + loc[1] - search + sr)
            refined[k] = True
        elif np.all(d[k] == 0):
            keep[k] = False              # unmeasured and exactly on the prewarp
    detail.update(refined=int(refined.sum()), dropped=int((~keep).sum()),
                  kept_unrefined=int((quantised & keep & ~refined).sum()))
    return Correspondences(src[keep].astype(np.float32), ref[keep].astype(np.float32),
                           np.asarray(c.confidence)[keep], c.method, c.kind, detail)


def refine_correspondences(c, A, Am, B, Bm, *, patch=33, search=4,
                           min_texture=0.002, min_peak=0.55,
                           forward_backward=0.35, keep_unmeasured=False):
    """Measure local translations on prealigned images, preserving source seeds.

    Every returned endpoint is measured from image content, including seeds on
    the prewarp and fractional detector positions. NCC initializes ECC rather
    than treating the integer correlation cell as the measurement. Correlated
    intensity (with either polarity) is preferred; gradient magnitude is the
    fallback for local photometric changes. A separate reverse measurement must
    return to the source seed. Textureless, ambiguous and boundary peaks are
    rejected, not retained as zero-error observations of the initial model.

    Inputs may be raw DN. ``patch`` is rounded up to an odd size. Source seeds
    are never rounded or moved, so their spatial fit/validation/test assignment
    remains unchanged. This function assumes local geometry has been prewarped;
    it does not replace the coarse rotation/scale estimation.
    """
    if c is None or not len(c):
        return c
    patch = max(13, int(patch) | 1)
    search = max(1, int(search))
    half = patch // 2

    def normalise(z, valid):
        z = np.asarray(z, np.float32)
        good = valid & np.isfinite(z)
        if not good.any():
            return np.zeros(z.shape, np.float32)
        lo, hi = np.percentile(z[good], (2, 98))
        return np.where(good, (z - lo) / max(hi - lo, 1e-8), 0).astype(np.float32)

    Aa, Bb = normalise(A, Am), normalise(B, Bm)
    Ag, Bg = gradient_magnitude(Aa, Am), gradient_magnitude(Bb, Bm)
    masks = (np.asarray(Am, np.float32), np.asarray(Bm, np.float32))
    refs = np.asarray(c.ref, np.float64).copy()
    measured = np.zeros(len(c), bool)
    confidence = np.asarray(c.confidence, np.float32).copy()
    counts = {"intensity": 0, "inverted_intensity": 0, "gradient": 0,
              "boundary": 0, "texture": 0, "peak": 0, "ecc": 0,
              "forward_backward": 0}

    def cut(z, point, radius):
        x, y = map(float, point)
        if not (radius + 1 <= x < z.shape[1] - radius - 1
                and radius + 1 <= y < z.shape[0] - radius - 1):
            return None
        return cv2.getRectSubPix(z, (2 * radius + 1, 2 * radius + 1), (x, y))

    def measure(t, win, allow_inverse, search):
        if t is None or win is None:
            return None, "boundary"
        if t.std() < min_texture or win.std() < min_texture:
            return None, "texture"
        response = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(response).all():
            return None, "peak"
        scores = np.abs(response) if allow_inverse else response
        iy, ix = np.unravel_index(int(np.argmax(scores)), scores.shape)
        peak = float(scores[iy, ix])
        if peak < min_peak or ix in (0, 2 * search) or iy in (0, 2 * search):
            return None, "peak"
        sign = -1.0 if response[iy, ix] < 0 else 1.0
        sr, sc = parabolic_peak(scores, iy, ix)
        initial = np.float32([[1, 0, ix + sc], [0, 1, iy + sr]])
        try:
            cc, warp = cv2.findTransformECC(
                t, sign * win, initial, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-5),
                None, 3)
        except cv2.error:
            return None, "ecc"
        delta = warp[:, 2].astype(np.float64) - search
        if (not np.isfinite(delta).all() or not np.isfinite(cc)
                or cc < min_peak or np.any(np.abs(delta) > search - 0.5)):
            return None, "ecc"
        return (delta, float(cc), sign), None

    def fitting(source, reference):
        """Largest patch, then search, that lies wholly inside both valid masks.

        A seed near an image border or a mask edge is measured with less
        support rather than dropped; below a 15 px patch it is dropped.
        """
        hh, ss = half, search
        while True:
            sm = cut(masks[0], source, hh)
            rm = cut(masks[1], reference, hh + ss)
            if sm is not None and rm is not None and sm.min() >= 0.999 and rm.min() >= 0.999:
                return hh, ss
            if hh > 7:
                hh = max(7, int(hh * 0.75))
            elif ss > 3:
                ss = max(3, ss // 2)
            else:
                return None, None

    reduced = 0
    for k, (source, reference) in enumerate(zip(c.src, c.ref)):
        hh, ss = fitting(source, reference)
        if hh is None:
            counts["boundary"] += 1
            continue
        reduced += int(hh < half or ss < search)
        # Correlation is measured separately for each patch, not inferred from
        # a whole-image mean which can hide contrast changes across a strip.
        intensity, why = measure(cut(Aa, source, hh),
                                 cut(Bb, reference, hh + ss), True, ss)
        if intensity is not None and intensity[1] >= max(0.7, min_peak):
            result, x, y, inverse = intensity, Aa, Bb, True
            representation = "intensity" if result[2] > 0 else "inverted_intensity"
        else:
            result, why = measure(cut(Ag, source, hh),
                                  cut(Bg, reference, hh + ss), False, ss)
            x, y, inverse, representation = Ag, Bg, False, "gradient"
        if result is None:
            counts[why] += 1
            continue
        target = np.asarray(reference, np.float64) + result[0]
        reverse, why = measure(cut(y, target, hh), cut(x, source, hh + ss), inverse, ss)
        if reverse is None or np.linalg.norm(reverse[0]) > forward_backward:
            counts["forward_backward"] += 1
            continue
        refs[k] = target
        measured[k] = True
        confidence[k] = min(result[1], reverse[1])
        counts[representation] += 1
    keep = np.ones(len(c), bool) if keep_unmeasured else measured
    detail = dict(c.detail, refined=int(measured.sum()), dropped=int((~keep).sum()),
                  kept_unrefined=int((keep & ~measured).sum()),
                  refinement="local-ecc-forward-backward",
                  refinement_counts=dict(counts, reduced_patch=reduced),
                  source_seeds_preserved=True)
    # Preserve auxiliary per-point arrays when refining an already native set.
    for key in ("back_x", "back_y"):
        if key in detail and len(detail[key]) == len(c):
            detail[key] = np.asarray(detail[key])[keep].tolist()
    return Correspondences(np.asarray(c.src)[keep].copy(), refs[keep], confidence[keep],
                           c.method, c.kind, detail)


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
    matcher = name
    representation = "intensity"
    if name in ("sift-gradient", "sift-phase"):
        matcher = "sift"
        representation = name.split("-", 1)[1]
        if representation == "gradient":
            A, B = gradient_magnitude(A, Am), gradient_magnitude(B, Bm)
        else:
            from ..phasecong import phase_congruency
            # Contrast-normalised Fourier phase is insensitive to radiometric
            # polarity. SIFT supplies actual spatial rotation/scale invariance;
            # cyclic permutation of orientation bins alone does not do that.
            A = phase_congruency(to_u8(A, Am).astype(np.float32) / 255, nscale=3)[0]
            B = phase_congruency(to_u8(B, Bm).astype(np.float32) / 255, nscale=3)[0]
        kw.setdefault("ratio", 0.85)
    Au, Bu = to_u8(A, Am), to_u8(B, Bm)
    try:
        r = matchers.run_matcher(matcher, Au, Bu, **kw)
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
                           name, "sparse", dict(r.detail, representation=representation))


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
    classical = ["sift", "sift-gradient", "sift-phase", "akaze", "orb"]
    if character.get("anti_correlated"):
        # measured: every sparse matcher failed on the anti-correlated TMC-2 pair
        return dense + learned + classical
    if character.get("same_sensor"):
        # Ordering informed by the reconciled Phase 2 corner-error benchmark.
        return learned + dense + classical
    return dense + learned + classical
