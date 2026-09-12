"""Sub-pixel correspondence refinement.

Descriptor keypoints are localised to roughly a pixel. Once a coarse transform
exists, each correspondence can be improved by matching a small patch directly:
warp the source neighbourhood into the reference frame with the current
transform, cross-correlate it against the reference over a few pixels of search,
and fit a paraboloid to the correlation peak.

This buys real accuracy on well-textured tiles and, importantly, it *fails
visibly* on the ones this dataset is full of: a patch with five distinct DN
values has no correlation peak to fit, and the refinement is rejected rather
than allowed to invent a displacement. `refine` reports how many
correspondences it actually improved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RefineResult:
    p_src: np.ndarray
    p_ref: np.ndarray
    moved_px: np.ndarray                 # per-match displacement applied
    accepted: np.ndarray                 # bool per match
    detail: dict = field(default_factory=dict)

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum())


def _parabola_peak(c: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel offset of the maximum of a 3x3 correlation neighbourhood."""
    cy, cx = 1, 1
    d2x = c[cy, cx - 1] - 2 * c[cy, cx] + c[cy, cx + 1]
    d2y = c[cy - 1, cx] - 2 * c[cy, cx] + c[cy + 1, cx]
    dx = 0.0 if abs(d2x) < 1e-12 else 0.5 * (c[cy, cx - 1] - c[cy, cx + 1]) / d2x
    dy = 0.0 if abs(d2y) < 1e-12 else 0.5 * (c[cy - 1, cx] - c[cy + 1, cx]) / d2y
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1)), float(c[cy, cx])


def refine(img_src: np.ndarray, img_ref: np.ndarray,
           p_src: np.ndarray, p_ref: np.ndarray, H: np.ndarray,
           *, patch: int = 21, search: int = 3,
           min_ncc: float = 0.35, min_std: float = 2.0,
           max_shift_px: float = 4.0) -> RefineResult:
    """Refine reference-side positions by local normalised cross-correlation.

    Parameters that matter:

    ``min_ncc``     correlation floor. Below it the match is left where the
                    detector put it - a flat patch would otherwise be dragged to
                    whichever noise pixel happened to correlate best.
    ``min_std``     patch texture floor, applied to both sides. This is what
                    stops the routine from "refining" shadowed patches.
    ``max_shift_px`` refusal threshold. A correction larger than this is not a
                    refinement, it is a different match.
    """
    n = len(p_src)
    out_src = p_src.astype(np.float64).copy()
    out_ref = p_ref.astype(np.float64).copy()
    moved = np.zeros(n, np.float64)
    accepted = np.zeros(n, bool)
    if n == 0 or H is None:
        return RefineResult(out_src, out_ref, moved, accepted,
                            {"reason": "nothing to refine"})

    half = patch // 2
    hs, ws = img_src.shape[:2]
    hr, wr = img_ref.shape[:2]
    src_f = img_src.astype(np.float32)
    ref_f = img_ref.astype(np.float32)
    rejected = {"edge": 0, "flat": 0, "low_ncc": 0, "big_shift": 0}

    for i in range(n):
        sx, sy = out_src[i]
        rx, ry = out_ref[i]
        if not (half + 1 <= sx < ws - half - 1 and half + 1 <= sy < hs - half - 1):
            rejected["edge"] += 1
            continue
        need = half + search + 1
        if not (need <= rx < wr - need and need <= ry < hr - need):
            rejected["edge"] += 1
            continue

        # Resample the source neighbourhood through H so both patches share
        # rotation and scale; correlating unaligned geometry biases the peak.
        #
        # Patch coordinate (u, v) -> reference pixel via T -> source pixel via
        # H^-1, so `Mw` maps destination to source and is handed to
        # warpPerspective with WARP_INVERSE_MAP.
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            rejected["edge"] += 1
            continue
        big = patch + 2 * search
        T = np.array([[1, 0, rx - (half + search)],
                      [0, 1, ry - (half + search)],
                      [0, 0, 1]], np.float64)
        Mw = Hinv @ T
        src_patch = cv2.warpPerspective(
            src_f, Mw, (big, big),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tmpl = src_patch[search:search + patch, search:search + patch]
        win = cv2.getRectSubPix(ref_f, (big, big), (float(rx), float(ry)))

        if float(tmpl.std()) < min_std or float(win.std()) < min_std:
            rejected["flat"] += 1
            continue
        c = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        if c.shape[0] < 3 or c.shape[1] < 3:
            rejected["edge"] += 1
            continue
        _, peak, _, loc = cv2.minMaxLoc(c)
        if peak < min_ncc:
            rejected["low_ncc"] += 1
            continue
        px, py = loc
        if not (1 <= px < c.shape[1] - 1 and 1 <= py < c.shape[0] - 1):
            rejected["edge"] += 1
            continue
        dx, dy, _ = _parabola_peak(c[py - 1:py + 2, px - 1:px + 2])
        shift_x = (px + dx) - search
        shift_y = (py + dy) - search
        mag = float(np.hypot(shift_x, shift_y))
        if mag > max_shift_px:
            rejected["big_shift"] += 1
            continue
        out_ref[i] = (rx + shift_x, ry + shift_y)
        moved[i] = mag
        accepted[i] = True

    return RefineResult(
        out_src, out_ref, moved, accepted,
        {"patch_px": patch, "search_px": search, "min_ncc": min_ncc,
         "min_std": min_std, "max_shift_px": max_shift_px,
         "n_input": n, "n_refined": int(accepted.sum()),
         "median_shift_px": round(float(np.median(moved[accepted])), 4)
         if accepted.any() else None,
         "mean_shift_px": round(float(moved[accepted].mean()), 4)
         if accepted.any() else None,
         "rejected": rejected,
         "note": ("refinement is rejected where a patch is too flat to hold a "
                  "correlation peak, which is the common case in shadow")})
