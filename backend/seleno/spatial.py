"""Stage 4 - spatially distributed match selection.

Why this stage exists
---------------------
Ranking correspondences purely by descriptor confidence concentrates them on the
few high-contrast structures in a scene - typically one or two crater rims. A
transform fitted from a tight cluster is well constrained locally and badly
extrapolated everywhere else: the residual at the cluster is small while the
corners of the image drift. Forcing the retained set to span the frame trades a
little per-match confidence for a much better conditioned estimate.

The selector is a greedy grid-quota + minimum-separation filter operating on
geometrically verified matches only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpatialResult:
    keep: np.ndarray                      # (N,) bool - the set the pipeline goes on to use
    keep_topk: np.ndarray = None          # (N,) bool - same budget, ranked by confidence only
    grid: tuple[int, int] = (6, 6)
    coverage_before: float = 0.0          # confidence-ranked top-K (the fair comparison)
    coverage_after: float = 0.0           # spatially selected
    coverage_all_inliers: float = 0.0     # every verified inlier, for reference
    dispersion_before: float = 0.0
    dispersion_after: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def cell_coverage(pts: np.ndarray, shape: tuple[int, int], grid: tuple[int, int]) -> float:
    """Fraction of grid cells that contain at least one point."""
    gy, gx = grid
    if len(pts) == 0:
        return 0.0
    h, w = shape[:2]
    cx = np.clip((pts[:, 0] / max(w, 1) * gx).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / max(h, 1) * gy).astype(int), 0, gy - 1)
    return float(len(set(zip(cx.tolist(), cy.tolist())))) / (gx * gy)


def dispersion(pts: np.ndarray, shape: tuple[int, int]) -> float:
    """Normalised spread: mean distance from the centroid over the image half-diagonal.

    Complements cell coverage - coverage saturates once every cell is touched,
    dispersion keeps responding to how far apart the points actually are.
    """
    if len(pts) < 2:
        return 0.0
    h, w = shape[:2]
    half_diag = 0.5 * float(np.hypot(h, w))
    c = pts.mean(axis=0)
    return float(np.linalg.norm(pts - c, axis=1).mean() / max(half_diag, 1e-6))


def select(
    pts: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
    *,
    enabled: bool = True,
    grid: tuple[int, int] = (6, 6),
    per_cell: int = 3,
    min_separation: float | None = None,
    min_keep: int = 8,
) -> SpatialResult:
    """Select a spatially distributed, high-confidence subset of `pts`.

    Parameters
    ----------
    pts      : (N,2) coordinates in the *source* image frame.
    scores   : (N,) confidence, higher is better.
    shape    : source image (h, w) - defines the grid and the separation scale.
    per_cell : maximum matches retained per grid cell.
    min_separation : minimum pixel distance between retained matches. Defaults to
        a quarter of the mean cell size, which keeps points from bunching at a
        cell corner and defeating the quota.
    """
    h, w = shape[:2]
    n = len(pts)
    cov_all = cell_coverage(pts, shape, grid)

    if n == 0:
        z = np.zeros(0, bool)
        return SpatialResult(z, z, grid, 0.0, 0.0, cov_all, 0.0, 0.0,
                             {"status": "no matches", "n_in": 0, "n_out": 0})

    gy, gx = grid
    if min_separation is None:
        min_separation = 0.25 * (w / gx + h / gy) / 2.0

    cx = np.clip((pts[:, 0] / max(w, 1) * gx).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / max(h, 1) * gy).astype(int), 0, gy - 1)

    order = np.argsort(-scores)           # best first
    quota: dict[tuple[int, int], int] = {}
    kept_idx: list[int] = []
    kept_pts: list[np.ndarray] = []
    sep2 = float(min_separation) ** 2

    for i in order:
        key = (int(cx[i]), int(cy[i]))
        if quota.get(key, 0) >= per_cell:
            continue
        p = pts[i]
        if kept_pts:
            arr = np.asarray(kept_pts)
            if np.min(((arr - p) ** 2).sum(axis=1)) < sep2:
                continue
        quota[key] = quota.get(key, 0) + 1
        kept_idx.append(int(i))
        kept_pts.append(p)

    # A degenerate scene can starve the quota; fall back to the top-scoring
    # matches so the stage never leaves the estimator with too few points.
    relaxed = False
    if len(kept_idx) < min_keep:
        relaxed = True
        extra = [int(i) for i in order if int(i) not in set(kept_idx)]
        kept_idx.extend(extra[: max(0, min_keep - len(kept_idx))])

    keep = np.zeros(n, bool)
    keep[np.asarray(kept_idx, dtype=int)] = True

    # The fair baseline: the SAME number of matches, chosen by confidence alone.
    # Comparing against "all inliers" would be meaningless, because every
    # selection strategy loses coverage when it drops points. Holding the budget
    # fixed isolates the effect of the spatial constraint itself.
    budget = int(keep.sum())
    topk = np.zeros(n, bool)
    topk[order[:budget]] = True

    top = pts[topk]
    # `coverage_after` must describe the set the pipeline actually goes on to
    # use. With the stage disabled that is the confidence-ranked set, so the two
    # figures coincide - reporting the spatial set's coverage there would
    # describe points no downstream stage ever sees.
    result_keep = keep if enabled else topk
    used = pts[result_keep]
    return SpatialResult(
        result_keep, topk, grid,
        cell_coverage(top, shape, grid), cell_coverage(used, shape, grid), cov_all,
        dispersion(top, shape), dispersion(used, shape),
        {
            "status": "applied" if enabled else "disabled (top-K by confidence used instead)",
            "n_in": n,
            "n_out": int(result_keep.sum()),
            "budget": budget,
            "cells_occupied": len(quota),
            "cells_total": gx * gy,
            "per_cell_quota": per_cell,
            "min_separation_px": round(float(min_separation), 2),
            "relaxed_to_min_keep": relaxed,
        },
    )
