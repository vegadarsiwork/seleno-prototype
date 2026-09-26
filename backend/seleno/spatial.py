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
    coverage_after: float = 0.0           # the set actually used downstream
    coverage_all_inliers: float = 0.0     # every verified inlier, for reference
    coverage_whole_window: float = 0.0    # denominator = every cell, matchable or not
    n_eligible_cells: int = 0
    n_cells: int = 0
    dispersion_before: float = 0.0
    dispersion_after: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def cell_index(pts: np.ndarray, shape: tuple[int, int], grid: tuple[int, int]):
    gy, gx = grid
    h, w = shape[:2]
    cx = np.clip((pts[:, 0] / max(w, 1) * gx).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / max(h, 1) * gy).astype(int), 0, gy - 1)
    return cx, cy


def eligible_cells(usable_mask, shape: tuple[int, int], grid: tuple[int, int],
                   min_usable: float = 0.05) -> set:
    """Grid cells holding enough matchable pixels to be worth covering.

    This is what makes coverage a fair metric on this dataset. Two thirds of a
    typical OHRC window is shadow, so a perfectly good registration whose
    matches necessarily sit in the lit minority scores badly against a
    whole-window denominator and gets refused for the wrong reason. Cells with
    essentially nothing matchable in them are dropped from the denominator.
    """
    gy, gx = grid
    if usable_mask is None:
        return {(i, j) for i in range(gx) for j in range(gy)}
    h, w = usable_mask.shape[:2]
    out = set()
    for j in range(gy):
        y0 = j * h // gy
        y1 = max((j + 1) * h // gy, y0 + 1)
        for i in range(gx):
            x0 = i * w // gx
            x1 = max((i + 1) * w // gx, x0 + 1)
            block = usable_mask[y0:y1, x0:x1]
            if block.size and float(block.mean()) >= min_usable:
                out.add((i, j))
    return out


def cell_coverage(pts: np.ndarray, shape: tuple[int, int], grid: tuple[int, int],
                  eligible: set | None = None) -> float:
    """Fraction of grid cells holding at least one point.

    With `eligible` given, only those cells count towards the denominator.
    """
    gy, gx = grid
    denom = len(eligible) if eligible is not None else gx * gy
    if len(pts) == 0 or denom == 0:
        return 0.0
    cx, cy = cell_index(pts, shape, grid)
    occupied = set(zip(cx.tolist(), cy.tolist()))
    if eligible is not None:
        occupied &= eligible
    return float(len(occupied)) / denom


def extrapolation_fraction(pts: np.ndarray, shape: tuple[int, int],
                           grid: tuple[int, int], eligible: set | None = None) -> float:
    """Fraction of eligible grid cells lying OUTSIDE the tie-point convex hull.

    Coverage counts cells that hold a match; this counts the cells the fitted
    transform is *extrapolated* over. The two come apart exactly where it
    matters. Tie points crowded into one lit strip can score respectable
    coverage against a small eligible set while leaving most of the frame
    unconstrained, and extrapolation beyond the hull is where a fitted
    transform's error grows fastest - an affine fit that is sub-pixel among its
    own points can be many pixels out a frame-width away. Reporting it is what
    "uniform distribution" has to mean operationally.
    """
    gy, gx = grid
    cells = sorted(eligible) if eligible is not None else [
        (i, j) for i in range(gx) for j in range(gy)]
    if not cells:
        return 0.0
    hull = _convex_hull(np.asarray(pts, float).reshape(-1, 2))
    if len(hull) < 3:
        return 1.0
    h, w = shape[:2]
    idx = np.asarray(cells, float)
    cx = (idx[:, 0] + 0.5) * w / gx
    cy = (idx[:, 1] + 0.5) * h / gy
    # hull is counter-clockwise, so a point is inside when it is left of every
    # edge; one cross product per edge over all cells at once
    inside = np.ones(len(cells), bool)
    for k in range(len(hull)):
        x0, y0 = hull[k]
        x1, y1 = hull[(k + 1) % len(hull)]
        inside &= ((x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)) >= -1e-9
    return float((~inside).sum()) / len(cells)


def valid_pixel_support(points, valid_mask, max_samples=100_000, scale=1.0, unit="mask px"):
    """Estimate hull support on actual valid pixel centres, not cell centres.

    Equally spaced ranks in the valid-pixel list give every valid pixel equal
    representation even for a one-pixel-wide strip. Scan in bounded blocks to
    avoid allocating coordinates for an entire full-resolution image.
    This diagnostic does not replace the established acceptance gates.

    Inside the hull is not the same as near a measurement, so the distance
    from each sampled pixel to its closest fit point is reported too; its
    maximum is the radius of the largest unsupported gap. `scale` converts
    mask pixels to the unit quoted (native reference pixels for a decimated
    working grid).
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    mask = np.asarray(valid_mask, bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be two dimensional")
    count = int(mask.sum())
    report = {"valid_pixels": count, "sampled_pixels": min(count, max_samples),
              "valid_fraction_of_frame": float(count / mask.size) if mask.size else 0.,
              "supported_fraction": None, "extrapolation_fraction": None,
              "distance_to_fit_point": None,
              "basis": "valid overlap pixel centres inside the fit-point convex hull",
              "sampling": "all valid pixels" if count <= max_samples else "uniform ranks among valid pixels"}
    if not count:
        return report
    ranks = np.linspace(0, count - 1, min(count, max_samples), dtype=np.int64)
    samples = []
    flat = mask.ravel()
    seen = 0
    for start in range(0, flat.size, 1_000_000):
        indices = np.flatnonzero(flat[start:start + 1_000_000])
        lo, hi = np.searchsorted(ranks, [seen, seen + len(indices)])
        if hi > lo:
            samples.append(start + indices[ranks[lo:hi] - seen])
        seen += len(indices)
    ids = np.concatenate(samples)
    y, x = np.divmod(ids, mask.shape[1])
    pts = np.asarray(points, float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    hull = _convex_hull(pts)
    inside = np.zeros(len(ids), bool)
    if len(hull) >= 3:
        inside[:] = True
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            inside &= ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) >= -1e-9
    report.update(supported_fraction=float(inside.mean()),
                  extrapolation_fraction=float((~inside).mean()))
    if len(pts):
        from scipy.spatial import cKDTree
        distance = cKDTree(pts).query(np.column_stack([x, y]))[0] * float(scale)
        report["distance_to_fit_point"] = {
            "median": float(np.median(distance)), "p95": float(np.percentile(distance, 95)),
            "max": float(distance.max()), "scale": float(scale),
            "unit": unit}
    return report


def _convex_hull(p: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull, monotone chain. Kept local to avoid
    pulling OpenCV or SciPy into this module for twenty lines of geometry."""
    if len(p) < 3:
        return p
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1], float)


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
    mode: str | None = None,
    grid: tuple[int, int] = (6, 6),
    per_cell: int = 3,
    min_separation: float | None = None,
    min_keep: int = 8,
    min_for_thinning: int = 60,
    usable_mask=None,
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
    # mode: "grid" (spatial quota) | "topk" (confidence, same budget) | "all"
    if mode is None:
        mode = "grid" if enabled else "all"
    # Thinning a set that is already sparse costs more than the conditioning it
    # buys: measured on real OHRC repeat-pass windows, the grid quota made
    # held-out error worse rather than better (1.71 px against 1.41 px when it
    # cut ~1300 inliers to ~44). Below `min_for_thinning` the quota is skipped
    # and every inlier is kept.
    thinning_skipped = False
    if mode == "grid" and n < min_for_thinning:
        mode = "all"
        thinning_skipped = True
    elig = eligible_cells(usable_mask, shape, grid)
    cov_all = cell_coverage(pts, shape, grid, elig)

    if n == 0:
        z = np.zeros(0, bool)
        return SpatialResult(
            keep=z, keep_topk=z, grid=grid,
            coverage_before=0.0, coverage_after=0.0, coverage_all_inliers=cov_all,
            coverage_whole_window=0.0,
            n_eligible_cells=len(elig), n_cells=grid[0] * grid[1],
            dispersion_before=0.0, dispersion_after=0.0,
            detail={"status": "no matches", "n_in": 0, "n_out": 0, "mode": mode})

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
    # `coverage_after` always describes the set the pipeline goes on to use.
    if mode == "grid":
        result_keep = keep
    elif mode == "topk":
        result_keep = topk
    else:
        result_keep = np.ones(n, bool)          # "all": every verified inlier
    used = pts[result_keep]
    status = {"grid": "grid quota applied",
              "topk": "top-K by confidence",
              "all": "all verified inliers retained"}[mode]
    return SpatialResult(
        keep=result_keep, keep_topk=topk, grid=grid,
        coverage_before=cell_coverage(top, shape, grid, elig),
        coverage_after=cell_coverage(used, shape, grid, elig),
        coverage_all_inliers=cov_all,
        coverage_whole_window=cell_coverage(used, shape, grid, None),
        n_eligible_cells=len(elig), n_cells=gx * gy,
        dispersion_before=dispersion(top, shape),
        dispersion_after=dispersion(used, shape),
        detail={
            "status": status + (" (grid quota skipped: only %d inliers, below the "
                                "%d needed to make thinning worthwhile)"
                                % (n, min_for_thinning) if thinning_skipped else ""),
            "mode": mode,
            "thinning_skipped": thinning_skipped,
            "n_in": n,
            "n_out": int(result_keep.sum()),
            "budget": budget,
            "cells_occupied": len(quota),
            "cells_eligible": len(elig),
            "cells_total": gx * gy,
            "coverage_denominator": ("matchable cells only" if usable_mask is not None
                                     else "every cell"),
            "per_cell_quota": per_cell,
            "min_separation_px": round(float(min_separation), 2),
            "relaxed_to_min_keep": relaxed,
        },
    )
