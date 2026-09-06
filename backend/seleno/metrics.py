"""Stage 6 - quantitative evaluation.

Every number here is computed from the run. The distinction that matters for an
honest demo is between:

  * self-consistency metrics (reprojection / transfer RMSE) - these say the
    correspondences agree with the fitted model, and they can look excellent for
    a model that is completely wrong, and
  * ground-truth metrics - only available for pairs where the true transform is
    known by construction, and reported separately and explicitly.

`ground_truth_available` is carried through to the UI so the two are never
presented as the same thing.
"""
from __future__ import annotations

import numpy as np

from .verify import transfer_error


def rmse(residuals: np.ndarray) -> float:
    if len(residuals) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(residuals))))


def reprojection_metrics(H, p_src, p_ref) -> dict:
    """Self-consistency of the retained correspondences under the fitted model."""
    if H is None or len(p_src) == 0:
        return {"rmse_px": None, "median_px": None, "max_px": None, "p90_px": None}
    e = transfer_error(H, p_src.astype(np.float64), p_ref.astype(np.float64))
    return {
        "rmse_px": round(rmse(e), 4),
        "median_px": round(float(np.median(e)), 4),
        "p90_px": round(float(np.percentile(e, 90)), 4),
        "max_px": round(float(e.max()), 4),
    }


def corner_error(H_est, H_gt, shape) -> dict:
    """Mean corner projection error against a known ground-truth transform.

    Projects the four image corners through both transforms and measures the
    distance. This is the standard registration accuracy figure and, unlike
    reprojection RMSE, it cannot be made small by a self-consistent wrong answer.
    """
    if H_est is None or H_gt is None:
        return {"corner_error_px": None, "corner_errors_px": None}
    h, w = shape[:2]
    c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    ch = np.hstack([c, np.ones((4, 1))]).T

    def proj(M):
        q = M @ ch
        return (q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])).T

    d = np.linalg.norm(proj(H_est) - proj(H_gt), axis=1)
    return {
        "corner_error_px": round(float(d.mean()), 4),
        "corner_errors_px": [round(float(x), 3) for x in d],
    }


def ground_truth_residuals(H_est, H_gt, p_src) -> dict:
    """RMSE of the estimated mapping against truth, sampled at the retained points."""
    if H_est is None or H_gt is None or len(p_src) == 0:
        return {"gt_rmse_px": None}
    src = np.hstack([p_src.astype(np.float64), np.ones((len(p_src), 1))]).T

    def proj(M):
        q = M @ src
        return (q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])).T

    d = np.linalg.norm(proj(H_est) - proj(H_gt), axis=1)
    return {"gt_rmse_px": round(rmse(d), 4),
            "gt_median_px": round(float(np.median(d)), 4)}


def summarise(*, candidates, verified, selected, timings, coverage, extra=None) -> dict:
    out = {
        "candidate_matches": int(candidates),
        "inliers": int(verified),
        "outliers_rejected": int(candidates - verified),
        "inlier_ratio": round(float(verified) / candidates, 4) if candidates else 0.0,
        "selected_matches": int(selected),
        "runtime_s": {k: round(v, 4) for k, v in timings.items()},
        "runtime_total_s": round(sum(timings.values()), 4),
    }
    out.update(coverage)
    if extra:
        out.update(extra)
    return out
