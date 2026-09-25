"""Stage 3 - geometric verification.

Candidate correspondences from descriptor matching are individually plausible but
collectively inconsistent: repeated crater rims and regolith texture produce many
matches that no single geometric transform can explain. A robust estimator finds
the largest self-consistent subset and labels the rest as outliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Which transform families we can estimate. See ASSUMPTIONS.md for when each is
# defensible on lunar imagery.
MODELS = {
    "similarity": "Similarity (4 dof: rotation, uniform scale, translation)",
    "affine": "Affine (6 dof)",
    "homography": "Homography (8 dof, planar-scene approximation)",
}


@dataclass
class VerifyResult:
    model: np.ndarray | None              # 3x3 matrix mapping source -> reference
    inlier_mask: np.ndarray               # (N,) bool
    n_candidates: int
    n_inliers: int
    residuals: np.ndarray                 # (N,) symmetric transfer error in reference px
    model_type: str = "homography"
    ok: bool = False
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def n_outliers(self) -> int:
        return int(self.n_candidates - self.n_inliers)

    @property
    def inlier_ratio(self) -> float:
        return float(self.n_inliers) / self.n_candidates if self.n_candidates else 0.0


def _to3x3(m: np.ndarray) -> np.ndarray:
    if m.shape == (2, 3):
        out = np.eye(3, dtype=np.float64)
        out[:2, :] = m
        return out
    return m.astype(np.float64)


def transfer_error(H: np.ndarray, p_src: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Symmetric transfer error: mean of forward and backward reprojection distance."""
    if len(p_src) == 0:
        return np.zeros((0,), np.float64)
    src = np.hstack([p_src, np.ones((len(p_src), 1))]).T
    fwd = H @ src
    fwd = (fwd[:2] / np.where(np.abs(fwd[2]) < 1e-12, 1e-12, fwd[2])).T
    e_f = np.linalg.norm(fwd - p_ref, axis=1)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return e_f
    ref = np.hstack([p_ref, np.ones((len(p_ref), 1))]).T
    bwd = Hi @ ref
    bwd = (bwd[:2] / np.where(np.abs(bwd[2]) < 1e-12, 1e-12, bwd[2])).T
    e_b = np.linalg.norm(bwd - p_src, axis=1)
    return 0.5 * (e_f + e_b)


def _min_points(model_type: str) -> int:
    return {"similarity": 2, "affine": 3, "homography": 4}[model_type]


def log10_nfa(n: int, k: int, model_type: str, threshold: float, area: float) -> float:
    """Number of false alarms of a consensus set, a-contrario (Moisan & Stival 2004).

    Under the null hypothesis the n candidates are unrelated: each reference
    point lands anywhere in `area` pixels, so a point agrees with a model to
    within `threshold` with probability alpha = pi * threshold^2 / area. The
    expected number of k-point consensus sets as good as this one, over every
    model the minimal samples can build, is

        NFA = (n - s) * C(n, k) * C(k, s) * alpha^(k - s)

    and a set is meaningful when NFA < 1. This replaces a fixed inlier count:
    7 of 45 random matches agreeing with a homography is noise, 7 of 7 local
    NCC measurements agreeing with a similarity is not. `area` is where a wrong
    match can land - the overlap for a global matcher, the search window for a
    local one.
    """
    from math import lgamma, log, log10, pi
    s = _min_points(model_type)
    if n <= s or k <= s:
        return float("inf")
    alpha = min(1.0, pi * float(threshold) ** 2 / max(float(area), 1.0))
    if alpha >= 1.0:
        return float("inf")

    def log10_binom(a, b):
        return (lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)) / log(10)
    return (log10(n - s) + log10_binom(n, k) + log10_binom(k, s)
            + (k - s) * log10(alpha))


def degenerate(H: np.ndarray | None, p_src: np.ndarray, *, max_anisotropy: float = 10.0,
               scale_range: tuple[float, float] = (0.05, 20.0), extent=None,
               max_area_ratio: float = 4.0) -> str | None:
    """Why a fitted model cannot describe two images of the same ground, or None.

    Checked where the model is actually used - at the Jacobian over the inlier
    hull and, when `extent` (x0, y0, x1, y1) is given, over the whole source
    region it will be applied to - so a homography that is benign at the
    origin but folds the image, or changes its scale several-fold across the
    frame, between or beyond the tie points is caught as well. A near-nadir
    image of the Moon has close to one scale everywhere; `max_area_ratio`
    bounds how much the local area scale may vary over the frame.
    """
    if H is None or not np.isfinite(H).all():
        return "no finite model"
    pts = np.asarray(p_src, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return "too few points to judge the model"
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    probes = [[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]],
              [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]]
    if extent is not None:
        x0, y0, x1, y1 = map(float, extent)
        probes += [[x0, y0], [x1, y0], [x0, y1], [x1, y1], [(x0 + x1) / 2, (y0 + y1) / 2]]
    probes = np.asarray(probes, np.float64)
    H = np.asarray(H, np.float64)
    dets = []
    for x, y in probes:
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if abs(w) < 1e-12:
            return "the model maps part of the tie-point hull to infinity"
        u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
        v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
        J = np.array([[H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]],
                      [H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]]]) / w
        if np.linalg.det(J) <= 0:
            return "the model folds the image (non-positive Jacobian)"
        dets.append(float(np.linalg.det(J)))
        sv = np.linalg.svd(J, compute_uv=False)
        if sv.min() < scale_range[0] or sv.max() > scale_range[1]:
            return "local scale %.3g-%.3g outside %.3g-%.3g" % (sv.min(), sv.max(), *scale_range)
        if sv.max() / sv.min() > max_anisotropy:
            return "local anisotropy %.1f exceeds %.1f" % (sv.max() / sv.min(), max_anisotropy)
    if max(dets) / min(dets) > max_area_ratio:
        return ("local area scale varies %.1fx across the frame (limit %.1fx)"
                % (max(dets) / min(dets), max_area_ratio))
    return None


def verify(
    p_src: np.ndarray,
    p_ref: np.ndarray,
    *,
    model_type: str = "homography",
    threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 10000,
    enabled: bool = True,
) -> VerifyResult:
    n = len(p_src)
    if not enabled:
        # Verification switched off in experiment mode: everything is an "inlier".
        return VerifyResult(None, np.ones(n, bool), n, n, np.zeros(n),
                            model_type, ok=False,
                            reason="geometric verification disabled",
                            detail={"estimator": "none"})

    need = _min_points(model_type)
    if n < need:
        return VerifyResult(None, np.zeros(n, bool), n, 0, np.zeros(n), model_type,
                            ok=False,
                            reason="only %d candidates, need >= %d for %s" % (n, need, model_type))

    src = p_src.astype(np.float64)
    ref = p_ref.astype(np.float64)
    est = "USAC_MAGSAC"
    if model_type == "homography":
        M, mask = cv2.findHomography(src, ref, cv2.USAC_MAGSAC,
                                     ransacReprojThreshold=threshold,
                                     confidence=confidence, maxIters=max_iters)
    elif model_type == "affine":
        M, mask = cv2.estimateAffine2D(src, ref, method=cv2.RANSAC,
                                       ransacReprojThreshold=threshold,
                                       confidence=confidence, maxIters=max_iters)
        est = "RANSAC (affine2D)"
    else:
        M, mask = cv2.estimateAffinePartial2D(src, ref, method=cv2.RANSAC,
                                              ransacReprojThreshold=threshold,
                                              confidence=confidence, maxIters=max_iters)
        est = "RANSAC (partialAffine2D)"

    if M is None or mask is None:
        return VerifyResult(None, np.zeros(n, bool), n, 0, np.zeros(n), model_type,
                            ok=False, reason="robust estimator returned no model",
                            detail={"estimator": est})

    H = _to3x3(np.asarray(M))
    mask = mask.ravel().astype(bool)
    res = transfer_error(H, src, ref)
    detail = {"estimator": est, "threshold_px": threshold, "confidence": confidence}
    detail.update(decompose(H))
    return VerifyResult(H, mask, n, int(mask.sum()), res, model_type, ok=True,
                        reason="", detail=detail)


def decompose(H: np.ndarray) -> dict[str, float]:
    """Report interpretable transform parameters so the estimate can be sanity-checked.

    A physically implausible fit (e.g. 40 percent scale change between two nadir
    passes at similar altitude) is a red flag even when the inlier ratio looks fine.
    """
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    sx = float(np.hypot(a, c))
    sy = float(np.hypot(b, d))
    rot = float(np.degrees(np.arctan2(c, a)))
    det = float(a * d - b * c)
    return {
        "est_scale_x": round(sx, 4),
        "est_scale_y": round(sy, 4),
        "est_rotation_deg": round(rot, 3),
        "est_shear_det": round(det, 4),
        "est_translation_px": [round(float(H[0, 2]), 2), round(float(H[1, 2]), 2)],
        "est_perspective": [float(H[2, 0]), float(H[2, 1])],
    }


def plausibility(H: np.ndarray | None, model_type: str, expected_scale: float = 1.0) -> list[str]:
    """Cheap physical sanity checks; returns a list of human-readable warnings.

    `expected_scale` is the scale the two products' ground sampling distances
    already imply (source GSD / reference GSD). Judging the fitted scale against
    that, rather than against 1.0, is what makes the check meaningful on a
    cross-resolution pair.
    """
    if H is None:
        return ["no model estimated"]
    d = decompose(H)
    warn = []
    e = float(expected_scale) if expected_scale else 1.0
    rx, ry = d["est_scale_x"] / e, d["est_scale_y"] / e
    if not (0.5 <= rx <= 2.0 and 0.5 <= ry <= 2.0):
        warn.append("fitted scale (%.3f, %.3f) departs from the %.3f implied by the "
                    "products' ground sampling distances" % (d["est_scale_x"], d["est_scale_y"], e))
    if abs(d["est_scale_x"] - d["est_scale_y"]) > 0.25 * max(d["est_scale_x"], 1e-6):
        warn.append("strongly anisotropic scale - unusual for two near-nadir passes")
    if d["est_shear_det"] <= 0:
        warn.append("model flips handedness (negative determinant)")
    if model_type == "homography" and max(abs(v) for v in d["est_perspective"]) > 1e-3:
        warn.append("large perspective terms - homography may be over-fitting relief")
    return warn
