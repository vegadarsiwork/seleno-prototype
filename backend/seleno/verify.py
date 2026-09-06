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
