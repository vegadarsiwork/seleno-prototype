"""Stage 5 - transformation refit, warping and visual products."""
from __future__ import annotations

import cv2
import numpy as np

from . import verify as V


def refit(p_src: np.ndarray, p_ref: np.ndarray, model_type: str) -> np.ndarray | None:
    """Least-squares refit on the finally selected correspondences.

    RANSAC returns the model of the best consensus set; refitting on the retained
    (now spatially distributed) points uses all of their information rather than
    the minimal sample that seeded the hypothesis.
    """
    need = {"similarity": 2, "affine": 3, "homography": 4}[model_type]
    if len(p_src) < need:
        return None
    src = p_src.astype(np.float64)
    ref = p_ref.astype(np.float64)
    if model_type == "homography":
        M, _ = cv2.findHomography(src, ref, 0)          # 0 = plain least squares
    elif model_type == "affine":
        M, _ = cv2.estimateAffine2D(src, ref, method=cv2.LMEDS)
    else:
        M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
    if M is None:
        return None
    return V._to3x3(np.asarray(M))


def warp_to_reference(img_src: np.ndarray, H: np.ndarray, ref_shape: tuple[int, int]):
    """Warp the source into the reference frame; also return a validity mask."""
    h, w = ref_shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(img_src.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask > 127


def match_intensity(warped: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares gain/bias fit of `warped` onto `ref` inside the overlap.

    DISPLAY ONLY. The two products differ radiometrically (different processing
    level, different illumination), and that difference dominates an overlay or a
    difference image, hiding the geometry the reader is trying to judge. Removing
    a single global gain and bias leaves geometric residual visible.

    This never touches the correspondences, the transform, or any metric. NCC is
    already invariant to an affine intensity change, so it is unaffected either
    way.
    """
    if mask.sum() < 64:
        return warped
    x = warped[mask].astype(np.float64)
    y = ref[mask].astype(np.float64)
    vx = x.var()
    if vx < 1e-9:
        return warped
    a = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    b = float(y.mean() - a * x.mean())
    out = np.clip(warped.astype(np.float64) * a + b, 0, 255).astype(np.uint8)
    return np.where(mask, out, 0).astype(np.uint8)


def anaglyph(ref: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Two-colour overlay: reference in green, registered source in magenta.

    Where the two agree the channels sum to grey; residual misalignment shows as
    coloured fringing, which is far easier to judge by eye than a 50/50 blend.
    """
    out = np.zeros(ref.shape[:2] + (3,), np.uint8)
    out[..., 1] = ref                                    # G  = reference
    out[..., 0] = np.where(mask, warped, 0)              # B  } magenta = warped
    out[..., 2] = np.where(mask, warped, 0)              # R  }
    return out


def checkerboard(ref: np.ndarray, warped: np.ndarray, mask: np.ndarray, tiles: int = 8) -> np.ndarray:
    """Alternating tiles from each image - structures must run straight across seams."""
    h, w = ref.shape[:2]
    ty, tx = max(h // tiles, 1), max(w // tiles, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    sel = (((yy // ty) + (xx // tx)) % 2) == 0
    out = np.where(sel, ref, np.where(mask, warped, ref)).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def difference(ref: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Absolute difference inside the overlap, contrast-stretched, false-coloured."""
    d = cv2.absdiff(ref, warped)
    d = np.where(mask, d, 0).astype(np.uint8)
    if mask.any():
        hi = float(np.percentile(d[mask], 99)) or 1.0
        d = np.clip(d.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    out[~mask] = (18, 18, 20)
    return out


def overlap_ncc(ref: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> float:
    """Normalised cross-correlation inside the overlap.

    A photometric check that does not use the correspondences at all, so it is an
    independent witness that alignment improved (it is still only a similarity
    measure, not a geodetic accuracy figure).
    """
    if mask.sum() < 64:
        return float("nan")
    a = ref[mask].astype(np.float64)
    b = warped[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / den) if den > 1e-9 else float("nan")


def register(img_src, img_ref, H, model_type="homography"):
    """Produce every visual product for one estimated transform."""
    ref_shape = img_ref.shape[:2]
    warped, mask = warp_to_reference(img_src, H, ref_shape)
    shown = match_intensity(warped, img_ref, mask)   # display only, see above
    return {
        "warped": warped,
        "mask": mask,
        "anaglyph": anaglyph(img_ref, shown, mask),
        "checker": checkerboard(img_ref, shown, mask),
        "diff": difference(img_ref, shown, mask),
        "overlap_fraction": float(mask.mean()),
        "ncc_after": overlap_ncc(img_ref, warped, mask),
        "model_type": model_type,
    }


def identity_baseline_ncc(img_src: np.ndarray, img_ref: np.ndarray) -> float:
    """NCC with no transform applied - the 'before registration' photometric score."""
    h = min(img_src.shape[0], img_ref.shape[0])
    w = min(img_src.shape[1], img_ref.shape[1])
    m = np.ones((h, w), bool)
    return overlap_ncc(img_ref[:h, :w], img_src[:h, :w], m)
