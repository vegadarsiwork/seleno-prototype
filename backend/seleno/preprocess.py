"""Stage 1 - preprocessing.

Only operations that are actually implemented are exposed here. Each returns a
record of what it did so the UI can report the truth rather than a claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    image: np.ndarray
    scale: float = 1.0          # factor applied to pixel coordinates (out = scale * in)
    steps: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(img.min()), float(img.max() or 1)
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)
    return img


def clahe(img: np.ndarray, clip: float = 2.5, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation.

    Lunar strips span deep shadow and bright regolith in one frame; a global
    stretch leaves the shadowed part featureless. CLAHE equalises locally.
    """
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


def flatten_illumination(img: np.ndarray, sigma: float = 51.0) -> np.ndarray:
    """Remove the low-frequency illumination gradient (homomorphic / retinex-lite).

    Divides by a heavily blurred copy of itself, which suppresses the large-scale
    brightness ramp caused by changing solar incidence across the strip while
    keeping crater-scale texture. This is *illumination flattening*, NOT a
    physically-based photometric (Hapke/Lommel-Seeliger) correction.
    """
    f = img.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    out = f / np.maximum(bg, 1e-3)
    lo, hi = np.percentile(out, (0.5, 99.5))
    out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def denoise(img: np.ndarray) -> np.ndarray:
    """Mild edge-preserving denoise. OHRC uses TDI; residual line noise is common."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=5)


def resample_to_gsd(img: np.ndarray, src_gsd: float, target_gsd: float) -> tuple[np.ndarray, float]:
    """Resolution harmonisation.

    Brings a finer image down to a coarser ground sampling distance so that both
    images of a pair sit at a comparable scale. Anti-aliases before decimating,
    which matters: naive `cv2.resize` on a 20x reduction aliases crater rims into
    noise and destroys the very features we want to match.
    """
    if target_gsd <= src_gsd * 1.05:
        return img, 1.0
    factor = src_gsd / target_gsd            # < 1 => shrink
    sigma = 0.5 / factor                      # anti-alias kernel in source pixels
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    out = cv2.resize(blurred, (max(int(round(w * factor)), 8), max(int(round(h * factor)), 8)),
                     interpolation=cv2.INTER_AREA)
    return out, factor


def preprocess(
    img: np.ndarray,
    *,
    enabled: bool = True,
    use_clahe: bool = True,
    use_flatten: bool = True,
    use_denoise: bool = False,
    src_gsd: float | None = None,
    target_gsd: float | None = None,
) -> PreprocessResult:
    steps: list[str] = []
    out = to_gray_u8(img)
    steps.append("grayscale/uint8 normalisation")
    scale = 1.0

    if src_gsd and target_gsd:
        out, scale = resample_to_gsd(out, src_gsd, target_gsd)
        if scale != 1.0:
            steps.append(f"anti-aliased resample {src_gsd:.3f} -> {target_gsd:.3f} m/px (x{scale:.4f})")

    if enabled:
        if use_denoise:
            out = denoise(out)
            steps.append("bilateral denoise")
        if use_flatten:
            out = flatten_illumination(out)
            steps.append("illumination flattening (divide-by-blur)")
        if use_clahe:
            out = clahe(out)
            steps.append("CLAHE (clip 2.5, 8x8 tiles)")
    else:
        steps.append("enhancement DISABLED")

    return PreprocessResult(
        image=out,
        scale=scale,
        steps=steps,
        stats={"mean": round(float(out.mean()), 2),
               "std": round(float(out.std()), 2),
               "shape": list(out.shape)},
    )
