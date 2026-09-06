"""Rendering of pipeline products. Drawing only - no numbers are invented here."""
from __future__ import annotations

import cv2
import numpy as np

BG = (18, 18, 20)
INLIER = (120, 235, 140)      # BGR - green
OUTLIER = (70, 70, 210)       # BGR - red
SELECTED = (60, 200, 255)     # BGR - amber
GRID = (60, 60, 66)


def _canvas(a: np.ndarray, b: np.ndarray, gap: int = 16):
    """Side-by-side canvas, images top-aligned, with the x-offset of the right image."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    H, W = max(ha, hb), wa + gap + wb
    out = np.full((H, W, 3), BG, np.uint8)
    out[:ha, :wa] = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a
    out[:hb, wa + gap:wa + gap + wb] = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR) if b.ndim == 2 else b
    return out, wa + gap


def draw_matches(img_a, img_b, p_a, p_b, mask=None, *, max_lines=400, thickness=1,
                 color_in=INLIER, color_out=OUTLIER, draw_outliers=True, seed=0):
    """Correspondence lines. `mask` True = inlier (green), False = outlier (red)."""
    out, xoff = _canvas(img_a, img_b)
    n = len(p_a)
    if n == 0:
        return out
    if mask is None:
        mask = np.ones(n, bool)
    idx = np.arange(n)
    if not draw_outliers:
        idx = idx[mask]
    if len(idx) > max_lines:                       # deterministic subsample
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, max_lines, replace=False))

    # outliers first so inliers draw on top
    for good in (False, True):
        for i in idx:
            if bool(mask[i]) != good:
                continue
            c = color_in if good else color_out
            pa = (int(round(p_a[i][0])), int(round(p_a[i][1])))
            pb = (int(round(p_b[i][0])) + xoff, int(round(p_b[i][1])))
            cv2.line(out, pa, pb, c, thickness, cv2.LINE_AA)
            cv2.circle(out, pa, 2, c, -1, cv2.LINE_AA)
            cv2.circle(out, pb, 2, c, -1, cv2.LINE_AA)
    return out


def draw_points_grid(img, pts, grid=(6, 6), color=SELECTED, radius=4, label=None):
    """Points over the analysis grid - makes spatial distribution readable at a glance."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    gy, gx = grid
    for i in range(1, gx):
        cv2.line(out, (i * w // gx, 0), (i * w // gx, h), GRID, 1)
    for j in range(1, gy):
        cv2.line(out, (0, j * h // gy), (w, j * h // gy), GRID, 1)
    for p in pts:
        c = (int(round(p[0])), int(round(p[1])))
        cv2.circle(out, c, radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, c, radius + 2, (12, 12, 14), 1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 240), 1, cv2.LINE_AA)
    return out


def draw_footprint(img_ref, H, src_shape, color=(60, 200, 255)):
    """Outline of where the source lands in the reference frame."""
    out = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if img_ref.ndim == 2 else img_ref.copy()
    if H is None:
        return out
    h, w = src_shape[:2]
    c = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(c, H.astype(np.float64)).reshape(-1, 2).astype(int)
    cv2.polylines(out, [q], True, color, 2, cv2.LINE_AA)
    return out


def side_by_side(a, b):
    out, _ = _canvas(a, b)
    return out


def to_png_bytes(img, quality=None):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def fit(img, max_side=900):
    """Downscale for transport only. Never used for measurement."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img
    return cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
