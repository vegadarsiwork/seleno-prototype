"""A smooth local correction on top of the global model.

One affine (or homography) cannot describe a pushbroom strip against an
orthorectified map: attitude jitter varies along the track and terrain relief
displaces ground across it. On the real pairs the held-out residual after the
global fit was measured to be mostly *systematic* - predictable from nearby
points - and only about one reference pixel random (see
reports/validation_fixes_20260923/diagnostics). A global model cannot remove
that part; a smooth field can.

The field corrects the inverse mapping on the reference side:

    source(q) = base_inverse(q) + E(q)

where `base_inverse` is the global (or segment-blended) model and E is a
uniform cubic B-spline over the working reference grid. Its smoothness is not a
free choice: the control spacing and the bending-energy weight are picked by
spatial cross-validation over the FIT cells only (whole cells are held out in
turn), so neither the validation nor the sealed test fold influences it. The
null model - no field - is always a candidate, and wins unless the field
predicts unseen cells better.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _bspline_weights(t):
    """Uniform cubic B-spline basis at fractional offsets t in [0, 1)."""
    t = np.asarray(t, np.float64)
    t2, t3 = t * t, t * t * t
    return np.stack([(1 - t) ** 3 / 6.0,
                     (3 * t3 - 6 * t2 + 4) / 6.0,
                     (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0,
                     t3 / 6.0], axis=-1)


def _layout(shape, spacing):
    """Control lattice covering `shape` (h, w) with one knot of margin."""
    h, w = shape
    ny = int(math.ceil(h / spacing)) + 3
    nx = int(math.ceil(w / spacing)) + 3
    return ny, nx


def _design(points, spacing, ny, nx):
    """Sparse matrix mapping control coefficients to field values at points."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    u = p[:, 0] / spacing + 1.0          # one knot of margin on each side
    v = p[:, 1] / spacing + 1.0
    iu = np.floor(u).astype(np.int64)
    iv = np.floor(v).astype(np.int64)
    wu = _bspline_weights(u - iu)
    wv = _bspline_weights(v - iv)
    rows, cols, vals = [], [], []
    n = len(p)
    for a in range(4):
        for b in range(4):
            cy = np.clip(iv - 1 + a, 0, ny - 1)
            cx = np.clip(iu - 1 + b, 0, nx - 1)
            rows.append(np.arange(n))
            cols.append(cy * nx + cx)
            vals.append(wv[:, a] * wu[:, b])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                     np.concatenate(cols))),
                             shape=(n, ny * nx))


def _bending(ny, nx):
    """Second-difference penalty on the control lattice (thin-plate energy)."""
    def d2(n):
        if n < 3:
            return sparse.csr_matrix((0, n))
        return sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
                            [0, 1, 2], shape=(n - 2, n))

    def d1(n):
        if n < 2:
            return sparse.csr_matrix((0, n))
        return sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    iy, ix = sparse.identity(ny), sparse.identity(nx)
    dxx = sparse.kron(iy, d2(nx))
    dyy = sparse.kron(d2(ny), ix)
    dxy = sparse.kron(d1(ny), d1(nx))
    return (dxx.T @ dxx + dyy.T @ dyy + 2 * dxy.T @ dxy).tocsr()


def fit(points, residuals, shape, spacing, smoothness, weights=None, iterations=5):
    """Robust (Huber IRLS) regularised fit of a field to residual vectors.

    `points` are reference working-grid positions, `residuals` the source-side
    corrections wanted there. `smoothness` is dimensionless: it is scaled by
    the data weight per control cell so one value means the same thing on a
    sparse and a dense set.
    """
    q = np.asarray(points, np.float64).reshape(-1, 2)
    r = np.asarray(residuals, np.float64).reshape(-1, 2)
    ny, nx = _layout(shape, spacing)
    A = _design(q, spacing, ny, nx)
    L = _bending(ny, nx)
    base = np.ones(len(q)) if weights is None else np.asarray(weights, np.float64)
    # Data weight per control cell, so `smoothness` is scale-free.
    density = base.sum() / max(1, ny * nx)
    lam = float(smoothness) * max(density, 1e-6)
    robust = np.ones(len(q))
    coef = np.zeros((ny * nx, 2))
    ridge = sparse.identity(ny * nx) * (1e-6 * max(density, 1e-6))
    for _ in range(max(1, iterations)):
        W = sparse.diags(base * robust)
        N = (A.T @ W @ A + lam * L + ridge).tocsc()
        coef = np.column_stack([spsolve(N, A.T @ (base * robust * r[:, k])) for k in (0, 1)])
        e = np.linalg.norm(A @ coef - r, axis=1)
        sigma = max(0.05, 1.4826 * float(np.median(np.abs(e - np.median(e)))))
        robust = np.minimum(1.0, 1.345 * sigma / np.maximum(e, 1e-12))
    return {"kind": "cubic_bspline_inverse_correction", "spacing": float(spacing),
            "smoothness": float(smoothness), "lattice": [ny, nx],
            "coefficients": coef.reshape(ny, nx, 2).tolist()}


def evaluate(field, points):
    """Field value (source-side correction) at reference working positions."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    if field is None or not len(p):
        return np.zeros_like(p)
    frame = field.get("frame")
    if frame is not None:
        # Carried onto another grid (see `conjugate`): evaluate where this
        # frame's points lie on the grid the field was fitted on, and express
        # the correction in this frame's pixels.
        F = np.asarray(frame, np.float64)
        Fi = np.linalg.inv(F)
        p = p @ Fi[:2, :2].T + Fi[:2, 2]
    ny, nx = field["lattice"]
    coef = np.asarray(field["coefficients"], np.float64).reshape(ny * nx, 2)
    out = _design(p, field["spacing"], ny, nx) @ coef
    if frame is not None:
        out = out @ F[:2, :2].T
    return out


def cross_validate(points, residuals, cells, shape, candidates, weights=None, folds=5, seed=0):
    """Pick (spacing, smoothness) by predicting whole held-out fit cells.

    Returns ``(best, table)``. `best` is None when no field beats the null
    model; `table` records every candidate's cross-validated error so the
    choice can be audited. The score is the median held-out error: a field
    that fits a few mismatches cannot win on a median.
    """
    q = np.asarray(points, np.float64).reshape(-1, 2)
    r = np.asarray(residuals, np.float64).reshape(-1, 2)
    cells = np.asarray(cells)
    unique = np.unique(cells)
    table = []
    if len(unique) < 4 or len(q) < 30:
        return None, [{"note": "too few fit cells or points for cross-validation"}]
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    k = int(min(folds, len(unique)))
    groups = [order[i::k] for i in range(k)]

    def score(spacing, smoothness):
        errs = []
        for g in groups:
            held = np.isin(cells, g)
            if held.all() or not held.any():
                continue
            if spacing is None:
                pred = np.zeros((int(held.sum()), 2))
            else:
                f = fit(q[~held], r[~held], shape, spacing, smoothness,
                        None if weights is None else np.asarray(weights)[~held], iterations=3)
                pred = evaluate(f, q[held])
            errs.append(np.linalg.norm(pred - r[held], axis=1))
        e = np.concatenate(errs) if errs else np.array([np.inf])
        return float(np.median(e)), float(np.percentile(e, 90))

    null = score(None, None)
    table.append({"spacing": None, "smoothness": None, "cv_median": null[0], "cv_p90": null[1]})
    best, best_score = None, null
    for spacing, smoothness in candidates:
        s = score(spacing, smoothness)
        table.append({"spacing": spacing, "smoothness": smoothness,
                      "cv_median": s[0], "cv_p90": s[1]})
        # Adopt only a clear improvement on held-out cells, judged on the
        # median and not paid for in the tail.
        if s[0] < 0.95 * best_score[0] and s[1] <= 1.05 * best_score[1]:
            best, best_score = (spacing, smoothness), s
    return best, table
