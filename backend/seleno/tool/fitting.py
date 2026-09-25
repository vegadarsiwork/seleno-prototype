"""Robust fits with bounded influence from heavily textured image regions."""
import numpy as np

from .coordinates import project
from ..spatial import cell_index


def balanced_refit(source, reference, model, shape, grid=(8, 8), initial=None):
    """Use every supplied inlier, giving each occupied cell equal total weight.

    Iteratively reweighted least squares suppresses remaining mismatches.
    Coordinates are centred and scaled to avoid conditioning problems in strips.
    No reference to validation or test data enters this estimator.
    """
    source, reference = np.asarray(source, float), np.asarray(reference, float)
    need = {"similarity": 3, "affine": 4, "homography": 6}[model]
    if len(source) < need:
        return initial
    centre = source.mean(axis=0)
    size = max(float(np.std(source)), 1.)
    T = np.array([[1 / size, 0, -centre[0] / size],
                  [0, 1 / size, -centre[1] / size], [0, 0, 1.]])
    a, b = project(T, source), project(T, reference)
    x, y = a.T
    u, v = b.T
    ones, zeros = np.ones(len(a)), np.zeros(len(a))
    if model == "similarity":
        rows_x = np.column_stack([x, -y, ones, zeros])
        rows_y = np.column_stack([y, x, zeros, ones])
    else:
        rows_x = np.column_stack([x, y, ones, zeros, zeros, zeros])
        rows_y = np.column_stack([zeros, zeros, zeros, x, y, ones])
        if model == "homography":
            rows_x = np.column_stack([rows_x, -u * x, -u * y])
            rows_y = np.column_stack([rows_y, -v * x, -v * y])
    design = np.stack([rows_x, rows_y], axis=1).reshape(2 * len(a), -1)
    target = b.ravel()
    cx, cy = cell_index(reference, shape, grid)
    cells = cy * grid[1] + cx
    counts = np.bincount(cells, minlength=grid[0] * grid[1])
    # Avoid promoting an isolated match above the combined evidence of a cell.
    base = 1 / np.maximum(counts[cells], 4)
    robust = np.ones(len(a))
    result = initial
    for _ in range(8):
        weights = np.repeat(np.sqrt(base * robust), 2)
        solution, _, rank, _ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        if rank < design.shape[1]:
            return initial
        if model == "similarity":
            c, s, tx, ty = solution
            H = np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.]])
        else:
            H = np.eye(3)
            H[:2] = solution[:6].reshape(2, 3)
            if model == "homography":
                H[2, :2] = solution[6:]
        result = np.linalg.inv(T) @ H @ T
        if not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
            return initial
        residual = np.linalg.norm(project(result, source) - reference, axis=1)
        sigma = max(.02, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        robust = np.minimum(1., 1.5 * sigma / np.maximum(residual, 1e-12))
    return result / result[2, 2]
