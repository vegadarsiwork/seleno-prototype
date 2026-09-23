"""Zero-based pixel centres everywhere; GDAL affines act on pixel edges.

Centre (x, y) has world position affine * (x + .5, y + .5). A reduced
working pixel represents the centre of the samples actually averaged, not
necessarily the middle of its stride when the averaging taps are capped.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def grid_to_reference(frame):
    step = frame.get("reference_decimation", 1)
    row, col = frame.get("reference_origin", [0, 0])
    dy, dx = frame.get("reference_sample_offset", [0, 0])
    return np.array([[step, 0, col + dx], [0, step, row + dy], [0, 0, 1]], np.float64)


def project(matrix, points):
    p = np.asarray(points, np.float64)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(matrix, np.float64).T
    return q[:, :2] / q[:, 2:]


def sample_backmap(back_x, back_y, points):
    """Evaluate the continuous source mapping, preserving fractional keypoints."""
    p = np.asarray(points, np.float64)
    args = dict(order=1, mode="nearest", prefilter=False)
    return np.column_stack([map_coordinates(back_x, p[:, ::-1].T, **args),
                            map_coordinates(back_y, p[:, ::-1].T, **args)])
