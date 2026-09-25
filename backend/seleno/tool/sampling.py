"""Continuous, nodata-aware sampling of large (possibly memory-mapped) rasters."""
import numpy as np


def bilinear(array, x, y, *, nodata=None, scale=1., offset=0.):
    """Sample zero-based pixel centres without OpenCV's 32767-pixel limit.

    Valid neighbours are renormalized; invalid values are excluded before any
    radiometric scaling. Work is proportional to the requested coordinates.
    """
    h, w = array.shape
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    inside = np.isfinite(x) & np.isfinite(y) & (x >= -.5) & (y >= -.5)
    inside &= (x < w - .5) & (y < h - .5)
    xx = np.clip(np.nan_to_num(x), 0, w - 1)
    yy = np.clip(np.nan_to_num(y), 0, h - 1)
    ix, iy = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    dx, dy = xx - ix, yy - iy
    total, weight = np.zeros(x.shape), np.zeros(x.shape)
    for j, wy in ((0, 1 - dy), (1, dy)):
        for i, wx in ((0, 1 - dx), (1, dx)):
            v = np.asarray(array[np.minimum(iy + j, h - 1), np.minimum(ix + i, w - 1)], float)
            good = inside & np.isfinite(v) & (v > -1e30)
            if nodata is not None:
                good &= v != nodata
            ww = wx * wy * good
            total += np.where(good, v, 0) * ww
            weight += ww
    valid = inside & (weight > 1e-6)
    out = total / np.maximum(weight, 1e-12) * scale + offset
    return np.where(valid, out, np.nan).astype(np.float32)
