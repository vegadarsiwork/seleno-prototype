"""Stream original bands onto the native reference grid.

The registration grid is only an estimation grid. Export composes the complete
residual inverse field with the original geometry, then samples each original
band once. It never enlarges the decimated pseudo-pan or interpolates a coarse
backmap at the edge of its support.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from .coordinates import grid_to_reference, project
from . import warp_model


class ExportTooLarge(RuntimeError):
    """The multi-band raster would not fit safely on the output disk."""


def reference_to_source(source, reference, frame, points, *,
                        lattice_interpolators=None, cache=None):
    """Metadata prealignment: native reference centres -> native source centres.

    These are the coordinates *before* the fitted residual correction. Callers
    evaluating the exported mapping must first apply that correction's inverse.
    The lattice factory is the same one used during registration; accepting it
    here keeps export independent of the registration orchestrator.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    out = np.full(points.shape, np.nan, np.float64)
    finite = np.isfinite(points).all(axis=1)
    p = points[finite]
    if not len(p):
        return out
    if (source.lonlat is not None or source.georeferenced) and reference.georeferenced:
        t = reference.transform
        x = t.a * (p[:, 0] + .5) + t.b * (p[:, 1] + .5) + t.c
        y = t.d * (p[:, 0] + .5) + t.e * (p[:, 1] + .5) + t.f
        if source.lonlat is not None:
            if lattice_interpolators is None:
                raise ValueError("source geometry lattice requires its interpolation factory")
            crs = reference.crs if not reference.crs.is_geographic else None
            fs, fl, prep = lattice_interpolators(source.lonlat, crs, cache=cache)
            x, y = prep(x, y)
            mapped = np.column_stack([fs(x, y), fl(x, y)])
        else:
            if source.crs != reference.crs:
                from rasterio.warp import transform
                x, y = transform(reference.crs, source.crs, x.tolist(), y.tolist())
                x, y = np.asarray(x), np.asarray(y)
            inv = ~source.transform
            mapped = np.column_stack([inv.a * x + inv.b * y + inv.c - .5,
                                      inv.d * x + inv.e * y + inv.f - .5])
    elif frame.get("placement") is not None:
        mapped = project(frame["placement"], p)
    else:
        ratio = (float(source.gsd_m) / float(reference.gsd_m)
                 if source.gsd_m and reference.gsd_m else 1.)
        mapped = (p + .5) / ratio - .5
    out[finite] = mapped
    return out


def sample_band(band, points):
    """Bilinear original DN samples with finite, nodata and explicit mask checks.

    Invalid neighbours with a nonzero interpolation weight invalidate a sample.
    Exact integer centres only require their own pixel. The outer half-pixel
    rim uses the closest centre, matching the source pixel's physical extent.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    h, w = band.array.shape
    x, y = points.T
    inside = (np.isfinite(points).all(axis=1) & (x >= -.5) & (x < w - .5)
              & (y >= -.5) & (y < h - .5))
    out = np.full(len(points), np.nan, np.float64)
    if not inside.any():
        return out
    xx, yy = np.clip(x[inside], 0, w - 1), np.clip(y[inside], 0, h - 1)
    x0, y0 = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xx - x0, yy - y0
    value = np.zeros(len(xx), np.float64)
    valid = np.ones(len(xx), bool)
    for rr, cc, weight in ((y0, x0, (1 - fx) * (1 - fy)),
                            (y0, x1, fx * (1 - fy)),
                            (y1, x0, (1 - fx) * fy), (y1, x1, fx * fy)):
        active = weight > 1e-12
        if not active.any():
            continue
        raw = np.asarray(band.array[rr[active], cc[active]], np.float64)
        good = np.isfinite(raw)
        if band.nodata is not None:
            good &= raw != band.nodata
        if band.valid is not None:
            good &= np.asarray(band.valid[rr[active], cc[active]], bool)
        valid[active] &= good
        value[active] += np.where(good, raw, 0.) * weight[active]
    out[inside] = np.where(valid, value, np.nan)
    return out


def _native_window(reference, frame):
    h, w = reference.shape
    r0, c0 = map(int, frame.get("reference_origin", [0, 0]))
    r1, c1 = h, w
    if frame.get("reference_window") is not None:
        _, _, r1, c1 = map(int, frame["reference_window"])
    r0, c0, r1, c1 = max(0, r0), max(0, c0), min(h, r1), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("empty native reference export window")
    return r0, c0, r1, c1


def write_registered_native(job_dir, source, reference, frame, model, *,
                            lattice_interpolators=None, cache=None, tile_size=128,
                            window=None, supersample=None, step=1):
    """Write all source bands at native reference resolution with bounded RAM.

    Returns export dimensions and sampling metadata suitable for metrics.json.
    Original values retain their original scale/offset, descriptions and units.
    A float output represents interpolation without integer quantisation; 64-bit
    source values retain a 64-bit output. BigTIFF is selected automatically.

    `window` (r0, c0, r1, c1, native reference px) limits the raster to where
    the registered source actually lies; without it the frame's whole search
    window is written. When one output pixel spans several source pixels -
    OHRC at 0.24 m onto a 1 m NAC grid spans four - a single bilinear sample
    is aliasing, so each output pixel averages a k x k grid of samples spread
    over its footprint through the local Jacobian of the mapping (k = the
    measured reduction, at most 4; `supersample` fixes it).

    `step` > 1 writes every step-th native reference pixel (output pixels are
    step x step native ones, on the same grid lines). It is for a reference
    much finer than the source: a 55 m IIRS cube on a 7.4 m TC grid would
    otherwise be 50 times the pixels - terabytes for 256 bands - with nothing
    the source did not already say.
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    step = int(step)
    if step < 1:
        raise ValueError("step must be a positive integer")
    cache = {} if cache is None else cache
    bands = source.bands
    r0, c0, r1, c1 = _native_window(reference, frame)
    if window is not None:
        h, w = reference.shape
        r0, c0 = max(0, int(window[0])), max(0, int(window[1]))
        r1, c1 = min(h, int(window[2])), min(w, int(window[3]))
        if r1 <= r0 or c1 <= c0:
            raise ValueError("empty native reference export window")
    height, width = -(-(r1 - r0) // step), -(-(c1 - c0) // step)
    c = grid_to_reference(frame)
    cinv = np.linalg.inv(c)
    # float32 cannot retain all 32-bit integer digital numbers.
    dtypes = [np.dtype(b.array.dtype) for b in bands]
    dtype = "float64" if any(d.itemsize > 4 or (d.kind in "iu" and d.itemsize >= 4)
                              for d in dtypes) else "float32"
    transform = ((reference.transform if reference.georeferenced else Affine.identity())
                 * Affine.translation(c0, r0) * Affine.scale(step))
    estimate = height * width * len(bands) * np.dtype(dtype).itemsize
    free = shutil.disk_usage(job_dir).free
    if estimate > 0.5 * free:
        # Compression usually shrinks this a lot, but a run must never be able
        # to fill the disk it shares with everything else.
        raise ExportTooLarge("registered.tif would be %.1f GB uncompressed (%d x %d x %d bands); "
                             "only %.1f GB is free" % (estimate / 1e9, height, width, len(bands),
                                                        free / 1e9))
    path = os.path.join(job_dir, "registered.tif")
    temporary = path + ".partial.tif"
    try:
        with rasterio.open(temporary, "w", driver="GTiff", height=height, width=width,
                           count=len(bands), dtype=dtype,
                           crs=reference.crs if reference.georeferenced else None,
                           transform=transform, nodata=float("nan"), compress="deflate",
                           predictor=3, tiled=True, blockxsize=128, blockysize=128,
                           BIGTIFF="IF_SAFER", interleave="band") as ds:
            ds.update_tags(**source.meta.get("raster_tags", {}))
            ds.update_tags(georeferenced=str(bool(reference.georeferenced)),
                           grid=("native reference pixel centres" if step == 1 else
                                 "every %d native reference pixels, centred in each block" % step),
                           reference_step=str(step),
                           reference_origin_row=str(r0), reference_origin_col=str(c0),
                           resampling="bilinear original bands",
                           transform_sha256=warp_model.digest(model))
            ds.scales = tuple(b.scale for b in bands)
            ds.offsets = tuple(b.offset for b in bands)
            for number, band in enumerate(bands, 1):
                if band.description:
                    ds.set_band_description(number, band.description)
                if band.unit:
                    ds.set_band_unit(number, band.unit)
                if band.tags:
                    ds.update_tags(number, **band.tags)
            used = set()
            for row in range(0, height, tile_size):
                bh = min(tile_size, height - row)
                for col in range(0, width, tile_size):
                    bw = min(tile_size, width - col)
                    # One extra row and column: their differences are the
                    # mapping's Jacobian, which sizes the supersampling.
                    yy, xx = np.mgrid[row:row + bh + 1, col:col + bw + 1]
                    # Output pixel j covers native j*step .. j*step+step-1; its centre:
                    centre = (step - 1) / 2.
                    native = np.column_stack([xx.ravel() * step + c0 + centre,
                                              yy.ravel() * step + r0 + centre])
                    prealigned = project(c, warp_model.inverse_points(model, project(cinv, native)))
                    grid = reference_to_source(
                        source, reference, frame, prealigned,
                        lattice_interpolators=lattice_interpolators,
                        cache=cache).reshape(bh + 1, bw + 1, 2)
                    source_xy = grid[:bh, :bw].reshape(-1, 2)
                    jx = (grid[:bh, 1:bw + 1] - grid[:bh, :bw]).reshape(-1, 2)
                    jy = (grid[1:bh + 1, :bw] - grid[:bh, :bw]).reshape(-1, 2)
                    k = supersample
                    if k is None:
                        scales = np.maximum(np.linalg.norm(jx, axis=1), np.linalg.norm(jy, axis=1))
                        finite_scales = scales[np.isfinite(scales)]
                        reduction = np.median(finite_scales) if finite_scales.size else 1.
                        k = int(np.clip(np.round(reduction), 1, 4)) if np.isfinite(reduction) else 1
                    k = max(1, int(k))
                    used.add(k)
                    offsets = (np.arange(k) + .5) / k - .5
                    window_ = Window(col, row, bw, bh)
                    for number, band in enumerate(bands, 1):
                        if k == 1:
                            pixels = sample_band(band, source_xy)
                        else:
                            total = np.zeros(len(source_xy))
                            count = np.zeros(len(source_xy))
                            for oy in offsets:
                                for ox in offsets:
                                    v = sample_band(band, source_xy + ox * jx + oy * jy)
                                    good = np.isfinite(v)
                                    total += np.where(good, v, 0.)
                                    count += good
                            # A pixel keeps a value only where most of its footprint does.
                            pixels = np.where(count >= .5 * k * k, total / np.maximum(count, 1), np.nan)
                        ds.write(pixels.reshape(bh, bw).astype(dtype), number, window=window_)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {"raster": "registered.tif", "shape": [height, width], "bands": len(bands),
            "reference_decimation": step, "reference_origin": [r0, c0],
            "resolution": ("native reference" if step == 1 else
                           "every %d native reference pixels (source coarser than reference)" % step),
            "dtype": dtype,
            "supersampling": sorted(used) if height and width else [],
            "sampling": "bilinear original bands, averaged over k x k samples per output "
                        "pixel where the source is finer; complete inverse registration field",
            "radiometry": "stored values with original per-band scale and offset"}
