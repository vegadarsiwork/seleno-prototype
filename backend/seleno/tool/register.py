"""`register(source, reference, out_dir)` - the tool.

Takes two arbitrary image files, puts the source into the reference's frame, and
writes the full artifact set. Nothing about ISRO geometry, `.spm` Sun parameters
or refined corners is assumed; whatever is missing is recorded in
`metrics.json:degraded` and the run continues with reduced capability.

The output contract, in `<out_dir>/<job_id>/`:

    matches.csv      src_x, src_y, ref_x, ref_y, confidence, inlier
    registered.tif   source warped into the reference frame, georeferenced when
                     the reference was, no-data preserved
    transform.json   model type, parameters, per-segment parameters for long strips
    metrics.json     RMSE (m and px), inlier count and ratio, coverage fraction,
                     dispersion, delta Sun azimuth, scale ratio, method used,
                     status and reason
    overlay.png      checkerboard + side-by-side QC
    report.md        human-readable summary

Failure is a first-class outcome. `metrics.json` always exists; on failure it
carries `status: "failed"` and one of `unreadable_input`, `no_overlap`,
`insufficient_matches`, `verification_failed`, `degenerate_transform`.

Accuracy reporting
------------------
A spatial split is frozen before candidate selection. Only fit points influence
verification, method/model selection and fine-window placement. ECC adoption uses
a separate validation fold. The test fold is scored once, after serialization,
against the exact exported transform; it is never filtered by that transform.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import spatial, verify as V
from . import locate as LOC
from . import methods as M
from . import evaluation as E
from . import warp_model as WM
from .coordinates import grid_to_reference, project, sample_backmap
from .profiles import Profiles
from .scene import Scene, UnreadableInput, load, normalised

FAILURE_CODES = ("unreadable_input", "no_overlap", "insufficient_matches",
                 "verification_failed", "degenerate_transform")


@dataclass
class Result:
    status: str                                 # pass | warning | failed
    reason: str | None = None
    job_id: str = ""
    out_dir: str = ""
    metrics: dict = field(default_factory=dict)
    transform: dict = field(default_factory=dict)
    n_matches: int = 0

    @property
    def ok(self) -> bool:
        return self.status != "failed"


# --------------------------------------------------------------------------- #
# common frame
# --------------------------------------------------------------------------- #

# Target pixels are transformed in row blocks. rasterio's xy()/transform() take
# and return PYTHON LISTS, so a 6144^2 grid in one call materialises ~2.4 GB of
# float objects - which, with the raster copies that used to happen in scene.py,
# is what made this tool exhaust 15 GB of RAM and get OOM-killed.
_CHUNK_ROWS = 256

# Bytes of working memory per target-grid pixel. The dominant term is NOT the
# image arrays - it is the masked NCC, which correlates in "full" mode and so
# allocates FFT buffers of (2H-1) x (2W-1) in float64/complex128, several at
# once. That is ~4x the target area at 8-16 bytes each. Measured: a 2048^2 grid
# on a TMC-2/SELENE pair peaks at 3.9 GB, i.e. ~930 B per target pixel including
# the memmap pages that get touched.
_BYTES_PER_TARGET_PX = 900

# Pixel budgets for what a person looks at: each preview panel, and each panel
# of the overlay.png composite (three of them, in colour, so it is kept smaller).
_PREVIEW_PX = 4_000_000
_COMPOSITE_PANEL_PX = 1_000_000


def _available_bytes() -> int:
    """Physical memory we may use, read from the OS rather than assumed."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 4 << 30


def _cap_max_side(max_side: int, budget_fraction: float = 0.5) -> tuple[int, str | None]:
    """Shrink the working grid so the run cannot exhaust memory.

    An earlier version of this tool asked for a 6144^2 working grid on a 15 GB
    machine and was OOM-killed. The size of an unseen reference is not something
    a caller should have to reason about, so the ceiling is computed here from
    what the OS actually has free, and the reduction is reported rather than
    applied silently.
    """
    budget = _available_bytes() * budget_fraction
    allowed_px = budget / _BYTES_PER_TARGET_PX
    allowed_side = int(max(512, allowed_px ** 0.5))
    if max_side <= allowed_side:
        return max_side, None
    return allowed_side, ("working grid capped at %d px a side (asked for %d): "
                          "%.1f GB available, budgeting %.0f%% of it"
                          % (allowed_side, max_side, _available_bytes() / 1e9,
                             100 * budget_fraction))


_AA_MAX_TAPS = 4
# From this reduction up, the source is area-averaged from a block-mean base
# instead of point-sampled. At OHRC -> IIRS (about 330x) a point sample is one
# 0.24 m pixel standing for an 80 m cell - noise to every matcher. Below it the
# measured behaviour of the comb/nearest rule above is kept unchanged.
_AREA_MIN_FACTOR = 16.0


def _boxcar_decimate(arr, r0, r1, c0, c1, step, nodata, scale, offset, taps=4):
    """Decimate a raster by `step`, averaging instead of dropping pixels.

    Plain striding (`arr[r0:r1:step]`) keeps one pixel in `step` and throws the
    rest away, which is aliasing, not downsampling. It matters here because the
    two sides are almost never decimated by the same factor: a 1 m NAC mosaic
    read at native resolution against an OHRC strip nearest-sampled from 0.24 m
    is a smooth image against a noise field, and every matcher fails on it -
    measured, at native resolution: DISK 0 candidates, SIFT 0, AKAZE 3.

    Averaging a few offsets is not a proper anti-aliasing filter, but it is
    within one read of free and removes most of the noise floor. `taps` caps the
    work so a 16x decimation does not become 256 reads.

    Returns ``(decimated, (centre_row, centre_col))``: the offset, in full-
    resolution pixels from each block's first pixel, of the taps that were
    actually averaged. With the taps capped and ragged offsets skipped that is
    not the block's midpoint, and anything placed against this grid has to be
    sampled at the same point.
    """
    base = np.asarray(arr[r0:r1:step, c0:c1:step], np.float32)
    if step <= 1:
        good = np.isfinite(base) & (base > -1e30)
        if nodata is not None:
            good &= base != nodata
        return np.where(good, base * scale + offset, np.nan), (0.0, 0.0)
    k = int(min(step, taps))
    h, w = base.shape
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    used = []
    for dy in range(k):
        for dx in range(k):
            blk = np.asarray(arr[r0 + dy:r1:step, c0 + dx:c1:step], np.float32)
            blk = blk[:h, :w]
            if blk.shape != (h, w):            # ragged tail; skip this offset
                continue
            used.append((dy, dx))
            good = np.isfinite(blk) & (blk > -1e30)
            if nodata is not None:
                good &= blk != nodata
            blk = blk * scale + offset
            good &= np.isfinite(blk)
            acc += np.where(good, blk, 0.0)
            cnt += good
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    centre = tuple(float(v) for v in np.mean(used, axis=0))
    return out.astype(np.float32), centre


def _target_grid(ref: Scene, max_side: int, window=None):
    """The reference's pixel grid, optionally cropped, decimated only if needed.

    `window` is a full-resolution (r0, c0, r1, c1) box. Cropping to the region the
    source can actually reach is what makes native-resolution work affordable: a
    strip covering a fifth of a 12 288 px tile needs a 2 000 px window, not a
    6x-decimated whole tile, and the decimation is what puts a floor under the
    reported accuracy.
    """
    h, w = ref.array.shape
    r0, c0, r1, c1 = window if window else (0, 0, h, w)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(h, int(r1)), min(w, int(c1))
    if r1 - r0 < 32 or c1 - c0 < 32:
        r0, c0, r1, c1 = 0, 0, h, w
    step = max(1, int(math.ceil(max(r1 - r0, c1 - c0) / float(max_side))))
    sub, centre = _boxcar_decimate(ref.array, r0, r1, c0, c1, step, ref.nodata,
                                   ref.meta_scale, ref.meta_offset)
    v = np.isfinite(sub) & (sub > -1e30)
    return step, (r0, c0), np.nan_to_num(sub).astype(np.float32), v, centre


def _source_window(src: Scene, ref: Scene, pad_frac: float = 0.35,
                   pad_min_px: int = 256):
    """Where the source lands in the reference, as a full-resolution pixel box.

    Without this, a small source against a GLOBAL reference is hopeless. The
    WAC mosaic is 109164 x 54582; decimating it to a 2048 px working grid is
    54x, and a 734 km IIRS strip lands on it about three pixels wide. Nothing
    can be matched there, and the second pass that would have rescued it needs
    an overlap it can no longer detect.

    The padding is deliberately generous: the whole premise of this project is
    that the source geometry is wrong by kilometres, so the box has to be large
    enough to still contain the truth.
    """
    if ref.transform is None or not ref.georeferenced:
        return None
    try:
        lon = lat = None
        if src.lonlat is not None:
            g = src.lonlat
            st = max(1, g.lon.shape[0] // 64)
            lon = np.asarray(g.lon[::st, ::st], float).ravel()
            lat = np.asarray(g.lat[::st, ::st], float).ravel()
            ok = np.isfinite(lon) & np.isfinite(lat)
            lon, lat = lon[ok], lat[ok]
            if lon.size < 4:
                return None
            if ref.crs is not None and not ref.crs.is_geographic:
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform("+proj=longlat +R=1737400 +no_defs", ref.crs,
                                        lon.tolist(), lat.tolist())
                xs, ys = np.asarray(xs), np.asarray(ys)
            else:
                xs, ys = lon, lat
        elif src.georeferenced and src.transform is not None:
            h, w = src.array.shape[:2]
            t = src.transform
            cx = np.array([0, w, 0, w], float)
            cy = np.array([0, 0, h, h], float)
            xs = t.a * cx + t.b * cy + t.c
            ys = t.d * cx + t.e * cy + t.f
            if str(src.crs) != str(ref.crs):
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform(src.crs, ref.crs, xs.tolist(), ys.tolist())
                xs, ys = np.asarray(xs), np.asarray(ys)
        else:
            return None

        inv = ~ref.transform
        cols = inv.a * xs + inv.b * ys + inv.c
        rows = inv.d * xs + inv.e * ys + inv.f
        cols, rows = cols[np.isfinite(cols)], rows[np.isfinite(rows)]
        if cols.size < 4 or rows.size < 4:
            return None
        H, W = ref.array.shape[:2]
        pad_c = max(pad_min_px, pad_frac * (cols.max() - cols.min()))
        pad_r = max(pad_min_px, pad_frac * (rows.max() - rows.min()))
        r0 = int(max(0, np.floor(rows.min() - pad_r)))
        c0 = int(max(0, np.floor(cols.min() - pad_c)))
        r1 = int(min(H, np.ceil(rows.max() + pad_r)))
        c1 = int(min(W, np.ceil(cols.max() + pad_c)))
        if r1 - r0 < 32 or c1 - c0 < 32:
            return None
        if (r1 - r0) * (c1 - c0) >= 0.9 * H * W:
            return None                      # no useful narrowing
        return (r0, c0, r1, c1)
    except Exception:                                                 # noqa: BLE001
        return None


def prealign(src: Scene, ref: Scene, max_side: int = 2048, window=None, cache=None,
             prewarp=None, placement=None):
    """Put the source on the reference's grid using whatever geometry exists.

    Returns ``(src_on_grid, src_valid, ref_grid, ref_valid, back_x, back_y, info)``
    where `back_x`/`back_y` give, for each target pixel, the ORIGINAL source pixel
    it came from - so match points can always be reported in the source's own
    coordinates, whatever route was taken to get here.

    `placement` is a located reference-pixel -> source-pixel similarity (see
    `seleno.tool.locate`). It stands in for the pixel-space assumption that
    both images start at the same corner, and is ignored when a CRS route
    applies. It is recorded in the returned info, so every later call that is
    handed that info places the source the same way.
    """
    if window is None:
        window = _source_window(src, ref)
    step, (orow, ocol), R, Rv, (tap_r, tap_c) = _target_grid(ref, max_side, window)
    H, W = R.shape
    info = {"target_shape": [H, W], "reference_decimation": step,
            "reference_origin": [orow, ocol], "reference_sample_offset": [tap_r, tap_c],
            "pixel_convention": "zero-based pixel centres; GDAL affine receives centre + 0.5",
            "route": None, "notes": []}
    if window is not None:
        info["reference_window"] = list(window)

    # --- route 1: the source carries a lon/lat lattice and the reference a CRS
    if src.lonlat is not None and ref.georeferenced:
        projected = ref.crs is not None and not ref.crs.is_geographic
        f_s, f_l, prep = _lattice_interpolators(src.lonlat,
                                                ref.crs if projected else None,
                                                cache=cache)
        S = np.empty((H, W), np.float64)
        L = np.empty((H, W), np.float64)
        for r0b in range(0, H, _CHUNK_ROWS):
            r1b = min(r0b + _CHUNK_ROWS, H)
            a, b = (_grid_xy if projected else _grid_lonlat)(
                ref, orow, ocol, step, r0b, r1b, W, prewarp, (tap_r, tap_c))
            a, b = prep(a, b)
            S[r0b:r1b] = f_s(a, b)
            L[r0b:r1b] = f_l(a, b)
        info["route"] = ("source geometry lattice -> reference %s"
                         % ("projected plane" if projected else "lon/lat"))
        return _sample(src, S + 0.5, L + 0.5, R, Rv, info, cache)

    # --- route 2: both georeferenced
    if src.georeferenced and ref.georeferenced:
        from rasterio.transform import rowcol
        from rasterio.warp import transform as warp_transform
        S = np.empty((H, W), np.float64)
        L = np.empty((H, W), np.float64)
        same = str(src.crs) == str(ref.crs)
        for r0b in range(0, H, _CHUNK_ROWS):
            r1b = min(r0b + _CHUNK_ROWS, H)
            xs, ys = _grid_xy(ref, orow, ocol, step, r0b, r1b, W, prewarp, (tap_r, tap_c))
            if not same:
                xs, ys = warp_transform(ref.crs, src.crs, xs.ravel().tolist(),
                                        ys.ravel().tolist())
                xs = np.asarray(xs).reshape(r1b - r0b, W)
                ys = np.asarray(ys).reshape(r1b - r0b, W)
            rr, cc = rowcol(src.transform, xs.ravel().tolist(), ys.ravel().tolist(),
                            op=lambda v: v)
            L[r0b:r1b] = np.asarray(rr, np.float64).reshape(r1b - r0b, W)
            S[r0b:r1b] = np.asarray(cc, np.float64).reshape(r1b - r0b, W)
        info["route"] = "both georeferenced: reference CRS -> source CRS"
        return _sample(src, S, L, R, Rv, info, cache)

    # --- route 3: no CRS route. A located placement, else pixel space.
    ratio = 1.0
    if placement is not None:
        info["placement"] = np.asarray(placement, float).tolist()
    elif src.gsd_m and ref.gsd_m:
        ratio = float(src.gsd_m) / float(ref.gsd_m)
        info["notes"].append("no shared frame; source resampled by the GSD ratio "
                             "(%.4f) and registered in pixel space" % ratio)
    else:
        info["notes"].append("no shared frame and no GSD on at least one side; "
                             "registered in pixel space at native sampling. No "
                             "metre-scale accuracy can be reported.")
    # Target pixels go to FULL-RESOLUTION reference coordinates first - window
    # origin and decimation both - because that is the frame `prewarp` is written
    # in and the only one a source pixel can be looked up from. Built from the
    # window-local index alone, every fine-stage window down a strip was filled
    # from the top of the source while the reference was cut from further down,
    # and a same-file IIRS run fell from 249/253 coarse inliers to 10/226.
    #
    # Each target pixel stands where the reference's averaged taps are centred
    # (pixel-index coordinates, so a pixel's centre sits on an integer), which
    # is not the block's midpoint once the taps are capped. The source is then
    # looked up in pixel-edge coordinates, as the georeferenced routes do, so
    # that `_sample`'s truncation picks the NEAREST source pixel rather than the
    # one below: truncating a centre coordinate put a half-pixel bias into every
    # native window, and a same-file strip came back 1.1 px off.
    cols, rows = np.meshgrid(ocol + np.arange(W) * float(step) + tap_c,
                             orow + np.arange(H) * float(step) + tap_r)
    if prewarp is not None:
        P = np.asarray(prewarp, float)
        den = P[2, 0] * cols + P[2, 1] * rows + P[2, 2]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        cols, rows = ((P[0, 0] * cols + P[0, 1] * rows + P[0, 2]) / den,
                      (P[1, 0] * cols + P[1, 1] * rows + P[1, 2]) / den)
    if placement is not None:
        T = np.asarray(placement, float)
        cols, rows = (T[0, 0] * cols + T[0, 1] * rows + T[0, 2],
                      T[1, 0] * cols + T[1, 1] * rows + T[1, 2])
        # the placement lands on source pixel centres; `_sample` wants edges
        S, L = cols + 0.5, rows + 0.5
        info["route"] = "located placement (%dx decimation)" % step
        return _sample(src, S, L, R, Rv, info, cache)
    S = (cols + 0.5) / max(ratio, 1e-9)
    L = (rows + 0.5) / max(ratio, 1e-9)
    info["route"] = "pixel space (GSD ratio %.4f, %dx decimation)" % (ratio, step)
    return _sample(src, S, L, R, Rv, info, cache)


def _grid_xy(ref, orow, ocol, step, r0b, r1b, W, prewarp=None, sample_offset=None):
    """Projected (x, y) of a block of target pixel centres, as arrays.

    `prewarp` maps full-resolution reference pixel coordinates through a known
    correction before the geometry is consulted. The fine stage needs it: the
    source is placed from its own geometry, and when that geometry is wrong -
    an unrefined OHRC strip is kilometres out, which is the whole reason this
    project exists - a native window positioned on a reference feature is
    filled with ground from kilometres away and nothing in it matches. Passing
    the inverse of the coarse solution here asks the geometry for the source
    pixel the coarse solve says belongs at each reference pixel.
    """
    t = ref.transform
    dy, dx = sample_offset if sample_offset is not None else ((step - 1) / 2,) * 2
    cols = ocol + np.arange(W) * step + dx
    rows = orow + np.arange(r0b, r1b) * step + dy
    C, Rr = np.meshgrid(cols, rows)
    if prewarp is not None:
        P = np.asarray(prewarp, float)
        den = P[2, 0] * C + P[2, 1] * Rr + P[2, 2]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        C, Rr = ((P[0, 0] * C + P[0, 1] * Rr + P[0, 2]) / den,
                 (P[1, 0] * C + P[1, 1] * Rr + P[1, 2]) / den)
    x = t.a * (C + 0.5) + t.b * (Rr + 0.5) + t.c
    y = t.d * (C + 0.5) + t.e * (Rr + 0.5) + t.f
    return x, y


def _grid_lonlat(ref, orow, ocol, step, r0b, r1b, W, prewarp=None, sample_offset=None):
    """Same block, converted to lon/lat on the Moon."""
    from rasterio.warp import transform as warp_transform
    x, y = _grid_xy(ref, orow, ocol, step, r0b, r1b, W, prewarp, sample_offset)
    if ref.crs is not None and not ref.crs.is_geographic:
        lon, lat = warp_transform(ref.crs, "+proj=longlat +R=1737400 +no_defs",
                                  x.ravel().tolist(), y.ravel().tolist())
        return np.asarray(lon).reshape(x.shape), np.asarray(lat).reshape(x.shape)
    return x, y                                  # already degrees


def _lattice_interpolators(grid, crs=None, cache=None):
    """Geometry lattice -> (sample, line), built once and reused across blocks.

    Returns ``(f_sample, f_line, prep)``. `prep` maps a block of reference-frame
    coordinates into whatever space the interpolators were built in, so the
    caller does not have to know which that was.

    The space matters. A south-polar strip's lattice spans the whole 0-360
    longitude range, so a triangulation built in lon/lat carries a seam at the
    antimeridian: a query on the far side of it falls outside every triangle and
    comes back NaN. An OHRC strip against a NAC polar mosaic lands squarely on
    that seam and the run reports `no_overlap` for two images that plainly
    overlap. Near the pole the same triangulation is degenerate anyway - every
    meridian converges, so the triangles are slivers. When the reference is
    projected, interpolate in its own plane instead: polar stereographic is
    continuous across the antimeridian and well conditioned at the pole.
    """
    from scipy.interpolate import LinearNDInterpolator
    # The Delaunay build dominates this function, and the fine stage places the
    # source dozens of times over different windows of the same reference. The
    # triangulation does not depend on the window, so build it once.
    key = ("lattice", id(grid), str(crs))
    if cache is not None and key in cache:
        return cache[key]
    step = max(1, grid.lon.shape[0] // 200)      # the lattice is far finer than needed
    lon = grid.lon[::step, ::step].ravel()
    lat = grid.lat[::step, ::step].ravel()
    SS, LL = np.meshgrid(grid.pixels[::step], grid.scans[::step])

    if crs is not None and not crs.is_geographic:
        from rasterio.warp import transform as warp_transform
        x, y = warp_transform("+proj=longlat +R=1737400 +no_defs", crs,
                              lon.tolist(), lat.tolist())
        pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])

        def prep(a, b):
            return a, b                           # already the reference plane
    else:
        pts = np.column_stack([lon, lat])
        # Geographic reference: match the lattice's own longitude convention
        # rather than assume one. Lattices in this archive run 0-360; rasterio
        # hands back -180..180.
        wrap360 = float(lon.max()) > 180.0

        def prep(a, b):
            return ((a % 360.0) if wrap360 else ((a + 180.0) % 360.0 - 180.0)), b

    out = (LinearNDInterpolator(pts, SS.ravel()),
           LinearNDInterpolator(pts, LL.ravel()), prep)
    if cache is not None:
        cache[key] = out
    return out


def _area_reduced(src: Scene, f: float, cache):
    """The source area-averaged to `f` source pixels per output pixel, as
    ``(array, rows_per_px, cols_per_px)``, or None when that is not cheap.

    Cheap means from the array itself when it is small, or from the viewer's
    cached block-mean overview when that is at least as fine as `f`. Pixels the
    source marks as nodata 0 are averaged in as black: at these reductions a
    shadow is part of what the coarse image shows, not a hole in it.
    """
    key = ("area", id(src), round(float(f), 1))
    if cache is not None and key in cache:
        return cache[key]
    h, w = src.array.shape
    if h * w > 16_000_000:
        # Decide from the shape alone before building anything: an overview
        # coarser than `f` is useless here, and building one reads the file.
        from .view import _factor
        if _factor(h, w) > f:
            return None
    base = LOC._base(src, (cache or {}).get("viewcache"))
    if base is None:
        return None
    b, fy, fx = base
    if max(fy, fx) > f:
        return None
    h, w = src.array.shape
    oh, ow = max(1, int(round(h / f))), max(1, int(round(w / f)))
    valid = np.isfinite(b)
    fill = float(np.nanmean(b)) if valid.any() else 0.0
    arr = cv2.resize(np.where(valid, b, fill).astype(np.float32), (ow, oh),
                     interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA) > 0.5
    arr = arr * src.meta_scale + src.meta_offset
    arr[~vm] = np.nan
    out = (arr, h / float(oh), w / float(ow))
    if cache is not None:
        cache[key] = out
    return out


def _sample(src: Scene, S, L, R, Rv, info, cache=None):
    """Lift the source onto the target grid, averaging when it is being reduced.

    Validity is evaluated on the SAMPLED values, so no source-wide boolean mask
    is ever allocated.

    When one target pixel spans several source pixels - OHRC at 0.24 m onto a
    1 m reference grid spans about seventeen - taking the nearest one is
    aliasing, and what lands on the grid is noise rather than a smaller picture
    of the same ground. The reduction factor is measured from the sampling map
    itself rather than from metadata, so it is right even when the geometry came
    from a lattice and no GSD was declared.
    """
    h, w = src.array.shape
    ok = np.isfinite(S) & np.isfinite(L) & (S >= 0) & (S < w) & (L >= 0) & (L < h)
    out = np.zeros(R.shape, np.float32)
    ov = np.zeros(R.shape, bool)
    if ok.any():
        si = np.clip(np.nan_to_num(S).astype(np.int64), 0, w - 1)
        li = np.clip(np.nan_to_num(L).astype(np.int64), 0, h - 1)

        # Source pixels per target pixel, measured from the map itself on a
        # cheap subsample. The subsample stride has to be divided back out:
        # diffing every 16th row measures the step over 16 target pixels, not
        # one, and leaving it in inflated the factor 16x - which quietly kept
        # the filter below switched off everywhere.
        _st = 16
        Ss, Ls = S[::_st, ::_st], L[::_st, ::_st]
        with np.errstate(invalid="ignore"):
            fx = (np.nanmedian(np.abs(np.diff(Ss, axis=1))) / _st
                  if Ss.shape[1] > 1 else 1.0)
            fy = (np.nanmedian(np.abs(np.diff(Ls, axis=0))) / _st
                  if Ls.shape[0] > 1 else 1.0)
        f = float(np.nanmax([fx, fy, 1.0]))
        # Average only when the taps we can afford actually COVER the footprint,
        # i.e. when they land about a source pixel apart. Spreading four taps
        # across a seventeen-pixel footprint is a sparse comb, not a box filter:
        # it band-limits nothing and adds its own structure. Measured on the
        # coarse OHRC/NAC pass, where the reduction is 16.7x, the comb cost
        # AKAZE 442 inliers -> 70. Past that point nearest sampling is both
        # cheaper and better, so take it and record that the grid is aliased.
        k = int(np.clip(round(f), 1, _AA_MAX_TAPS))
        aa = f <= _AA_MAX_TAPS + 0.5
        info["source_oversample"] = round(f, 3)
        info["source_taps"] = k if aa else 1
        red = _area_reduced(src, f, cache) if (not aa and f >= _AREA_MIN_FACTOR) else None
        if red is not None:
            arr, ry, rx = red
            ri = np.clip((L[ok] / ry).astype(np.int64), 0, arr.shape[0] - 1)
            ci = np.clip((S[ok] / rx).astype(np.int64), 0, arr.shape[1] - 1)
            vals = arr[ri, ci]
            out[ok] = np.nan_to_num(vals)
            ov[ok] = np.isfinite(vals)
            info["source_taps"] = "area"
            info.setdefault("notes", []).append(
                "source reduced %.1fx onto the working grid by area averaging" % f)
            back_x = np.where(ok, S - 0.5, np.nan).astype(np.float64)
            back_y = np.where(ok, L - 0.5, np.nan).astype(np.float64)
            info["source_coverage"] = round(float(ov.mean()), 4)
            return out, ov, R, Rv, back_x, back_y, info
        if not aa:
            info.setdefault("notes", []).append(
                "source reduced %.1fx onto the working grid and sampled without "
                "anti-aliasing; a filter wide enough to band-limit it costs more "
                "reads than the stage is worth" % f)

        acc = np.zeros(int(ok.sum()), np.float32)
        cnt = np.zeros(int(ok.sum()), np.float32)
        offs = ((np.arange(k) - (k - 1) / 2.0) * (f / max(k, 1))) if aa else np.zeros(1)
        for dy in offs:
            lj = np.clip(li[ok] + int(round(dy)), 0, h - 1)
            for dx in offs:
                ii = np.clip(si[ok] + int(round(dx)), 0, w - 1)
                v = np.asarray(src.array[lj, ii], np.float32)
                v = v * src.meta_scale + src.meta_offset
                g = np.isfinite(v) & (v > -1e30)
                if src.nodata is not None and src.meta_scale == 1.0 \
                        and src.meta_offset == 0.0:
                    g &= v != src.nodata
                acc += np.where(g, v, 0.0)
                cnt += g
        vals = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        out[ok] = np.nan_to_num(vals)
        ov[ok] = cnt > 0
    back_x = np.where(ok, S - 0.5, np.nan).astype(np.float64)
    back_y = np.where(ok, L - 0.5, np.nan).astype(np.float64)
    info["source_coverage"] = round(float(ov.mean()), 4)
    return out, ov, R, Rv, back_x, back_y, info


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def _job_id(a: str, b: str) -> str:
    h = hashlib.sha1(("%s|%s|%.3f" % (a, b, time.time())).encode()).hexdigest()[:12]
    return h


def _fail(out_dir, job_id, code, message, extra=None):
    os.makedirs(os.path.join(out_dir, job_id), exist_ok=True)
    m = {"status": "failed", "reason": code, "message": message,
         "job_id": job_id, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if extra:
        m.update(extra)
    assert code in FAILURE_CODES, "undeclared failure code %r" % code
    with open(os.path.join(out_dir, job_id, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    with open(os.path.join(out_dir, job_id, "report.md"), "w") as fh:
        fh.write("# Registration failed\n\n**Reason:** `%s`\n\n%s\n" % (code, message))
    return Result("failed", code, job_id, os.path.join(out_dir, job_id), m)


def register(source: str, reference: str, out_dir: str = "outputs", *,
             model: str = "auto", max_side: int = 2048, grid=(8, 8),
             holdout: float = 0.35, seed: int = 0, profiles: Profiles | None = None,
             segments: int = 0, subpixel: bool = True,
             fine: bool = True, fine_tiles: int = 0, locate: str = "auto",
             progress=None, verbose: bool = True) -> Result:
    """Register `source` onto `reference`. Always writes an artifact set.

    `progress`, if given, is called with each log line as it happens, so a
    caller driving this from a server can stream real stage transitions rather
    than inventing a percentage.

    `locate` is "auto" (find where the source lies in the reference whenever
    no map projection places it - see `seleno.tool.locate`), "force" or "off".
    """
    t_start = time.time()
    profiles = profiles or Profiles.load()
    job_id = _job_id(source, reference)
    job_dir = os.path.join(out_dir, job_id)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        if verbose:
            print(msg, flush=True)
        if progress is not None:
            try:
                progress(msg)
            except Exception:                                         # noqa: BLE001
                pass                      # a broken listener must not fail a run
    max_side, cap_note = _cap_max_side(max_side)
    if cap_note:
        log("memory   : %s" % cap_note)

    # ---- 1. read -----------------------------------------------------------
    try:
        S = load(source, profiles)
        R = load(reference, profiles)
    except UnreadableInput as exc:
        return _fail(out_dir, job_id, "unreadable_input", str(exc))
    log("source    : %s" % json.dumps(S.summary()))
    log("reference : %s" % json.dumps(R.summary()))

    sp, rp = profiles.get(S.profile), profiles.get(R.profile)
    degraded = ["source: " + d for d in S.degraded] + ["reference: " + d for d in R.degraded]

    # ---- 2. where the source lies, when no map projection says -------------
    placement, locate_info = None, None
    try:
        pl = LOC.locate(S, R, os.path.join(out_dir, ".viewcache"), profiles=profiles,
                        mode=locate, log=log)
    except LOC.Disjoint as exc:
        return _fail(out_dir, job_id, "no_overlap", str(exc), {"degraded": degraded})
    except Exception as exc:                                          # noqa: BLE001
        pl = None
        log("locate    : skipped after an error (%s: %s)" % (type(exc).__name__, exc))
    if pl is not None:
        locate_info = pl.info
        if pl.T_r2s is not None:
            placement = pl.T_r2s
        else:
            # Nothing measured says where the source is. Assuming the two share a
            # corner is only defensible when their footprints are comparable,
            # and that is exactly when `locate` is not asked; so say so and stop
            # rather than register against ground chosen by assumption.
            degraded.append("locate: no unique position for the source in the reference")
            return _fail(out_dir, job_id, "insufficient_matches",
                         "the source could not be located in the reference: %s"
                         % LOC._summary(pl.info, None).replace(
                             "; no placement: the pixel-space assumption stands", ""),
                         {"locate": pl.info, "degraded": degraded})

    # ---- 2b. common frame --------------------------------------------------
    # lattice triangulations and area-reduced sources, reused by the fine stage;
    # "viewcache" is where block-mean overviews of large inputs are kept
    cache: dict = {"viewcache": os.path.join(out_dir, ".viewcache")}
    try:
        A_raw, Am, B_raw, Bm, back_x, back_y, frame = prealign(
            S, R, max_side=max_side, cache=cache, placement=placement,
            window=(pl.window if (pl is not None and placement is not None) else None))
        # Second pass: having found where the source actually lands, redo the
        # placement at the finest sampling that fits inside `max_side` over just
        # that region. Without this the reported RMSE is floored by the whole-frame
        # decimation rather than by the method.
        bb = M.overlap_bbox(Am & Bm, pad=8)
        if bb is not None and frame["reference_decimation"] > 1:
            st, (orow, ocol) = frame["reference_decimation"], frame["reference_origin"]
            win = (orow + bb[0] * st, ocol + bb[1] * st,
                   orow + bb[2] * st, ocol + bb[3] * st)
            finer = prealign(S, R, max_side=max_side, window=win, cache=cache,
                             placement=placement)
            if finer[6]["reference_decimation"] < st and (finer[1] & finer[3]).sum() > 4096:
                A_raw, Am, B_raw, Bm, back_x, back_y, frame = finer
                frame["notes"].append(
                    "refined: re-placed at %d x decimation over the overlap instead of "
                    "%d x over the whole reference" % (frame["reference_decimation"], st))
    except Exception as exc:
        return _fail(out_dir, job_id, "no_overlap",
                     "could not place the source in the reference frame: %s: %s"
                     % (type(exc).__name__, exc))
    if locate_info is not None:
        frame["locate"] = locate_info
    degraded += frame.get("notes", [])
    if cap_note:
        degraded.append(cap_note)
    both = Am & Bm
    log("frame     : %s, source covers %.1f%% of the reference grid"
        % (frame["route"], 100 * Am.mean()))
    if both.mean() < 0.02 or both.sum() < 4096:
        return _fail(out_dir, job_id, "no_overlap",
                     "the two images share %.2f%% of the reference grid (%d pixels), "
                     "which is below the 2%% / 4096 px floor"
                     % (100 * both.mean(), int(both.sum())),
                     {"frame": frame, "degraded": degraded})

    A = normalised(Scene(path=S.path, array=A_raw, valid=Am, reader=S.reader), sp)
    B = normalised(Scene(path=R.path, array=B_raw, valid=Bm, reader=R.reader), rp)

    split = E.SpatialSplit(B.shape, holdout, seed)
    fit_pixels = split.fit_mask()

    # ---- 3. pair character and candidate plan ------------------------------
    gsd_ratio = (S.gsd_m / R.gsd_m) if (S.gsd_m and R.gsd_m) else None
    character = M.pair_character(A, Am & fit_pixels, B, Bm & fit_pixels, S.profile, R.profile, gsd_ratio)
    character["overlap_fraction"] = round(float(both.mean()), 4)
    from .. import matchers as _mt
    available = {k: v["available"] for k, v in _mt.MATCHERS.items()}
    plan = M.candidate_plan(character, available)
    log("character : %s" % json.dumps(character))
    log("plan      : %s" % " -> ".join(plan))

    # ---- 4. run candidates, verify each ------------------------------------
    # Coverage is scored against cells that BOTH images actually reach. A strip
    # crossing a quarter of a map tile cannot put matches in the other three
    # quarters, and scoring it against the whole frame would refuse a good
    # registration for the wrong reason.
    eligible = spatial.eligible_cells(both, B.shape, grid)
    max_shift = _search_radius_px(S, R, sp, frame)
    attempts, best = [], None
    for name in plan:
        t0 = time.time()
        corr, note = _run_one(name, A, Am, B, Bm, max_shift, grid)
        if corr is not None:
            corr, validation, sealed_test = E.partition(corr, split)
        rec = {"method": name, "seconds": round(time.time() - t0, 2),
               "candidates": 0 if corr is None else len(corr), "note": note}
        if corr is not None and len(corr) >= 4:
            mt = model if model != "auto" else _auto_model(len(corr), character)
            vr = V.verify(corr.src, corr.ref, model_type=mt, threshold=3.0)
            rec.update({"model": mt, "inliers": int(vr.n_inliers),
                        "inlier_ratio": round(vr.inlier_ratio, 4),
                        "verified": bool(vr.ok)})
            if vr.ok and vr.n_inliers >= 6:
                cov = spatial.cell_coverage(corr.ref[vr.inlier_mask], B.shape, grid,
                                            eligible=eligible)
                disp = spatial.dispersion(corr.ref[vr.inlier_mask], B.shape)
                score = vr.n_inliers * (0.5 + cov) * (0.5 + disp)
                rec.update({"coverage": round(cov, 4), "dispersion": round(disp, 4),
                            "score": round(float(score), 2)})
                if best is None or score > best["score"]:
                    best = {"name": name, "corr": corr, "vr": vr, "model": mt,
                            "score": float(score), "validation": validation, "test": sealed_test}
        attempts.append(rec)
        log("  %-16s %s" % (name, json.dumps({k: v for k, v in rec.items()
                                              if k != "note" or v})))
        if best is not None and best["name"] == name and rec.get("inliers", 0) >= 40 \
                and rec.get("coverage", 0) >= 0.5:
            log("  (stopping early: %s is comfortably sufficient)" % name)
            break

    if best is None:
        any_cand = max((a["candidates"] for a in attempts), default=0)
        code = "insufficient_matches" if any_cand < 8 else "verification_failed"
        return _fail(out_dir, job_id, code,
                     "no candidate method produced a geometrically verified "
                     "transform; best attempt had %d correspondences" % any_cand,
                     {"attempts": attempts, "character": character,
                      "frame": frame, "degraded": degraded})

    # ---- 5. plausibility ---------------------------------------------------
    Hm = best["vr"].model
    warn = V.plausibility(Hm, best["model"], expected_scale=1.0)
    d = V.decompose(Hm) if Hm is not None else {}
    if Hm is None or not np.isfinite(Hm).all():
        return _fail(out_dir, job_id, "degenerate_transform",
                     "the estimator returned no usable model",
                     {"attempts": attempts, "degraded": degraded})
    if abs(d.get("est_scale_x", 1.0) - 1.0) > 0.25 or abs(d.get("est_rotation_deg", 0.0)) > 30.0:
        return _fail(out_dir, job_id, "degenerate_transform",
                     "the fitted transform is implausible on a shared frame: "
                     "scale %.3f, rotation %.2f deg"
                     % (d.get("est_scale_x", float("nan")),
                        d.get("est_rotation_deg", float("nan"))),
                     {"attempts": attempts, "decomposition": d, "degraded": degraded})

    # ---- 5b. fine stage at native reference resolution ---------------------
    # The coarse residual is floored by the working grid's decimation, so a
    # "sub-pixel" claim made there is a claim about a pixel several times wider
    # than either input's own. Re-measure at native resolution before believing
    # any number.
    corr, vr = best["corr"], best["vr"]
    validation, sealed_test = best["validation"], best["test"]
    step0 = int(frame.get("reference_decimation", 1) or 1)
    ocol0, orow0 = grid_to_reference(frame)[:2, 2]
    fine_info = {"attempted": False}
    per_point_back = None
    if fine:
        fine_plan = [best["name"]] + [p for p in plan if p != best["name"]]
        fcorr, fine_info = _fine_stage(S, R, corr, vr, frame, fine_plan, max_side,
                                       grid, log, cache, Hcoarse=Hm,
                                       tiles=fine_tiles, split=split)
        if fcorr is not None:
            # Back onto the working grid, where every downstream stage already
            # lives. The positions are now measured at native resolution, so
            # these are sub-working-pixel by construction.
            w_src = np.column_stack([(fcorr.src[:, 0] - ocol0) / step0,
                                     (fcorr.src[:, 1] - orow0) / step0]).astype(np.float64)
            w_ref = np.column_stack([(fcorr.ref[:, 0] - ocol0) / step0,
                                     (fcorr.ref[:, 1] - orow0) / step0]).astype(np.float64)
            whole_fine = M.Correspondences(w_src, w_ref, fcorr.confidence,
                                          fcorr.method, fcorr.kind, dict(fcorr.detail))
            fit_fine, val_fine, test_fine = E.partition(whole_fine, split)
            w_src, w_ref = fit_fine.src, fit_fine.ref
            fvr = V.verify(w_src, w_ref, model_type=best["model"], threshold=3.0 / step0)

            # Verifying is not enough to be adopted: the fine set must also not
            # fail a quality check the coarse set passed. Counting inliers alone
            # let 10 of 226 native points replace 249 of 253 coarse ones and
            # turned a clean same-file run into a warning.
            def _gates(v, ref_pts):
                inl = ref_pts[v.inlier_mask]
                return _quality_gates(
                    v.n_inliers, v.inlier_ratio,
                    spatial.cell_coverage(inl, B.shape, grid, eligible=eligible),
                    spatial.extrapolation_fraction(inl, B.shape, grid, eligible=eligible))
            verified = (fvr.model is not None
                        and fvr.n_inliers >= max(8, V._min_points(best["model"])))
            fine_gates = _gates(fvr, w_ref) if verified else {}
            regressed = [k for k in fine_gates if k not in _gates(vr, corr.ref)]
            if verified and not regressed:
                corr = fit_fine
                validation, sealed_test = val_fine, test_fine
                vr = fvr
                Hm = fvr.model
                d = V.decompose(Hm)
                per_point_back = (corr.detail["back_x"], corr.detail["back_y"])
                fine_info["adopted"] = True
                fine_info["inliers"] = int(fvr.n_inliers)
                log("fine      : %d points from %d native windows, %d verified inliers"
                    % (fine_info["points"], fine_info["windows_used"], fvr.n_inliers))
            elif verified:
                fine_info["adopted"] = False
                fine_info["note"] = ("native points fail checks the coarse set passed "
                                     "(%s); the coarse solution was kept"
                                     % "; ".join(fine_gates[k] for k in regressed))
                log("fine      : %s" % fine_info["note"])
            else:
                fine_info["adopted"] = False
                fine_info["note"] = ("native points did not verify (%d inliers); the "
                                     "coarse solution was kept" % fvr.n_inliers)
                log("fine      : %s" % fine_info["note"])
        elif fine_info.get("attempted"):
            log("fine      : not used (%s)" % fine_info.get("note"))
    fine_info.setdefault("adopted", False)

    # ---- 6. ECC: optimize only fit pixels, adopt only on validation ---------
    # The test fold remains sealed; no metric from it is used for any decision.
    ecc = {"attempted": False, "adopted": False}
    if subpixel and len(validation) >= 3:
        # Erode the source mask to exclude gradient/filter support across folds.
        fit_mask = cv2.erode((Am & fit_pixels).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ref_mask = cv2.warpPerspective(fit_mask.astype(np.uint8), Hm,
                                      (B.shape[1], B.shape[0]), flags=cv2.INTER_NEAREST).astype(bool) & Bm
        Hp, ecc = _ecc_polish(A, fit_mask, B, ref_mask, Hm, best["model"])
        if Hp is not None:
            before = V.transfer_error(Hm, validation.src.astype(np.float64), validation.ref.astype(np.float64))
            after = V.transfer_error(Hp, validation.src.astype(np.float64), validation.ref.astype(np.float64))
            old, new = float(np.sqrt(np.mean(before ** 2))), float(np.sqrt(np.mean(after ** 2)))
            ecc.update(validation_n=len(validation), validation_rmse_px_before=old,
                       validation_rmse_px_after=new, adoption_basis="validation fold, never test")
            if new < old:
                Hm = Hp
                ecc["adopted"] = True
                d = V.decompose(Hm)
                log("subpixel : ECC adopted on validation %.4f -> %.4f px" % (old, new))
            else:
                ecc["note"] = "ECC did not improve the validation fold; fit model retained"
    else:
        ecc["note"] = "ECC disabled or too few validation points"

    # ---- 7. write artifacts ------------------------------------------------
    os.makedirs(job_dir, exist_ok=True)
    m_per_px = _metres_per_pixel(R, frame)
    seg = _segment_fit(corr, vr, best["model"], B.shape, segments)

    conv = _unit_conversions(S, R, frame)
    _write_matches(job_dir, corr, vr, back_x, back_y, frame, per_point_back)
    tj = _write_transform(job_dir, Hm, best["model"], d, seg, frame, best["name"], conv)
    # JSON round-trip is the single model used by both raster export and scoring.
    with open(os.path.join(job_dir, "transform.json")) as fh:
        tj = json.load(fh)
    Hm = np.asarray(tj["matrix"], np.float64)
    warped, wvalid = (WM.warp(A_raw, Am, tj, B_raw.shape) if seg else
                       _warp(A_raw, Am, Hm, B_raw.shape, best["model"]))
    _write_registered(job_dir, warped, wvalid, R, frame)
    acc = E.score_export(job_dir, sealed_test, split)
    acc.update(subpixel_method="parabolic+ecc" if ecc.get("adopted") else "matcher", ecc=ecc)
    if acc["held_out_rmse_px"] is None:
        degraded.append("fewer than three held-out matches; accuracy is unknown, no fit-set fallback")
    ov_cov = spatial.cell_coverage(corr.ref[vr.inlier_mask], B.shape, grid,
                                   eligible=eligible)
    ov_disp = spatial.dispersion(corr.ref[vr.inlier_mask], B.shape)
    ov_extrap = spatial.extrapolation_fraction(corr.ref[vr.inlier_mask], B.shape,
                                               grid, eligible=eligible)
    metrics = _write_metrics(job_dir, job_id, S, R, character, frame, attempts, best,
                             vr, acc, ov_cov, ov_disp, m_per_px, degraded, warn, d,
                             time.time() - t_start, grid, eligible, ov_extrap,
                             conv, fine_info)
    layers = _display_layers(S, R, sp, rp, frame, A, Am, B, Bm, warped, wvalid, Hm,
                             best["model"], corr, cache, log, export_model=tj)
    _write_overlay(job_dir, *layers, vr)
    _write_preview_json(job_dir, layers[0], layers[2], layers[6], vr)
    _write_report(job_dir, job_id, S, R, metrics, tj, attempts)

    status = metrics["status"]
    log("status    : %s  (%s)" % (status, metrics.get("reason") or "all checks passed"))
    log("wrote     : %s" % job_dir)
    return Result(status, metrics.get("reason"), job_id, job_dir, metrics, tj, len(corr))


# --------------------------------------------------------------------------- #
# pieces
# --------------------------------------------------------------------------- #

def _unit_conversions(S: Scene, R: Scene, frame: dict) -> dict:
    """How to express one reference pixel in the other units that matter.

    The problem statement asks for sub-pixel accuracy *of the source image*, so
    an error has to be convertible into source pixels, not only into whatever
    grid the solver happened to run on. Anything that cannot be derived honestly
    is None rather than 1.0 - a bare PNG has no scale, and inventing one turns a
    pixel count into a fake distance.
    """
    step = int(frame.get("reference_decimation", 1) or 1)
    m_per_ref = None
    if R.gsd_m and (R.georeferenced or R.reader in ("pds4", "pds3")):
        m_per_ref = float(R.gsd_m)
    src_per_ref = None
    if S.gsd_m and R.gsd_m:
        src_per_ref = float(R.gsd_m) / float(S.gsd_m)
    elif str(frame.get("route", "")).startswith("pixel space"):
        # Registered in pixel space: the source was resampled onto the reference
        # grid by the frame's own scale factor, so that factor IS the ratio.
        try:
            src_per_ref = 1.0 / float(str(frame["route"]).split("scale ")[1].rstrip(")"))
        except Exception:                                             # noqa: BLE001
            src_per_ref = None
    return {"reference_decimation": step, "metres_per_reference_px": m_per_ref,
            "source_px_per_reference_px": src_per_ref,
            "metres_per_working_px": (m_per_ref * step) if m_per_ref else None}


def _rmse_units(rmse_ref_px, conv: dict) -> dict:
    """One error, reported in every unit it can honestly be reported in."""
    if rmse_ref_px is None:
        return {"rmse_reference_px": None, "rmse_working_px": None,
                "rmse_source_px": None, "rmse_m": None}
    r = float(rmse_ref_px)
    m = conv["metres_per_reference_px"]
    sp = conv["source_px_per_reference_px"]
    return {"rmse_reference_px": round(r, 4),
            "rmse_working_px": round(r / conv["reference_decimation"], 4),
            "rmse_source_px": (round(r * sp, 4) if sp else None),
            "rmse_m": (round(r * m, 3) if m else None)}


def _fine_stage(S: Scene, R: Scene, corr, vr, frame, plan, max_side: int,
                grid, log, cache, Hcoarse=None, tiles: int = 0, split=None):
    """Re-measure the surviving tie points at NATIVE reference resolution.

    The coarse solve runs on a decimated working grid, so the residual it can
    report is floored by that decimation, not by the method. A TMC-2 strip
    against a 3-degree SELENE tile decimates 6x: every reported pixel is 44 m
    wide, and "1.3 px" is 59 m and 10.8 SOURCE pixels. No sub-pixel claim
    survives that, whatever the internal flag says.

    Whole-frame native placement is not affordable - that is why the grid was
    decimated in the first place - so this re-places the source over a handful
    of native-resolution WINDOWS positioned on the tie points the coarse stage
    already found, and re-correlates inside each. Peak memory is one window, the
    same as one coarse pass; the windows are walked in sequence.

    Returns ``(Correspondences in FULL-RESOLUTION reference pixels, info)``, with
    `back` carrying the original source pixel for each point. Both sides live on
    the reference's own native grid, so residuals come out in reference pixels
    and convert cleanly into metres and source pixels.
    """
    info = {"attempted": True, "windows": 0, "windows_used": 0, "points": 0,
            "native": False, "note": None, "method": None, "probe": []}
    step = int(frame.get("reference_decimation", 1) or 1)
    if step <= 1:
        info.update(attempted=False, native=True,
                    note="the coarse grid was already at native reference resolution")
        return None, info

    ocol, orow = grid_to_reference(frame)[:2, 2]

    # The coarse model, expressed in full-resolution reference pixels, inverted:
    # this is what tells each native window which source ground belongs in it.
    prewarp = None
    if Hcoarse is not None:
        try:
            Hw = np.eye(3)
            Hw[:2, :] = np.asarray(Hcoarse, float)[:2, :]
            if np.asarray(Hcoarse).shape == (3, 3):
                Hw = np.asarray(Hcoarse, float)
            T = np.array([[1.0 / step, 0, -ocol / float(step)],
                          [0, 1.0 / step, -orow / float(step)],
                          [0, 0, 1.0]])
            prewarp = np.linalg.inv(np.linalg.inv(T) @ Hw @ T)
        except np.linalg.LinAlgError:
            prewarp = None

    idx = np.nonzero(vr.inlier_mask)[0]
    if len(idx) < 4:
        info.update(attempted=False, note="too few coarse inliers to place windows on")
        return None, info

    # Full-resolution reference position of every surviving tie point.
    pts = np.column_stack([ocol + corr.ref[idx, 0] * step,
                           orow + corr.ref[idx, 1] * step])

    # Spread the windows over the tie points rather than over the frame - an
    # empty window costs a placement and returns nothing - and spread them
    # EVENLY over the whole extent of those points. Walking greedily from one
    # end instead packs the windows into that end and the distribution the
    # coarse stage achieved is thrown away: the first version of this scored
    # 0.36 coverage against the coarse stage's 0.87 on the same pair, which is a
    # real loss of constraint, not a reporting artefact.
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    # Size the window so the overlap gets tiled a few times across rather than
    # swallowed by one or two boxes; coverage is a deliverable, and two windows
    # spanning a 3400 px overlap scored 0.15 where the coarse stage managed
    # 0.21 on the same pair.
    side = int(np.clip(max(x1 - x0, y1 - y0) / 3.0, 512, max_side))
    half = max(256, side // 2)
    win = 2.0 * half
    # Tile the overlap in BOTH axes. Laying windows out along the long axis only
    # leaves the cross-axis uncovered whenever the overlap is wider than one
    # window, and the coverage the coarse stage achieved is lost: on this
    # TMC-2 strip, bands-only placement scored 0.62 against the coarse 0.87,
    # because a square window spans about 60% of the strip's width.
    nx = max(1, int(math.ceil((x1 - x0) / win)))
    ny = max(1, int(math.ceil((y1 - y0) / win)))
    cand = []
    for jy in range(ny):
        for ix in range(nx):
            cx = x0 + (ix + 0.5) * (x1 - x0) / nx
            cy = y0 + (jy + 0.5) * (y1 - y0) / ny
            near = int(np.count_nonzero((np.abs(pts[:, 0] - cx) < half)
                                        & (np.abs(pts[:, 1] - cy) < half)))
            if near:
                cand.append((cx, cy, near))
    if not cand:
        info.update(attempted=False, note="no window contained a coarse tie point")
        return None, info
    budget = tiles if tiles and tiles > 0 else min(24, len(cand))
    if len(cand) > budget:
        # Thin evenly rather than by point count, so the survivors still span
        # the overlap instead of crowding where the texture happens to be.
        keep = np.linspace(0, len(cand) - 1, budget).round().astype(int)
        cand = [cand[i] for i in sorted(set(keep.tolist()))]
    centres = [(c[0], c[1]) for c in cand]
    info["windows"] = len(centres)
    info["placement"] = ("%d windows tiling %.0f x %.0f reference px (%d x %d grid)"
                         % (len(centres), x1 - x0, y1 - y0, nx, ny))

    H, W = R.array.shape
    # With the coarse solve applied, what is left is its residual - a few
    # working pixels at most - so the search does not need to be wide.
    max_shift = max(16, 4 * step) if prewarp is None else max(12, 2 * step)
    plan = [plan] if isinstance(plan, str) else list(plan)
    method = None                 # chosen by probing the first usable window
    src_all, ref_all, conf_all, bx_all, by_all = [], [], [], [], []
    for cx, cy in centres:
        r0 = int(np.clip(cy - half, 0, max(0, H - 2 * half)))
        c0 = int(np.clip(cx - half, 0, max(0, W - 2 * half)))
        win = (r0, c0, min(H, r0 + 2 * half), min(W, c0 + 2 * half))
        try:
            A, Am, B, Bm, bx, by, fr = prealign(S, R, max_side=max_side,
                                                window=win, cache=cache,
                                                prewarp=prewarp,
                                                placement=frame.get("placement"))
        except Exception as exc:                                      # noqa: BLE001
            log("fine      : window at (%d, %d) could not be placed: %s" % (c0, r0, exc))
            continue
        if fr["reference_decimation"] != 1:
            info["note"] = ("windows still decimated %dx; raise --max-side or lower "
                            "--fine-tiles" % fr["reference_decimation"])
        both_w = Am & Bm
        if both_w.sum() < 4096:
            continue
        if method is None:
            # Do NOT assume the coarse winner still wins. The whole point of
            # this stage is that it runs at a different resolution, and the
            # ranking moves with resolution: on the OHRC/NAC pair AKAZE won at
            # 4 m/px and then produced 4 usable points at 1 m/px, because
            # downsampling had been suppressing the fine shadow structure that
            # breaks descriptors. Probe the plan once, on the first usable
            # window, and carry the winner across the rest.
            for name in plan:
                try:
                    ci, _ = _run_one(name, A, Am, B, Bm, max_shift, grid)
                except Exception:                                     # noqa: BLE001
                    ci = None
                probe = ci
                if ci is not None and split is not None:
                    fr_r, fr_c = fr["reference_origin"]
                    full = project(grid_to_reference(fr), ci.src)
                    if prewarp is not None:
                        ph = np.column_stack([full, np.ones(len(full))]) @ prewarp.T
                        full = ph[:, :2] / ph[:, 2:]
                    working = (full - [ocol, orow]) / step
                    probe = E.subset(ci, split.labels(working) == split.FIT)
                n_i = 0 if probe is None else len(probe)
                vi = None
                if n_i >= 4:
                    vi = V.verify(probe.src, probe.ref, model_type="affine", threshold=3.0)
                info["probe"].append({"method": name, "candidates": n_i,
                                      "inliers": (0 if vi is None else int(vi.n_inliers))})
                if vi is not None and vi.n_inliers >= 8:
                    method, c = name, ci
                    break
            else:
                continue                  # this window suits nothing; try the next
            info["method"] = method
            log("fine      : native probe -> %s  (%s)" % (
                method, ", ".join("%s %d/%d" % (x["method"], x["inliers"], x["candidates"])
                                  for x in info["probe"])))
        else:
            c, why = _run_one(method, A, Am, B, Bm, max_shift, grid)
        if c is None or len(c) == 0:
            continue
        wr0, wc0 = fr["reference_origin"]
        st = fr["reference_decimation"]
        # window-local -> full-resolution reference pixels
        ref_all.append(project(grid_to_reference(fr), c.ref))
        src_all.append(project(grid_to_reference(fr), c.src))
        conf_all.append(np.asarray(c.confidence, np.float32))
        # original source pixel behind each point, for matches.csv
        original = sample_backmap(bx, by, c.src)
        bx_all.append(original[:, 0])
        by_all.append(original[:, 1])
        info["windows_used"] += 1

    if not ref_all:
        info["note"] = info["note"] or "no native window produced a tie point"
        return None, info

    src_pts = np.vstack(src_all)
    if prewarp is not None:
        # In a prewarped window the source arrives already carrying the coarse
        # solution, so these positions measure only the residual. Map them back
        # through the same prewarp to recover where the source geometry actually
        # put each point; refitting on those gives the FULL transform rather
        # than the leftover correction to it.
        P = np.asarray(prewarp, float)
        den = P[2, 0] * src_pts[:, 0] + P[2, 1] * src_pts[:, 1] + P[2, 2]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        src_pts = np.column_stack([
            (P[0, 0] * src_pts[:, 0] + P[0, 1] * src_pts[:, 1] + P[0, 2]) / den,
            (P[1, 0] * src_pts[:, 0] + P[1, 1] * src_pts[:, 1] + P[1, 2]) / den])

    out = M.Correspondences(
        src=src_pts.astype(np.float64),
        ref=np.vstack(ref_all).astype(np.float64),
        confidence=np.concatenate(conf_all),
        method=(method or "?") + "@native", kind="dense",
        detail={"stage": "fine", "windows": info["windows_used"]})
    out.detail["back_x"] = np.concatenate(bx_all)
    out.detail["back_y"] = np.concatenate(by_all)
    info["points"] = len(out)
    info["native"] = True
    return out, info


def _search_radius_px(S: Scene, R: Scene, sp: dict, frame: dict) -> int:
    """How far to search, from the profile's declared geolocation error.

    Phase 2 measured 3.9-4.6 km on unrefined polar OHRC; a radius under that
    cannot find the answer, and the earliest spike in this project failed for
    exactly that reason.
    """
    err_m = sp.get("expected_geolocation_error_m")
    step = frame.get("reference_decimation", 1)
    px = None
    if err_m and R.gsd_m:
        px = err_m / (R.gsd_m * step)
    if px is None:
        px = 0.25 * max(frame["target_shape"])
    return int(max(24, min(px, 0.45 * max(frame["target_shape"]))))


def _auto_model(n: int, character: dict) -> str:
    if character.get("same_sensor") and n >= 30:
        return "homography"
    return "similarity" if n < 30 else "affine"


def _run_one(name, A, Am, B, Bm, max_shift, grid):
    if name == "dense-ncc":
        t = M.dense_translation(A, Am, B, Bm, max_shift_px=max_shift)
        if t is None:
            return None, "no correlation surface"
        bbox = M.overlap_bbox(Am & Bm)
        c = M.grid_tiepoints(A, Am, B, Bm, t["dx"], t["dy"], grid=grid, bbox=bbox)
        if c is None:
            return None, "locked at (%.2f, %.2f) but no tie point survived" % (t["dx"], t["dy"])
        c.detail.update({k: t[k] for k in ("peak", "margin", "subpixel_dx",
                                           "subpixel_dy", "at_search_edge")})
        return c, ""
    return M.sparse(A, Am, B, Bm, name), ""


def _warp(src, svalid, Hm, shape, model):
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = Hm[:2, :] if Hm.shape[0] >= 2 else Hm
    if Hm.shape == (3, 3):
        H = Hm.astype(np.float64)
    out = cv2.warpPerspective(np.nan_to_num(src).astype(np.float32), H,
                              (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vm = cv2.warpPerspective(svalid.astype(np.uint8), H, (shape[1], shape[0]),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)
    return out, vm


_ECC_MOTION = {"translation": cv2.MOTION_TRANSLATION,
               "euclidean": cv2.MOTION_EUCLIDEAN,
               "similarity": cv2.MOTION_AFFINE,   # OpenCV has no similarity motion
               "affine": cv2.MOTION_AFFINE,
               "homography": cv2.MOTION_HOMOGRAPHY,
               "projective": cv2.MOTION_HOMOGRAPHY}


def _ecc_polish(A, Am, B, Bm, H0, model, iters=60, eps=1e-6,
                max_corner_shift=None):
    """Sub-pixel refinement of a verified model by ECC, warm-started from H0.

    The discrete stages cannot beat the sampling they search on. Dense NCC
    returns an integer peak - the parabolic fit pulls that back to a fraction of
    a pixel, but only along the correlation surface it was given - and sparse
    keypoints inherit their detector's localisation. ECC (Evangelidis and
    Psarakis 2008) optimises the warp against the image data directly, so it is
    bounded by neither.

    It runs on GRADIENT MAGNITUDE rather than intensity, for the reason that runs
    through this whole project: ECC maximises a correlation coefficient, which is
    invariant to linear photometric change but NOT to inversion, and inversion is
    exactly what a Sun-azimuth reversal does to these scenes - the TMC-2/SELENE
    pair measures -0.41 on raw intensity. Gradient magnitude is polarity-blind,
    so one criterion covers both sides of that flip.

    Returns (H, info). The caller decides whether to adopt it; this function
    refuses only warps that have clearly run away from the verified model. That
    guard is proportional to the frame, not a fixed pixel count: a fixed budget
    is strict on a large working grid and loose on a small one, which is the
    wrong way round. The decisive test is the separate validation RMSE the caller applies;
    this is only a rail against divergence.
    """
    motion = _ECC_MOTION.get(model)
    info = {"attempted": True, "converged": False, "adopted": False,
            "correlation": None, "max_corner_shift_px": None, "note": None}
    if motion is None or H0 is None or not np.isfinite(H0).all():
        info.update(attempted=False, note="no ECC motion model for %r" % model)
        return None, info

    ta = M.gradient_magnitude(np.nan_to_num(A).astype(np.float32), Am)
    tb = M.gradient_magnitude(np.nan_to_num(B).astype(np.float32), Bm)
    for t in (ta, tb):
        np.nan_to_num(t, copy=False)

    # Direction matters and is easy to get backwards. Our Hm maps SOURCE pixels
    # to REFERENCE pixels (cv2.warpPerspective(src, Hm) lands in the reference
    # frame). findTransformECC returns the map used with WARP_INVERSE_MAP, i.e.
    # REFERENCE to SOURCE - the inverse. Warm-starting it with Hm instead of
    # Hm^-1 starts the search at roughly double the true displacement, and it
    # diverges rather than refining.
    H0f = np.eye(3, dtype=np.float64)
    H0a = np.asarray(H0, np.float64)
    H0f[:2, :] = H0a[:2, :]
    if H0a.shape == (3, 3):
        H0f = H0a
    try:
        W0 = np.linalg.inv(H0f)
    except np.linalg.LinAlgError:
        info["note"] = "the model is singular; nothing to polish"
        return None, info
    W0 = W0 / W0[2, 2] if abs(W0[2, 2]) > 1e-12 else W0
    W = np.ascontiguousarray(
        W0 if motion == cv2.MOTION_HOMOGRAPHY else W0[:2, :], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    mask = (Am.astype(np.uint8) * 255) if Am is not None else None
    try:
        cc, W = cv2.findTransformECC(tb, ta, W, motion, crit, mask, 5)
    except cv2.error as exc:
        # ECC raises rather than returning when it cannot converge; that is a
        # normal outcome on a low-texture pair, not a bug.
        info["note"] = "ECC did not converge: %s" % str(exc).strip().splitlines()[-1][:120]
        return None, info

    Wp = np.eye(3, dtype=np.float64)
    if motion == cv2.MOTION_HOMOGRAPHY:
        Wp = np.asarray(W, np.float64)
    else:
        Wp[:2, :] = np.asarray(W, np.float64)
    if not np.isfinite(Wp).all():
        info["note"] = "ECC returned a non-finite warp"
        return None, info
    try:
        Hp = np.linalg.inv(Wp)          # back to our source -> reference convention
    except np.linalg.LinAlgError:
        info["note"] = "ECC returned a singular warp"
        return None, info
    if abs(Hp[2, 2]) > 1e-12:
        Hp = Hp / Hp[2, 2]
    if not np.isfinite(Hp).all():
        info["note"] = "ECC warp did not invert cleanly"
        return None, info

    h, w = A.shape[:2]
    if max_corner_shift is None:
        max_corner_shift = max(3.0, 0.005 * float(np.hypot(h, w)))
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    def proj(Hx):
        q = Hx @ corners
        return (q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])).T
    shift = float(np.linalg.norm(proj(Hp) - proj(H0f), axis=1).max())
    info.update(converged=True, correlation=round(float(cc), 6),
                max_corner_shift_px=round(shift, 4))
    info["max_corner_shift_allowed_px"] = round(float(max_corner_shift), 3)
    if shift > max_corner_shift:
        info["note"] = ("ECC moved the frame corners by %.2f px, beyond the %.2f px "
                        "(0.5%% of the frame diagonal) this stage is allowed to "
                        "change a verified model; rejected" % (shift, max_corner_shift))
        return None, info
    return Hp, info


def _metres_per_pixel(R: Scene, frame: dict):
    """Metres per working-grid pixel, or None when the reference has no scale.

    Returning a number here when the reference is a bare PNG would turn a pixel
    RMSE into a fake metre RMSE, which is exactly the kind of claim this project
    is supposed to refuse to make.
    """
    if R.gsd_m and R.georeferenced:
        return float(R.gsd_m) * frame.get("reference_decimation", 1)
    if R.gsd_m and R.reader in ("pds4", "pds3"):
        return float(R.gsd_m) * frame.get("reference_decimation", 1)
    return None


def _segment_fit(corr, vr, model, shape, segments):
    """Per-segment transforms for a long strip.

    A single 3x3 homography over a 730 km pushbroom strip is not slightly wrong,
    it is the wrong model class - each line has its own exterior orientation. When
    asked, the frame is split along its long axis and a transform fitted per band,
    so the residual per segment is visible rather than averaged away.
    """
    if segments < 2:
        return None
    from .. import register as _rg
    h, w = shape
    axis = 0 if h >= w else 1
    pts_s, pts_r = corr.src[vr.inlier_mask], corr.ref[vr.inlier_mask]
    coord = pts_r[:, 1] if axis == 0 else pts_r[:, 0]
    edges = np.linspace(0, h if axis == 0 else w, segments + 1)
    out = []
    for k in range(segments):
        sel = (coord >= edges[k]) & (coord < edges[k + 1])
        rec = {"segment": k, "axis": "row" if axis == 0 else "col",
               "from": float(edges[k]), "to": float(edges[k + 1]),
               "n_inliers": int(sel.sum())}
        if sel.sum() >= V._min_points(model):
            Hs = _rg.refit(pts_s[sel], pts_r[sel], model)
            if Hs is not None and np.isfinite(Hs).all() and abs(np.linalg.det(Hs)) > 1e-10:
                res = V.transfer_error(Hs, pts_s[sel].astype(np.float64),
                                       pts_r[sel].astype(np.float64))
                rec["matrix"] = np.asarray(Hs, float).tolist()
                rec["fit_rmse_px"] = round(float(np.sqrt((res ** 2).mean())), 4)
                rec["selection_basis"] = "fit fold only"
        if "matrix" not in rec:
            rec["fallback"] = "global matrix: insufficient or singular local fit"
        out.append(rec)
    return out


def _write_matches(job_dir, corr, vr, back_x, back_y, frame, per_point_back=None):
    """The match points, in the coordinates a consumer of the product needs.

    `per_point_back` carries the original source pixel measured for each point by
    the fine stage. It is preferred over the `back_x`/`back_y` lookup because
    that lookup is indexed by working-grid position and so quantises the source
    coordinate to the decimation - which would throw away exactly the precision
    the fine stage was run to obtain.
    """
    original = sample_backmap(back_x, back_y, corr.src)
    if per_point_back is not None:
        measured = np.column_stack(per_point_back)
        good = np.isfinite(measured).all(axis=1)
        original[good] = measured[good]
    reference = project(grid_to_reference(frame), corr.ref)
    with open(os.path.join(job_dir, "matches.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "confidence", "inlier"])
        for k in range(len(corr)):
            writer.writerow([*original[k], *reference[k], float(corr.confidence[k]),
                             int(vr.inlier_mask[k])])


def _write_registered(job_dir, warped, wvalid, R: Scene, frame):
    path = os.path.join(job_dir, "registered.tif")
    out = np.where(wvalid, warped, np.nan).astype(np.float32)
    try:
        import rasterio
        from rasterio.transform import Affine
        step = frame.get("reference_decimation", 1)
        tr = None
        if R.georeferenced:
            t, (orow, ocol) = R.transform, frame.get("reference_origin", [0, 0])
            dy, dx = frame.get("reference_sample_offset", [(step - 1) / 2] * 2)
            tr = (t * Affine.translation(ocol + dx + 0.5 - step / 2,
                                         orow + dy + 0.5 - step / 2) * Affine.scale(step))
        with rasterio.open(path, "w", driver="GTiff", height=out.shape[0],
                           width=out.shape[1], count=1, dtype="float32",
                           crs=R.crs if R.georeferenced else None,
                           transform=tr, nodata=float("nan"),
                           compress="deflate") as ds:
            ds.write(out, 1)
            ds.update_tags(georeferenced=str(bool(R.georeferenced)))
    except Exception:
        cv2.imwrite(path.replace(".tif", ".png"),
                    np.clip(np.nan_to_num(out) * 255, 0, 255).astype(np.uint8))


def _write_transform(job_dir, Hm, model, decomp, seg, frame, method, conv=None):
    # The working-grid matrix is what the warp uses, but it is expressed in a
    # grid that exists only inside this run. Conjugating it by the grid-to-
    # reference map gives the same transform in FULL-RESOLUTION reference
    # pixels, which is the one a consumer of the product can actually apply.
    native = None
    step = int((frame or {}).get("reference_decimation", 1) or 1)
    orow, ocol = (frame or {}).get("reference_origin", [0, 0])
    try:
        Hw = np.eye(3)
        Hw[:2, :] = np.asarray(Hm, float)[:2, :]
        if np.asarray(Hm).shape == (3, 3):
            Hw = np.asarray(Hm, float)
        C = grid_to_reference(frame)
        Hn = C @ Hw @ np.linalg.inv(C)
        if abs(Hn[2, 2]) > 1e-12:
            Hn = Hn / Hn[2, 2]
        native = Hn.tolist()
    except Exception:                                                 # noqa: BLE001
        native = None

    tj = {"model": model, "matrix": np.asarray(Hm, float).tolist(),
          "matrix_reference_px": native,
          "units": conv or {},
          "decomposition": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                            for k, v in (decomp or {}).items()},
          "frame": frame, "method": method,
          "pixel_convention": "zero-based pixel centres; world = GDAL affine * (x+0.5, y+0.5)",
          "coordinates": "`matrix` is in working-grid centre coordinates; "
                         "`matrix_reference_px` is the same transform in "
                         "full-resolution reference pixels. Both are residual corrections after "
                         "metadata prealignment. For segmented exports use the complete "
                         "blended inverse field, not the global matrix alone.",
          "segments": seg,
          "application": {"kind": "blended_segments" if seg else "global",
                          "sampling": "inverse mapping; smoothstep between segment centres" if seg else "inverse matrix",
                          "segment_fallback": "global matrix",
                          "raster": "registered.tif"}}
    with open(os.path.join(job_dir, "transform.json"), "w") as fh:
        json.dump(tj, fh, indent=1)
    return tj


def _quality_gates(n_inliers, ratio, cov, extrap) -> dict:
    """The match-quality checks a result is graded on, as {check: reason}.

    One definition, used twice: by the status line, and by the fine stage's
    adoption test - so a native-resolution set can never replace a coarse one
    while failing a check the coarse one passed.
    """
    g = {}
    if n_inliers < 12:
        g["inliers"] = "only %d verified inliers" % n_inliers
    if ratio < 0.15:
        g["inlier_ratio"] = "inlier ratio %.1f%% is below 15%%" % (100 * ratio)
    if cov < 0.25:
        g["coverage"] = ("matches cover %.0f%% of the reference grid; the transform is "
                         "extrapolated over the rest" % (100 * cov))
    if extrap > 0.5:
        # Coverage alone misses this: tie points crowded into one lit strip can
        # clear the coverage bar against a small eligible set while most of the
        # frame still sits outside their hull, where the fit is extrapolated and
        # its error is unbounded by anything we measured.
        g["extrapolation"] = ("%.0f%% of the reference area lies outside the tie-point "
                              "hull, so the transform is extrapolated there and the "
                              "quoted RMSE does not describe it" % (100 * extrap))
    return g


def _write_metrics(job_dir, job_id, S, R, character, frame, attempts, best, vr,
                   acc, cov, disp, m_per_px, degraded, warn, decomp, secs, grid,
                   eligible, extrap, conv, fine_info):
    n = int(vr.inlier_mask.size)      # the correspondence set behind THIS result,
    ratio = vr.inlier_ratio           # which is the fine set once that stage runs
    rmse_px = acc.get("held_out_rmse_px", acc.get("fit_rmse_px"))
    d_az = None
    if S.sun_azimuth_deg is not None and R.sun_azimuth_deg is not None:
        d = abs(S.sun_azimuth_deg - R.sun_azimuth_deg) % 360.0
        d_az = round(min(d, 360.0 - d), 2)

    units = _rmse_units(None if rmse_px is None else rmse_px * conv["reference_decimation"],
                        conv)

    reasons = list(_quality_gates(vr.n_inliers, ratio, cov, extrap).values())
    if character.get("overlap_fraction", 1) < 0.15:
        reasons.append("the images share only %.0f%% of the frame"
                       % (100 * character["overlap_fraction"]))
    # A limit of the reference is not a defect in the registration, so this is
    # stated rather than counted against the run's status. Quoting "sub-pixel:
    # False" without it would read as a shortfall in the matching when in fact
    # no method could do better against a reference this coarse.
    notes = []
    sp_floor = conv.get("source_px_per_reference_px")
    if sp_floor and sp_floor >= 1.0 and units["rmse_source_px"] is not None:
        notes.append("one reference pixel is %.2f source pixels, so sub-source-pixel "
                     "accuracy is not reachable against this reference at all; the "
                     "result is %.2f source px, %.2f reference px, %s m"
                     % (sp_floor, units["rmse_source_px"], units["rmse_reference_px"],
                        units["rmse_m"]))
    reasons += list(warn or [])
    status = "pass" if not reasons else "warning"

    m = {"status": status, "reason": "; ".join(reasons) if reasons else None,
         "notes": notes,
         "job_id": job_id, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "runtime_s": round(secs, 2),
         "method_used": best["name"], "model": best["model"],
         "accuracy": dict(
             acc, rmse_px=rmse_px,
             # The same error in every unit it can honestly be stated in. The
             # working grid is an internal artefact of this run, so quoting a
             # residual only in its pixels says nothing about either input.
             **units,
             metres_per_pixel=m_per_px,
             units=conv,
             # The problem statement asks for sub-pixel accuracy OF THE SOURCE
             # IMAGE, so that is what the flag reports. It is None, not False,
             # when there is no scale to convert with - a bare PNG cannot answer
             # the question either way.
             subpixel=(None if units["rmse_source_px"] is None
                       else bool(units["rmse_source_px"] < 1.0)),
             subpixel_basis="source pixels",
             subpixel_working_grid=bool(rmse_px is not None and rmse_px < 1.0),
             # One reference pixel, expressed in source pixels. Nothing can
             # localise a source pixel against a reference better than about a
             # reference pixel, so when this is >= 1 a sub-source-pixel result
             # is not available from this pair at all - no method would give it,
             # and the honest thing is to say so rather than keep reporting
             # False as though it were a shortfall in the matching.
             subpixel_floor_source_px=(round(conv["source_px_per_reference_px"], 3)
                                       if conv["source_px_per_reference_px"] else None),
             subpixel_attainable=(None if not conv["source_px_per_reference_px"]
                                  else bool(conv["source_px_per_reference_px"] < 1.0))),
         "fine_stage": fine_info,
         "matches": {"candidates": n, "inliers": int(vr.n_inliers),
                     "inlier_ratio": round(ratio, 4)},
         "distribution": {"grid": list(grid), "coverage_fraction": round(cov, 4),
                          "dispersion": round(disp, 4),
                          "extrapolation_fraction": round(extrap, 4),
                          "eligible_cells": len(eligible), "total_cells": grid[0] * grid[1],
                          "note": "coverage is the fraction of ELIGIBLE reference grid "
                                  "cells holding at least one inlier, where eligible "
                                  "means both images have data there; dispersion is the "
                                  "mean distance from the centroid over the half-diagonal; "
                                  "extrapolation_fraction is the share of eligible cells "
                                  "OUTSIDE the convex hull of the inliers, where the "
                                  "transform is extrapolated rather than interpolated"},
         "illumination": {"delta_sun_azimuth_deg": d_az,
                          "source_sun_azimuth_deg": S.sun_azimuth_deg,
                          "reference_sun_azimuth_deg": R.sun_azimuth_deg,
                          "source_sun_incidence_deg": S.sun_incidence_deg,
                          "reference_sun_incidence_deg": R.sun_incidence_deg},
         "pair": character, "frame": frame, "decomposition": decomp,
         "attempts": attempts,
         "source": S.summary(), "reference": R.summary(),
         "degraded": degraded}
    with open(os.path.join(job_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def _display_layers(S, R, sp, rp, frame, A, Am, B, Bm, warped, wvalid, Hm, model,
                    corr, cache, log, export_model=None):
    """The preview panels at a resolution meant for looking at.

    The working grid is sized for matching, not viewing: `max_side` caps the LONG
    axis, so a 250 x 12945 IIRS strip comes out 20 px across, and the viewer
    blew that up 28x into a blur. Here the panels are placed again from the
    inputs themselves over the same window, at the finest sampling that fits
    `_PREVIEW_PX` - native, for that strip - and the transform and tie points
    are carried onto them by the same grid-to-grid map.

    Returns ``(A, Am, B, Bm, warped, wvalid, corr)`` on the display grid, or the
    working-grid ones when they are already that fine or the placement fails: a
    preview is never worth failing a run over.
    """
    same = (A, Am, B, Bm, warped, wvalid, corr)
    step_w = int(frame.get("reference_decimation", 1) or 1)
    orow_w, ocol_w = frame.get("reference_origin", [0, 0])
    rh, rw = R.array.shape
    win = (orow_w, ocol_w, min(rh, orow_w + B.shape[0] * step_w),
           min(rw, ocol_w + B.shape[1] * step_w))
    long_side = max(win[2] - win[0], win[3] - win[1])
    area = (win[2] - win[0]) * (win[3] - win[1])
    step_d = max(1, int(math.ceil(math.sqrt(area / float(_PREVIEW_PX)))))
    if step_d >= step_w:
        return same
    try:
        A_raw, Am_d, B_raw, Bm_d, _, _, fr = prealign(
            S, R, max_side=int(math.ceil(long_side / float(step_d))), window=win,
            cache=cache, placement=frame.get("placement"))
        step_d = int(fr["reference_decimation"])
        orow_d, ocol_d = fr["reference_origin"]
        # working-grid pixels -> display-grid pixels, through full resolution
        k = step_w / float(step_d)
        D = np.linalg.inv(grid_to_reference(fr)) @ grid_to_reference(frame)
        Hw = np.eye(3)
        Hw[:2, :] = np.asarray(Hm, float)[:2, :]
        if np.asarray(Hm).shape == (3, 3):
            Hw = np.asarray(Hm, float)
        warped_d, wvalid_d = (WM.warp(A_raw, Am_d, WM.conjugate(export_model, D), B_raw.shape)
                              if export_model and export_model.get("segments") else
                              _warp(A_raw, Am_d, D @ Hw @ np.linalg.inv(D), B_raw.shape, model))
        A_d = normalised(Scene(path=S.path, array=A_raw, valid=Am_d, reader=S.reader), sp)
        B_d = normalised(Scene(path=R.path, array=B_raw, valid=Bm_d, reader=R.reader), rp)

        def to_d(p):
            return (np.asarray(p, np.float64) * k + D[:2, 2]).astype(np.float32)

        corr_d = M.Correspondences(src=to_d(corr.src), ref=to_d(corr.ref),
                                   confidence=corr.confidence, method=corr.method,
                                   kind=corr.kind, detail=corr.detail)
        log("preview   : panels at %d x %d (%dx decimation; working grid %dx)"
            % (B_d.shape[1], B_d.shape[0], step_d, step_w))
        return A_d, Am_d, B_d, Bm_d, warped_d, wvalid_d, corr_d
    except Exception as exc:                                          # noqa: BLE001
        log("preview   : kept the working grid (%s: %s)" % (type(exc).__name__, exc))
        return same


def _spread(idx, n):
    """At most `n` of `idx`, evenly spaced through it.

    Taking the first `n` instead draws only the start of the set: fine-stage
    points arrive window by window, so the first 600 of 20 000 all sit in the
    first window and a strip's match lines bunched at one end.
    """
    if n <= 0:
        return idx[:0]
    if len(idx) <= n:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def _write_overlay(job_dir, A, Am, B, Bm, warped, wvalid, corr, vr):
    def u8(a, m):
        return M.to_u8(a, m)

    # Separate panels first, at full preview resolution. A UI needs the layers
    # apart so it can toggle between them and draw its own match lines at
    # whatever zoom the user is at.
    cv2.imwrite(os.path.join(job_dir, "source.png"), u8(A, Am))
    cv2.imwrite(os.path.join(job_dir, "reference.png"), u8(B, Bm))
    cv2.imwrite(os.path.join(job_dir, "registered.png"), u8(warped, wvalid))

    # The composite is what goes in a report. It holds three panels in colour,
    # so it is reduced to its own budget rather than written at full size.
    f = min(1.0, math.sqrt(_COMPOSITE_PANEL_PX / float(max(1, B.size))))
    if f < 1.0:
        def rs(a, interp=cv2.INTER_AREA):
            return cv2.resize(np.asarray(a, np.float32), None, fx=f, fy=f,
                              interpolation=interp)
        A, B, warped = rs(A), rs(B), rs(warped)
        Am, Bm, wvalid = (rs(m.astype(np.float32), cv2.INTER_NEAREST) > 0.5
                          for m in (Am, Bm, wvalid))
        corr = M.Correspondences(src=corr.src * f, ref=corr.ref * f,
                                 confidence=corr.confidence, method=corr.method,
                                 kind=corr.kind)

    Bu = u8(B, Bm)
    Wu = u8(warped, wvalid)
    h, w = Bu.shape
    tiles = 8
    checker = Bu.copy()
    ty, tx = max(1, h // tiles), max(1, w // tiles)
    for j in range(tiles + 1):
        for i in range(tiles + 1):
            if (i + j) % 2 == 0:
                sl = (slice(j * ty, min((j + 1) * ty, h)), slice(i * tx, min((i + 1) * tx, w)))
                checker[sl] = np.where(wvalid[sl], Wu[sl], Bu[sl])
    side = np.hstack([cv2.cvtColor(u8(A, Am), cv2.COLOR_GRAY2BGR),
                      np.full((h, 12, 3), 255, np.uint8),
                      cv2.cvtColor(Bu, cv2.COLOR_GRAY2BGR)])
    for k in _spread(np.nonzero(vr.inlier_mask)[0], 400):
        p1 = tuple(np.round(corr.src[k]).astype(int))
        p2 = tuple(np.round(corr.ref[k]).astype(int) + np.array([w + 12, 0]))
        cv2.line(side, p1, p2, (90, 220, 120), 1, cv2.LINE_AA)
    ch3 = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    pad = abs(side.shape[1] - ch3.shape[1])
    if side.shape[1] > ch3.shape[1]:
        ch3 = np.hstack([ch3, np.full((h, pad, 3), 255, np.uint8)])
    else:
        side = np.hstack([side, np.full((side.shape[0], pad, 3), 255, np.uint8)])
    cv2.imwrite(os.path.join(job_dir, "overlay.png"),
                np.vstack([side, np.full((14, side.shape[1], 3), 255, np.uint8), ch3]))


def _write_preview_json(job_dir, A, B, corr, vr, limit=600):
    """Match coordinates in PREVIEW-PANEL pixels (source.png, reference.png), for drawing.

    matches.csv stays exactly as the contract specifies - original source pixels
    and full-resolution reference pixels - because that is what anyone consuming
    the product needs. Those are not the coordinates of the preview panels, and
    recovering one from the other needs the sampling map, which is not part of
    the artifact set. So the display coordinates are written separately rather
    than by bending the deliverable to suit the viewer.
    """
    n = len(corr)
    keep = np.arange(n)
    if n > limit:                       # keep every inlier we can, then fill
        inl = _spread(np.nonzero(vr.inlier_mask)[0], limit)
        out = _spread(np.nonzero(~vr.inlier_mask)[0], limit - len(inl))
        keep = np.concatenate([inl, out])
    pv = {"source": {"width": int(A.shape[1]), "height": int(A.shape[0])},
          "reference": {"width": int(B.shape[1]), "height": int(B.shape[0])},
          "n_total": int(n), "n_shown": int(len(keep)),
          "matches": [{"sx": round(float(corr.src[k][0]), 2),
                       "sy": round(float(corr.src[k][1]), 2),
                       "rx": round(float(corr.ref[k][0]), 2),
                       "ry": round(float(corr.ref[k][1]), 2),
                       "c": round(float(corr.confidence[k]), 4),
                       "inlier": bool(vr.inlier_mask[k])} for k in keep]}
    with open(os.path.join(job_dir, "preview.json"), "w") as fh:
        json.dump(pv, fh)


def _write_report(job_dir, job_id, S, R, m, tj, attempts):
    a, d = m["accuracy"], m["distribution"]
    L = ["# Registration report\n",
         "Job `%s`, %s.\n" % (job_id, m["generated_utc"]),
         "| | source | reference |", "|---|---|---|",
         "| file | `%s` | `%s` |" % (os.path.basename(S.path), os.path.basename(R.path)),
         "| reader | %s | %s |" % (S.reader, R.reader),
         "| instrument | %s | %s |" % (S.instrument, R.instrument),
         "| GSD (m) | %s | %s |" % (S.gsd_m, R.gsd_m),
         "| georeferenced | %s | %s |" % (S.georeferenced, R.georeferenced),
         "\n## Result\n",
         "**Status: %s**%s\n" % (m["status"].upper(),
                                 ("  \n" + m["reason"]) if m.get("reason") else ""),
         "| metric | value |", "|---|--:|",
         "| method used | `%s` |" % m["method_used"],
         "| model | %s |" % m["model"],
         "| candidates | %d |" % m["matches"]["candidates"],
         "| inliers | %d |" % m["matches"]["inliers"],
         "| inlier ratio | %.1f%% |" % (100 * m["matches"]["inlier_ratio"]),
         "| RMSE (px) | %s |" % a.get("rmse_px"),
         "| RMSE (m) | %s |" % a.get("rmse_m"),
         "| held-out points | %s |" % a.get("held_out_n"),
         "| sub-pixel | %s |" % a.get("subpixel"),
         "| coverage fraction | %.2f |" % d["coverage_fraction"],
         "| dispersion | %.2f |" % d["dispersion"],
         "| extrapolated area | %.0f%% |" % (100 * d.get("extrapolation_fraction", 0.0)),
         "| Δ Sun azimuth | %s |" % m["illumination"]["delta_sun_azimuth_deg"],
         "| scale ratio | %s |" % m["pair"]["scale_ratio"],
         "| cross-correlation | %s |" % m["pair"]["cross_correlation"],
         "| runtime (s) | %s |" % m["runtime_s"],
         "\n## Methods tried\n",
         "| method | candidates | inliers | ratio | coverage | seconds |",
         "|---|--:|--:|--:|--:|--:|"]
    for t in attempts:
        L.append("| `%s` | %d | %s | %s | %s | %s |"
                 % (t["method"], t.get("candidates", 0), t.get("inliers", "-"),
                    t.get("inlier_ratio", "-"), t.get("coverage", "-"), t.get("seconds")))
    L += ["\nThe winner is chosen by geometric verification, not by a hardcoded",
          "preference: Phase 2 measured that sparse learned matching wins on",
          "same-sensor illumination change while dense NCC wins on anti-correlated",
          "cross-sensor pairs, so the tool tries both and reports which won.\n"]
    if m.get("degraded"):
        L += ["## What was missing\n"]
        L += ["- %s" % x for x in m["degraded"]]
        L.append("")
    L += ["\n## Artifacts\n",
          "`matches.csv`, `registered.tif`, `transform.json`, `metrics.json`,",
          "`overlay.png`, `report.md`\n"]
    with open(os.path.join(job_dir, "report.md"), "w") as fh:
        fh.write("\n".join(L))
