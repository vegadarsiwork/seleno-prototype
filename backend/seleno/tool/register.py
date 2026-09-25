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
from . import terrain
from .fitting import balanced_refit
from .export import write_registered_native, reference_to_source
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

# Search half-width of the coarse dense tie points (`methods.grid_tiepoints`),
# which bounds where a wrong dense match can land.
_DENSE_SEARCH = 12


def _cgroup_headroom() -> int | None:
    """Bytes left under this process's cgroup v2 memory limit, if it has one.

    MemAvailable describes the whole machine. Inside a container or a
    memory-capped scope the limit that matters is the cgroup's, and budgeting
    against the machine figure is how a capped run gets OOM-killed.
    """
    try:
        with open("/proc/self/cgroup") as fh:
            rel = next(line.split("::", 1)[1].strip() for line in fh if line.startswith("0::"))
        base = "/sys/fs/cgroup" + rel
        headroom = None
        # A limit can be set on any ancestor; the tightest one binds.
        while base.startswith("/sys/fs/cgroup"):
            try:
                limit = open(os.path.join(base, "memory.max")).read().strip()
                if limit != "max":
                    used = int(open(os.path.join(base, "memory.current")).read())
                    # Page cache is charged to the cgroup but reclaimed before
                    # anything is killed; a long run that has read a 3 GB cube
                    # would otherwise see no headroom left at all.
                    stat = dict(line.split() for line in open(os.path.join(base, "memory.stat")))
                    used -= int(stat.get("file", 0)) - int(stat.get("file_dirty", 0))
                    left = max(0, int(limit) - max(used, 0))
                    headroom = left if headroom is None else min(headroom, left)
            except OSError:
                pass
            if base == "/sys/fs/cgroup":
                break
            base = os.path.dirname(base)
        return headroom
    except Exception:                                                 # noqa: BLE001
        return None


def _available_bytes() -> int:
    """Physical memory we may use, read from the OS rather than assumed."""
    avail = 4 << 30
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except Exception:
        pass
    headroom = _cgroup_headroom()
    return min(avail, headroom) if headroom is not None else avail


def _cap_max_side(max_side: int, budget_fraction: float = 0.5, shape=None) -> tuple[int, str | None]:
    """Shrink the working grid so the run cannot exhaust memory.

    An earlier version of this tool asked for a 6144^2 working grid on a 15 GB
    machine and was OOM-killed. The size of an unseen reference is not something
    a caller should have to reason about, so the ceiling is computed here from
    what the OS actually has free, and the reduction is reported rather than
    applied silently.
    """
    budget = _available_bytes() * budget_fraction
    allowed_px = budget / _BYTES_PER_TARGET_PX
    aspect_factor = (max(shape) / math.sqrt(shape[0] * shape[1])) if shape else 1.0
    allowed_side = int(max(512, allowed_px ** 0.5 * aspect_factor))
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
            ny, nx = g.lon.shape
            ry = np.unique(np.r_[np.arange(0, ny, max(1, ny // 64)), ny - 1])
            rx = np.unique(np.r_[np.arange(0, nx, max(1, nx // 16)), nx - 1])
            lon = np.asarray(g.lon, float)[np.ix_(ry, rx)].ravel()
            lat = np.asarray(g.lat, float)[np.ix_(ry, rx)].ravel()
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
    # The lattice is far finer along track than the triangulation needs, but
    # it is thin across track (41 nodes on TMC-2). One stride for both axes -
    # the earlier `[::step, ::step]` - kept TMC-2 nodes 700 px apart and ended
    # at column 3500 of 4000, so the last eighth of the strip fell outside the
    # triangulation. Stride each axis separately and always keep the last
    # row and column.
    ny, nx = grid.lon.shape
    ry = np.unique(np.r_[np.arange(0, ny, max(1, ny // 400)), ny - 1])
    rx = np.unique(np.r_[np.arange(0, nx, max(1, nx // 64)), nx - 1])
    lon = np.asarray(grid.lon, float)[np.ix_(ry, rx)].ravel()
    lat = np.asarray(grid.lat, float)[np.ix_(ry, rx)].ravel()
    SS, LL = np.meshgrid(np.asarray(grid.pixels)[rx], np.asarray(grid.scans)[ry])

    full_lon = np.asarray(grid.lon, float)
    full_lat = np.asarray(grid.lat, float)
    if crs is not None and not crs.is_geographic:
        from pyproj import Transformer
        tr = Transformer.from_crs("+proj=longlat +R=1737400 +no_defs", crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        FX, FY = (np.asarray(v, float).reshape(full_lon.shape)
                  for v in tr.transform(full_lon.ravel(), full_lat.ravel()))

        def prep(a, b):
            return a, b                           # already the reference plane
    else:
        pts = np.column_stack([lon, lat])
        FX, FY = full_lon, full_lat
        # Geographic reference: match the lattice's own longitude convention
        # rather than assume one. Lattices in this archive run 0-360; rasterio
        # hands back -180..180.
        wrap360 = float(lon.max()) > 180.0

        def prep(a, b):
            return ((a % 360.0) if wrap360 else ((a + 180.0) % 360.0 - 180.0)), b

    inverse = _LatticeInverse(FX, FY, np.asarray(grid.pixels, float), np.asarray(grid.scans, float),
                              _Extrapolating(pts, SS.ravel()), _Extrapolating(pts, LL.ravel()))
    out = (inverse.sample, inverse.line, prep)
    if cache is not None:
        cache[key] = out
    return out


class _LatticeInverse:
    """Ground position -> (sample, line), exact on the geometry lattice.

    The lattice is a regular grid in (pixel, scan) carrying a ground position
    per node, so its natural interpolant is bilinear FORWARD: pixel -> ground.
    Delaunay-interpolating the inverse instead fills the convex hull of the
    nodes, and a 740 km strip's edge is not straight: along the TMC-2 edge a
    700 m band of sliver triangles, every vertex on column 3999, mapped all of
    that ground to column 3999 and the source came out as horizontal streaks.
    So the triangulation only seeds the answer; Newton iterations on the
    bilinear forward map then solve it exactly, and a solution outside the
    image (or one that does not converge) is NaN, not an edge column.
    """

    def __init__(self, FX, FY, pixels, scans, guess_s, guess_l):
        self.FX, self.FY = FX, FY
        self.px, self.sc = pixels, scans
        self.gs, self.gl = guess_s, guess_l
        self._query = None
        self._val = None
        steps = np.hypot(np.diff(FX, axis=1), np.diff(FY, axis=1))
        self.tol = 1e-3 * float(np.nanmedian(steps)) if steps.size else 1e-6

    def forward(self, s, l):
        px, sc = self.px, self.sc
        j = np.clip(np.searchsorted(px, s) - 1, 0, len(px) - 2)
        i = np.clip(np.searchsorted(sc, l) - 1, 0, len(sc) - 2)
        dpx, dsc = px[j + 1] - px[j], sc[i + 1] - sc[i]
        u, v = (s - px[j]) / dpx, (l - sc[i]) / dsc
        out, jac = [], []
        for F in (self.FX, self.FY):
            f00, f01, f10, f11 = F[i, j], F[i, j + 1], F[i + 1, j], F[i + 1, j + 1]
            out.append((1 - u) * (1 - v) * f00 + u * (1 - v) * f01 + (1 - u) * v * f10 + u * v * f11)
            jac.append((((1 - v) * (f01 - f00) + v * (f11 - f10)) / dpx,
                        ((1 - u) * (f10 - f00) + u * (f11 - f01)) / dsc))
        return out, jac

    def solve(self, a, b):
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        # Cache both coordinate components for sample()/line(), retaining the
        # values rather than raw pointers which can be reused or mutated.
        if (self._query is not None
                and np.array_equal(a, self._query[0], equal_nan=True)
                and np.array_equal(b, self._query[1], equal_nan=True)):
            return self._val
        s = np.asarray(self.gs(a, b), float).ravel()
        l = np.asarray(self.gl(a, b), float).ravel()
        x, y = a.ravel(), b.ravel()
        live = np.isfinite(s) & np.isfinite(l) & np.isfinite(x) & np.isfinite(y)
        for _ in range(6):
            if not live.any():
                break
            (fx, fy), ((xs, xl), (ys, yl)) = self.forward(s[live], l[live])
            rx, ry = fx - x[live], fy - y[live]
            det = xs * yl - xl * ys
            ok = np.abs(det) > 1e-18
            ds = np.where(ok, (yl * rx - xl * ry) / np.where(ok, det, 1), 0.0)
            dl = np.where(ok, (-ys * rx + xs * ry) / np.where(ok, det, 1), 0.0)
            s[live] -= ds
            l[live] -= dl
        (fx, fy), _ = self.forward(np.nan_to_num(s), np.nan_to_num(l))
        resid = np.hypot(fx - x, fy - y)
        good = (live & (resid <= self.tol)
                & (s >= self.px[0] - 0.5) & (s <= self.px[-1] + 0.5)
                & (l >= self.sc[0] - 0.5) & (l <= self.sc[-1] + 0.5))
        s = np.where(good, s, np.nan).reshape(a.shape)
        l = np.where(good, l, np.nan).reshape(a.shape)
        self._query, self._val = (a.copy(), b.copy()), (s, l)
        return s, l

    def sample(self, a, b):
        return self.solve(a, b)[0]

    def line(self, a, b):
        return self.solve(a, b)[1]


class _Extrapolating:
    """Linear interpolation over the lattice, continued a little beyond it.

    The triangulation ends at the outermost lattice nodes, which are pixel
    CENTRES; the image itself reaches half a pixel further, and a held-out
    point there, or a warp sample at the strip's edge, came back NaN. Queries
    outside the hull but within two node spacings of it are extended by a
    local plane through the nearest nodes; anything further stays NaN.
    """

    def __init__(self, pts, values):
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree
        self.lin = LinearNDInterpolator(pts, values)
        self.pts, self.values = pts, np.asarray(values, float)
        self.tree = cKDTree(pts)
        d, _ = self.tree.query(pts[:: max(1, len(pts) // 2000)], k=2)
        self.reach = 2.0 * float(np.median(d[:, 1]))

    def __call__(self, a, b):
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        out = np.asarray(self.lin(a, b), float)
        miss = ~np.isfinite(out) & np.isfinite(a) & np.isfinite(b)
        if miss.any():
            q = np.column_stack([a[miss], b[miss]])
            dist, idx = self.tree.query(q, k=min(6, len(self.pts)))
            val = np.full(len(q), np.nan)
            near = dist[:, 0] <= self.reach
            for i in np.flatnonzero(near):
                P = self.pts[idx[i]]
                X = np.column_stack([P - P.mean(axis=0), np.ones(len(P))])
                coef, *_ = np.linalg.lstsq(X, self.values[idx[i]], rcond=None)
                val[i] = np.r_[q[i] - P.mean(axis=0), 1.0] @ coef
            out[miss] = val
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
        from .sampling import bilinear
        # The full Jacobian is essential for rotated strips. Using only dS/dx
        # and dL/dy underestimates reduction by cos(rotation), reaching zero
        # at a right angle even when the input spans many pixels per sample.
        stride = 16
        Ss, Ls = S[::stride, ::stride], L[::stride, ::stride]
        factors = [1.0]
        for axis in (0, 1):
            if Ss.shape[axis] > 1:
                distance = np.hypot(np.diff(Ss, axis=axis), np.diff(Ls, axis=axis)) / stride
                finite = distance[np.isfinite(distance)]
                if finite.size:
                    factors.append(float(np.median(finite)))
        f = max(factors)
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
            vals = bilinear(arr, S[ok] / rx - .5, L[ok] / ry - .5)
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

        acc = np.zeros(int(ok.sum()), np.float64)
        cnt = np.zeros(int(ok.sum()), np.float64)
        offs = ((np.arange(k) - (k - 1) / 2.0) * (f / max(k, 1))) if aa else np.zeros(1)
        for dy in offs:
            for dx in offs:
                v = bilinear(src.array, S[ok] - .5 + dx, L[ok] - .5 + dy,
                             nodata=src.nodata, scale=src.meta_scale, offset=src.meta_offset)
                good = np.isfinite(v)
                acc += np.where(good, v, 0.)
                cnt += good
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
             model: str = "auto", max_side: int | None = None, grid=None,
             holdout: float = 0.35, seed: int = 0, profiles: Profiles | None = None,
             segments: int = 0, subpixel: bool = True,
             fine: bool = True, fine_tiles: int = 0, fine_patch: int | None = None, locate: str = "auto",
             progress=None, verbose: bool = True, ground_truth: str | None = None) -> Result:
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
    # ---- 1. read -----------------------------------------------------------
    try:
        S = load(source, profiles)
        R = load(reference, profiles)
    except UnreadableInput as exc:
        return _fail(out_dir, job_id, "unreadable_input", str(exc))
    log("source    : %s" % json.dumps(S.summary()))
    log("reference : %s" % json.dumps(R.summary()))

    sp, rp = profiles.get(S.profile), profiles.get(R.profile)
    defaults = sp.get("registration", {})
    max_side = int(max_side if max_side is not None else defaults.get("max_side", 2048))
    grid = tuple(grid) if grid is not None else tuple(defaults.get("grid", [8, 8]))
    # A long narrow strip needs enough cross-track pixels to match. Its memory
    # cost follows area, not the square of its longest dimension.
    window = _source_window(S, R)
    shape = (window[2] - window[0], window[3] - window[1]) if window else R.array.shape[:2]
    max_side, cap_note = _cap_max_side(max_side, shape=shape)
    if cap_note:
        log("memory   : %s" % cap_note)
    log("settings  : max_side=%d, grid=%s (explicit options override sensor defaults)" % (max_side, grid))
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

    # Cells are laid out square whatever the frame's shape: an 8 x 8 grid over a
    # 250 x 17 strip is cells two pixels wide, which neither measures coverage
    # nor keeps held-out points away from fit points.
    grid = _square_cells(grid, B.shape)
    # Held-out cells follow the overlap - its principal axes and extent - and
    # are fixed here, from the placement alone, before any matching.
    split_frame = _overlap_frame(both)
    split_extent = (B.shape if split_frame is None else
                    (split_frame["extent"][3] - split_frame["extent"][1],
                     split_frame["extent"][2] - split_frame["extent"][0]))
    split = E.SpatialSplit(B.shape, holdout, seed, frame=split_frame,
                           grid=_square_cells((8, 8), split_extent))
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
    # A shared map frame (or a located placement) leaves only a residual
    # correction, which cannot be a large rotation or scale change. Without
    # one, any rotation is allowed, but the model must still be a physically
    # possible mapping between two images of the same ground.
    constrained = (((S.georeferenced or S.lonlat is not None) and R.georeferenced)
                   or frame.get("placement") is not None)
    # The source region every candidate model will be applied to.
    _sy, _sx = np.nonzero(both)
    src_extent = (_sx.min(), _sy.min(), _sx.max(), _sy.max())
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
            # Where a wrong match can land: anywhere in the overlap for a global
            # matcher, inside the local search box for the dense tie points.
            area = (float((2 * _DENSE_SEARCH + 1) ** 2) if name == "dense-ncc"
                    else float(both.sum()))
            nfa = V.log10_nfa(len(corr), int(vr.n_inliers), mt, 3.0, area)
            rec.update({"model": mt, "inliers": int(vr.n_inliers),
                        "inlier_ratio": round(vr.inlier_ratio, 4),
                        "verified": bool(vr.ok),
                        "log10_nfa": round(nfa, 2) if np.isfinite(nfa) else None})
            why = None
            if vr.ok:
                why = V.degenerate(vr.model, corr.src[vr.inlier_mask], extent=src_extent)
                dd = V.decompose(vr.model) if vr.model is not None else {}
                if why is None and constrained and (
                        abs(dd.get("est_scale_x", 1.0) - 1.0) > 0.25
                        or abs(dd.get("est_rotation_deg", 0.0)) > 30.0):
                    why = ("scale %.3f, rotation %.1f deg is not a residual correction "
                           "of a shared map frame" % (dd.get("est_scale_x", float("nan")),
                                                       dd.get("est_rotation_deg", float("nan"))))
                if why is None and not nfa < 0.0:
                    why = ("consensus is not significant: %d of %d agree, NFA 10^%.1f"
                           % (vr.n_inliers, len(corr), nfa))
                if why:
                    rec["rejected"] = why
            if vr.ok and vr.n_inliers >= 6 and why is None:
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
        # Consensus was found but only for models no two images of the same
        # ground can have: that is a degenerate transform, not a lack of matches.
        if any(a.get("inliers", 0) >= 6 and a.get("rejected")
               and not a["rejected"].startswith("consensus") for a in attempts):
            code = "degenerate_transform"
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
    # Every candidate was screened in the loop above; this is the same test on
    # the model actually carried forward, kept as a rail.
    why = V.degenerate(Hm, best["corr"].src[best["vr"].inlier_mask], extent=src_extent)
    if why is None and constrained and (abs(d.get("est_scale_x", 1.) - 1.) > .25
                                        or abs(d.get("est_rotation_deg", 0.)) > 30.):
        why = "not a residual correction of a shared map frame"
    if why:
        return _fail(out_dir, job_id, "degenerate_transform",
                     "the fitted transform violates the available geometry constraints "
                     "(%s): scale %.3f, rotation %.2f deg"
                     % (why, d.get("est_scale_x", float("nan")),
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
    H_coarse = np.asarray(Hm, float).copy()
    Cw0 = grid_to_reference(frame)
    fine_info = {"attempted": False}
    per_point_back = None
    field = None
    parallax = None
    tiepoints = None
    if fine:
        fine_plan = [best["name"]] + [p for p in plan if p != best["name"]]
        support = cv2.warpPerspective(Am.astype(np.uint8), Hm, (B.shape[1], B.shape[0]),
                                      flags=cv2.INTER_NEAREST).astype(bool) & Bm
        fcorr, fine_info = _fine_stage(S, R, corr, vr, frame, fine_plan, max_side,
                                       grid, log, cache, Hcoarse=Hm,
                                       tiles=fine_tiles, split=split, support=support,
                                       patch=fine_patch if fine_patch is not None else defaults.get("fine_patch"))
        if fcorr is not None:
            # Back onto the working grid, where every downstream stage already
            # lives. The positions are now measured at native resolution, so
            # these are sub-working-pixel by construction.
            Ci0 = np.linalg.inv(Cw0)
            whole_fine = M.Correspondences(project(Ci0, fcorr.src), project(Ci0, fcorr.ref),
                                          fcorr.confidence, fcorr.method, fcorr.kind,
                                          dict(fcorr.detail))
            fit_fine, val_fine, test_fine = E.partition(whole_fine, split)
            tiepoints = whole_fine
            heights = None
            tspec = None
            try:
                tspec = terrain.spec_for(R, frame)
            except Exception as exc:                                  # noqa: BLE001
                log("terrain   : DEM unavailable (%s)" % exc)
            if tspec is not None:
                heights = (tspec, terrain.sampler(tspec))
                fine_info["dem"] = os.path.basename(tspec["dem"])
            fvr, ffield, dense_info = _fit_dense(fit_fine, best["model"], B.shape, grid, split,
                                                 tight=3.0 / step0, log=log, heights=heights)
            fine_info["model_fit"] = dense_info

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
            verified = (fvr.model is not None and fvr.ok
                        and fvr.n_inliers >= max(8, V._min_points(best["model"])))
            fine_gates = _gates(fvr, fit_fine.ref) if verified else {}
            regressed = [k for k in fine_gates if k not in _gates(vr, corr.ref)]
            if verified and not regressed:
                corr = fit_fine
                validation, sealed_test = val_fine, test_fine
                vr = fvr
                Hm = fvr.model
                field = ffield
                parallax = getattr(fvr, "parallax", None)
                d = V.decompose(Hm)
                per_point_back = (corr.detail["back_x"], corr.detail["back_y"])
                fine_info["adopted"] = True
                fine_info["inliers"] = int(fvr.n_inliers)
                log("fine      : %d points from %d native tiles, %d fit points, %d inliers"
                    % (fine_info["points"], fine_info["windows_used"], len(fit_fine),
                       fvr.n_inliers))
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

    # The native model above is already a cell-balanced fit; a coarse one is
    # not. Equalize spatial influence without discarding verified evidence: a
    # fit from thousands of points on one crater must not overwhelm a quiet cell.
    if not fine_info["adopted"] and vr.n_inliers >= 12:
        Hb = balanced_refit(corr.src[vr.inlier_mask], corr.ref[vr.inlier_mask],
                            best["model"], B.shape, grid, Hm)
        if Hb is not None and len(validation) >= 3:
            old = np.mean(V.transfer_error(Hm, validation.src, validation.ref) ** 2)
            new = np.mean(V.transfer_error(Hb, validation.src, validation.ref) ** 2)
            if new <= old:
                Hm = Hb
                vr.model = Hm
                fine_info["spatially_balanced_refit"] = True
                d = V.decompose(Hm)

    # ---- 6. ECC: optimize only fit pixels, adopt only on validation ---------
    # The test fold remains sealed; no metric from it is used for any decision.
    # ECC works on the decimated working grid, so it can refine a coarse model
    # but not one measured at native resolution.
    ecc = {"attempted": False, "adopted": False}
    if fine_info["adopted"]:
        ecc["note"] = "not run: the model was measured at native resolution"
    elif subpixel and len(validation) >= 3:
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
                vr.model = Hm
                d = V.decompose(Hm)
                log("subpixel : ECC adopted on validation %.4f -> %.4f px" % (old, new))
            else:
                ecc["note"] = "ECC did not improve the validation fold; fit model retained"
    else:
        ecc["note"] = "ECC disabled or too few validation points"

    # ---- 7. write artifacts ------------------------------------------------
    os.makedirs(job_dir, exist_ok=True)
    m_per_px = _metres_per_pixel(R, frame)
    # A cross-validated field already follows along-track change; stacking
    # per-segment matrices under it would fit the same variation twice.
    nonlinear = field is not None or parallax is not None
    seg = None if nonlinear else _segment_fit(corr, vr, best["model"], B.shape, segments)
    if nonlinear and segments and segments >= 2:
        fine_info["segments_note"] = ("segments not fitted: the local field and terrain term "
                                      "model along-track change")

    conv = _unit_conversions(S, R, frame)
    delivered = _write_matches(job_dir, corr, vr, back_x, back_y, frame, per_point_back,
                               shape=B.shape, grid=grid)
    if tiepoints is not None and fine_info.get("adopted"):
        # Every native measurement with its fold, for audit: the working-grid
        # positions the model was fitted and scored on, and the source pixels.
        np.savez_compressed(os.path.join(job_dir, "tiepoints.npz"),
                            src_working=tiepoints.src, ref_working=tiepoints.ref,
                            fold=split.labels(tiepoints.src),
                            source_px=np.column_stack([tiepoints.detail["back_x"],
                                                       tiepoints.detail["back_y"]]),
                            confidence=tiepoints.confidence)
    tj = _write_transform(job_dir, Hm, best["model"], d, seg, frame, best["name"], conv,
                          field=field, parallax=parallax)
    # JSON round-trip is the single model used by both raster export and scoring.
    with open(os.path.join(job_dir, "transform.json")) as fh:
        tj = json.load(fh)
    Hm = np.asarray(tj["matrix"], np.float64)
    warped, wvalid = (WM.warp(A_raw, Am, tj, B_raw.shape) if (seg or nonlinear) else
                       _warp(A_raw, Am, Hm, B_raw.shape, best["model"]))
    # Only where the registered source lies, not the whole search window: an
    # IIRS strip written over its padded WAC window was 256 bands of mostly
    # empty 10653 x 471 pixels.
    footprint = None
    if wvalid.any():
        fy, fx = np.nonzero(wvalid)
        corners = project(grid_to_reference(frame),
                          [[fx.min() - 1.0, fy.min() - 1.0], [fx.max() + 1.0, fy.max() + 1.0]])
        footprint = (int(np.floor(corners[0, 1])), int(np.floor(corners[0, 0])),
                     int(np.ceil(corners[1, 1])) + 1, int(np.ceil(corners[1, 0])) + 1)
    export_info = write_registered_native(job_dir, S, R, frame, tj,
                         lattice_interpolators=_lattice_interpolators, cache=cache,
                         window=footprint)
    def working_to_source(points):
        return reference_to_source(S, R, frame, project(grid_to_reference(frame), points),
                                   lattice_interpolators=_lattice_interpolators, cache=cache)
    # Check points: held-out correspondences that move with their held-out
    # neighbours relative to the COARSE model. Neither the exported model nor
    # any fit point takes part, so a model error (which moves a neighbourhood)
    # is kept and only an isolated mismatch is set aside - and counted.
    tight = 3.0 / step0
    if len(sealed_test):
        check = _consistent(np.asarray(sealed_test.src, np.float64),
                            np.asarray(sealed_test.ref, np.float64)
                            - project(H_coarse, sealed_test.src), floor=tight)
    else:
        check = np.zeros(0, bool)
    rule = ("held-out correspondence within max(3 reference px, 3 robust sigma) of the "
            "median displacement of its 10 nearest held-out neighbours, displacements "
            "taken against the coarse model; the exported model is not consulted")
    acc = E.score_export(job_dir, sealed_test, split, working_to_source=working_to_source,
                         check_points=check, check_point_rule=rule)
    independent = None
    if ground_truth is not None:
        try:
            independent = E.score_ground_truth(job_dir, ground_truth,
                working_to_source=working_to_source, source_path=source, reference_path=reference,
                source_shape=S.shape, reference_shape=R.shape)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            independent = {"passed": False, "reasons": ["ground truth rejected: " + str(exc)]}
        acc["independent_ground_truth"] = independent
    acc.update(subpixel_method="parabolic+ecc" if ecc.get("adopted") else "matcher", ecc=ecc)
    if acc["held_out_rmse_px"] is None:
        degraded.append("fewer than three held-out matches; accuracy is unknown, no fit-set fallback")
    eligible = spatial.eligible_cells(wvalid & Bm, B.shape, grid)
    character["registered_overlap_fraction"] = float((wvalid & Bm).mean())
    ov_cov = spatial.cell_coverage(corr.ref[vr.inlier_mask], B.shape, grid,
                                   eligible=eligible)
    ov_disp = spatial.dispersion(corr.ref[vr.inlier_mask], B.shape)
    ov_extrap = spatial.extrapolation_fraction(corr.ref[vr.inlier_mask], B.shape,
                                               grid, eligible=eligible)
    metrics = _write_metrics(job_dir, job_id, S, R, character, frame, attempts, best,
                             vr, acc, ov_cov, ov_disp, m_per_px, degraded, warn, d,
                             time.time() - t_start, grid, eligible, ov_extrap,
                             conv, fine_info)
    metrics["distribution"]["delivered"] = delivered
    metrics["export"] = export_info
    with open(os.path.join(job_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
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
            src_per_ref = 1.0 / float(str(frame["route"]).split("GSD ratio ")[1].split(",")[0])
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


# Native-resolution tie points. Seeds sit on a lattice over the whole overlap,
# not wherever a detector fired, so the delivered points are uniform by
# construction; each is measured by local NCC + ECC with a forward-backward
# check (`methods.refine_correspondences`).
_FINE_SEEDS = 6000          # lattice seeds aimed for over the overlap
_FINE_PATCH = 41            # correlation patch, native reference px
_FINE_TILE = 1024           # native reference px placed per window
_FINE_MAX_WINDOWS = 400


def _prewarp_from(model, frame):
    """Native reference px -> prealigned source position, from a working-grid model."""
    C = grid_to_reference(frame)
    Hw = np.asarray(model, float)
    if Hw.shape != (3, 3):
        Hw = np.vstack([Hw[:2], [0.0, 0.0, 1.0]])
    return np.linalg.inv(C @ Hw @ np.linalg.inv(C))


def _seed_lattice(support, frame, target):
    """Evenly spaced native reference seeds over the working-grid `support` mask.

    The spacing follows the overlap's area, so a small overlap is sampled as
    densely as a large one is thinly - but never closer than half a patch,
    where neighbouring measurements would share most of their pixels.
    """
    C = grid_to_reference(frame)
    step = float(C[0, 0])
    area = float(support.sum()) * step * step
    spacing = max(_FINE_PATCH / 2.0, math.sqrt(area / max(int(target), 1)))
    ys, xs = np.nonzero(support)
    lo = project(C, [[xs.min() - 0.5, ys.min() - 0.5]])[0]
    hi = project(C, [[xs.max() + 0.5, ys.max() + 0.5]])[0]
    gx = np.arange(lo[0] + spacing / 2.0, hi[0], spacing)
    gy = np.arange(lo[1] + spacing / 2.0, hi[1], spacing)
    X, Y = np.meshgrid(gx, gy)
    lattice = np.column_stack([X.ravel(), Y.ravel()])
    if not len(lattice):
        return lattice, spacing
    wk = np.rint(project(np.linalg.inv(C), lattice)).astype(np.int64)
    h, w = support.shape
    inside = (wk[:, 0] >= 0) & (wk[:, 0] < w) & (wk[:, 1] >= 0) & (wk[:, 1] < h)
    keep = np.zeros(len(lattice), bool)
    keep[inside] = support[wk[inside, 1], wk[inside, 0]]
    return lattice[keep], spacing


def _measure_seeds(A, Am, B, Bm, seeds, search, patch=None):
    """Measure window-local seeds on a prewarped pair; source seeds never move."""
    seeds = np.asarray(seeds, np.float64).reshape(-1, 2)
    if not len(seeds):
        return None
    corr = M.Correspondences(seeds, seeds.copy(), np.ones(len(seeds), np.float32),
                             "grid-refined", "dense")
    return M.refine_correspondences(corr, A, Am, B, Bm,
                                     patch=_FINE_PATCH if patch is None else patch, search=search)


def _select_fine_patch(candidates, split, frame, default=41):
    """Compare patches on common probe seeds, cross-validating whole FIT cells.

    Validation/test coordinates and their measurement success never enter the
    score. An alternative needs a 5% median improvement without a worse tail
    or a loss of more than 20% of the default's fit measurements.
    """
    table = []
    if split is None or default not in candidates:
        return default, [{"note": "no fit partition or default measurements"}]
    ci = np.linalg.inv(grid_to_reference(frame))
    fit = {}
    for size, corr in candidates.items():
        keep = split.labels(project(ci, corr.src)) == split.FIT
        fit[size] = {tuple(p): q for p, q in zip(corr.src[keep], corr.ref[keep])}
    common = set(fit[default])
    for points in fit.values():
        common.intersection_update(points)
    if len(common) < 30:
        return default, [{"note": "fewer than 30 common fit measurements", "common_n": len(common)}]
    src = np.asarray(sorted(common), np.float64)
    cells = split.cells(project(ci, src))
    unique, counts = np.unique(cells, return_counts=True)
    if len(unique) < 4:
        return default, [{"note": "fewer than four common fit cells", "cells": len(unique)}]
    weights = 1. / counts[np.searchsorted(unique, cells)]
    order = np.random.default_rng(0).permutation(unique)
    folds = [order[i::min(5, len(unique))] for i in range(min(5, len(unique)))]
    for size, points in fit.items():
        ref = np.asarray([points[tuple(p)] for p in src])
        errors = []
        for group in folds:
            held = np.isin(cells, group)
            B, _ = _inverse_fit(ref[~held], src[~held], None, weights[~held])
            pred = np.column_stack([ref[held], np.ones(held.sum())]) @ B.T
            errors.extend(np.linalg.norm(pred - src[held], axis=1))
        table.append({"patch_px": size, "fit_n": len(points), "common_n": len(src),
                      "fit_cells": len(unique), "cv_median": float(np.median(errors)),
                      "cv_p90": float(np.percentile(errors, 90)),
                      "measurement_fraction": len(points) / max(1, len(fit[default]))})
    best = next(row for row in table if row["patch_px"] == default)
    for row in table:
        if (row["measurement_fraction"] >= .8 and row["cv_median"] < .95 * best["cv_median"]
                and row["cv_p90"] <= 1.05 * best["cv_p90"]):
            best = row
    return best["patch_px"], table


def _fine_stage(S: Scene, R: Scene, corr, vr, frame, plan, max_side: int,
                grid, log, cache, Hcoarse=None, tiles: int = 0, split=None, support=None,
                seeds: int = _FINE_SEEDS, patch=None):
    """Measure tie points at NATIVE reference resolution over the whole overlap.

    The coarse solve runs on a decimated working grid, so the residual it can
    report is floored by that decimation, not by the method. A TMC-2 strip
    against a 3-degree SELENE tile decimates 6x: every reported pixel is 44 m
    wide, and "1.3 px" is 59 m and 10.8 SOURCE pixels. No sub-pixel claim
    survives that, whatever the internal flag says.

    Seeds are laid on a lattice over the overlap the coarse model predicts
    (`support`), so the result covers it uniformly by construction instead of
    clustering where a detector finds texture. The lattice is walked in tiles:
    each tile is placed at native resolution, prewarped by the coarse model so
    only its residual is left to find, and padded by the patch and search
    radius so seeds at a tile's edge are measured like any other. Peak memory
    is one tile. A tile that cannot be placed at native resolution is counted,
    and the result is then not called native.

    The measurement is local NCC refined by ECC, with a forward-backward check
    (`methods.refine_correspondences`). When that yields too little on the
    first tiles - it needs correlated texture - the sparse matchers of `plan`
    are probed there instead, and the first that verifies is used throughout.

    Returns ``(Correspondences in FULL-RESOLUTION reference pixels, info)``, with
    `back_x`/`back_y` carrying the original source pixel of each point. The
    source side is the prealigned position (the prewarp undone), so a fit on
    these gives the full transform, not the residual to the coarse one.
    """
    info = {"attempted": True, "windows": 0, "windows_used": 0, "points": 0,
            "native": False, "note": None, "method": None, "probe": []}
    patches = [int(patch)] if patch is not None else sorted({_FINE_PATCH, 61})
    if any(p < 9 or p % 2 != 1 for p in patches):
        raise ValueError("fine_patch must be an odd integer of at least 9")
    selected_patch = patches[0]
    step = int(frame.get("reference_decimation", 1) or 1)
    prewarp = None
    if Hcoarse is not None:
        try:
            prewarp = _prewarp_from(Hcoarse, frame)
        except np.linalg.LinAlgError:
            prewarp = None
    if support is None or not support.any():
        idx = np.nonzero(vr.inlier_mask)[0]
        if len(idx) < 4:
            info.update(attempted=False, note="too few coarse inliers to place windows on")
            return None, info
        h, w = frame["target_shape"]
        support = np.zeros((h, w), bool)
        x0, y0 = np.floor(corr.ref[idx].min(axis=0)).astype(int)
        x1, y1 = np.ceil(corr.ref[idx].max(axis=0)).astype(int)
        support[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
    lattice, spacing = _seed_lattice(support, frame, seeds)
    if len(lattice) < 8:
        info.update(attempted=False, note="the overlap holds only %d seed positions" % len(lattice))
        return None, info

    # With the coarse solve applied, what is left is its residual: a working
    # pixel or two, more where the coarse model extrapolates.
    search = int(np.clip(3 * step, 8, 32)) if prewarp is not None else int(np.clip(4 * step, 16, 48))
    pad = max(patches) // 2 + search + 3
    core = float(max(64, min(_FINE_TILE, max_side) - 2 * pad))
    origin = lattice.min(axis=0)
    key = np.floor((lattice - origin) / core).astype(np.int64)
    uniq = sorted({(int(a), int(b)) for a, b in key}, key=lambda t: (t[1], t[0]))
    groups = []
    for kx, ky in uniq:
        members = np.flatnonzero((key[:, 0] == kx) & (key[:, 1] == ky))
        lo = origin + np.array([kx, ky], float) * core
        groups.append((members, lo, lo + core))
    cap = int(tiles) if tiles and tiles > 0 else _FINE_MAX_WINDOWS
    if len(groups) > cap:
        keep = np.unique(np.linspace(0, len(groups) - 1, cap).round().astype(int))
        info["note"] = ("%d of %d tiles measured (window budget); the rest of the overlap "
                        "is interpolated" % (len(keep), len(groups)))
        groups = [groups[i] for i in keep]
    info.update(windows=len(groups), seeds=int(len(lattice)), seed_spacing_px=round(spacing, 2),
                search_px=search, patch_px=selected_patch,
                placement="%d tiles over %d lattice seeds at %.1f px spacing"
                          % (len(groups), len(lattice), spacing))

    H, W = R.array.shape

    def place(g, pw):
        members, _, _ = g
        pts = lattice[members]
        r0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
        c0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
        r1 = int(min(H, np.ceil(pts[:, 1].max()) + pad + 1))
        c1 = int(min(W, np.ceil(pts[:, 0].max()) + pad + 1))
        try:
            return prealign(S, R, max_side=max(max_side, r1 - r0, c1 - c0),
                            window=(r0, c0, r1, c1), cache=cache, prewarp=pw,
                            placement=frame.get("placement"))
        except Exception as exc:                                      # noqa: BLE001
            log("fine      : tile at (%d, %d) could not be placed: %s" % (c0, r0, exc))
            return None

    counts = {}

    def measure(name, g, placed, pw, patch_size=None):
        members, lo, hi = g
        A, Am, B, Bm, bx, by, fr = placed
        Cw = grid_to_reference(fr)
        Ci = np.linalg.inv(Cw)
        if name == "grid-refined":
            c = _measure_seeds(A, Am, B, Bm, project(Ci, lattice[members]), search,
                                patch=selected_patch if patch_size is None else patch_size)
        else:
            try:
                c, _ = _run_one(name, A, Am, B, Bm, search, grid)
            except Exception:                                         # noqa: BLE001
                c = None
            if c is None or not len(c):
                return None
            # Only keypoints inside this tile's own square, so overlapping
            # windows never report the same ground twice.
            nat = project(Cw, c.src)
            c = E.subset(c, np.all((nat >= lo) & (nat < hi), axis=1))
            if not len(c):
                return None
            sel = spatial.select(c.src, c.confidence, A.shape, grid=grid, per_cell=8,
                                 min_keep=0, min_for_thinning=0, min_separation=3.).keep
            c = M.refine_correspondences(E.subset(c, sel), A, Am, B, Bm,
                                         search=min(12, search))
        if c is None or not len(c):
            return None
        for k, v in (c.detail.get("refinement_counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
        src = project(Cw, c.src)
        if pw is not None:
            src = project(pw, src)
        back = sample_backmap(bx, by, c.src)
        return (src, project(Cw, c.ref), np.asarray(c.confidence, np.float32), back,
                int(fr["reference_decimation"]))

    def fit_fold(src_native):
        if split is None:
            return np.ones(len(src_native), bool)
        working = project(np.linalg.inv(grid_to_reference(frame)), src_native)
        return split.labels(working) == split.FIT

    # Probe on the first few usable tiles: the lattice measurement when it
    # verifies there, else the first sparse matcher that does.
    names = ["grid-refined"] + [p for p in (plan if isinstance(plan, (list, tuple)) else [plan])
                                if p not in ("grid-refined", "dense-ncc")]
    # Probe where the fit fold is: a tile lying wholly in held-out cells can
    # offer the probe nothing, whatever it measures. Rank tiles by the fit
    # seeds they hold (their source side is roughly the prewarped seed).
    seed_fit = fit_fold(project(prewarp, lattice) if prewarp is not None else lattice)
    ranked = sorted(range(len(groups)), key=lambda i: -int(seed_fit[groups[i][0]].sum()))
    probe_tiles = []
    for i in ranked:
        g = groups[i]
        if not seed_fit[g[0]].any():
            break
        placed = place(g, prewarp)
        if placed is None or (placed[1] & placed[3]).sum() < 1024:
            continue
        probe_tiles.append((i, placed))
        if len(probe_tiles) >= 3:
            break
    if not probe_tiles:
        info["note"] = "no native tile overlapped usable data"
        return None, info
    patch_results = {}
    if len(patches) > 1:
        candidates = {}
        for size in patches:
            got = {i: measure("grid-refined", groups[i], placed, prewarp, size)
                   for i, placed in probe_tiles}
            got = {i: r for i, r in got.items() if r is not None}
            patch_results[size] = got
            if got:
                candidates[size] = M.Correspondences(
                    np.vstack([r[0] for r in got.values()]), np.vstack([r[1] for r in got.values()]),
                    np.concatenate([r[2] for r in got.values()]), "grid-refined", "dense")
        selected_patch, table = _select_fine_patch(candidates, split, frame, patches[0])
        info.update(patch_px=selected_patch, patch_cv=table,
                    patch_selection="spatial cross-validation on common fit-fold probe seeds only")
    else:
        info["patch_selection"] = "explicit or sensor-profile setting"
    method, results = None, {}
    for name in names:
        got = (patch_results[selected_patch] if name == "grid-refined" and patch_results else
               {i: measure(name, groups[i], placed, prewarp) for i, placed in probe_tiles})
        got = {i: r for i, r in got.items() if r is not None}
        n_i = n_in = 0
        if got:
            src = np.vstack([r[0] for r in got.values()])
            ref = np.vstack([r[1] for r in got.values()])
            ff = fit_fold(src)
            n_i = int(ff.sum())
            if n_i >= 4:
                n_in = int(V.verify(src[ff], ref[ff], model_type="affine", threshold=3.0).n_inliers)
        info["probe"].append({"method": name, "candidates": n_i, "inliers": n_in})
        if n_in >= 8:
            method, results = name, got
            break
    if method is None:
        info["note"] = "no method verified on the first native tiles"
        return None, info
    info["method"] = method
    log("fine      : native probe -> %s  (%s)" % (method, ", ".join(
        "%s %d/%d" % (x["method"], x["inliers"], x["candidates"]) for x in info["probe"])))
    del probe_tiles, patch_results

    # Tiles far from the coarse tie points may lie beyond the search radius of
    # the coarse prewarp: a model from one cluster of matches extrapolates
    # badly down a long strip. So grow outwards - refit on the fit-fold points
    # measured so far and retry the tiles that came back (nearly) empty with
    # that better prewarp. Only fit-fold points steer the prewarp.
    done = {i: r for i, r in results.items()}
    for i, g in enumerate(groups):
        if i in done:
            continue
        placed = place(g, prewarp)
        done[i] = measure(method, g, placed, prewarp) if placed is not None else None
    passes = [{"prewarp": "coarse model", "tiles_with_points": 0}]

    def enough(i):
        r = done.get(i)
        return (r is not None and fit_fold(r[0]).sum()
                >= max(3, 0.2 * seed_fit[groups[i][0]].sum()))

    for attempt in range(2):
        passes[-1]["tiles_with_points"] = int(sum(enough(i) for i in range(len(groups))))
        weak = [i for i in range(len(groups)) if not enough(i)]
        good = [done[i] for i in range(len(groups)) if enough(i)]
        if not weak or not good or method != "grid-refined":
            break
        src = np.vstack([r[0] for r in good])
        ref = np.vstack([r[1] for r in good])
        ff = fit_fold(src)
        if ff.sum() < 20:
            break
        fit = V.verify(src[ff], ref[ff], model_type="affine", threshold=3.0)
        if fit.model is None or fit.n_inliers < 20:
            break
        try:
            grown = np.linalg.inv(fit.model)
        except np.linalg.LinAlgError:
            break
        before = passes[-1]["tiles_with_points"]
        passes.append({"prewarp": "affine from %d fit-fold points" % fit.n_inliers,
                       "retried": len(weak), "tiles_with_points": 0})
        for i in weak:
            placed = place(groups[i], grown)
            r = measure(method, groups[i], placed, grown) if placed is not None else None
            # Only fit-fold counts choose a retry. Tiles with no fit seeds
            # follow the latest fit-derived prewarp without comparing tests.
            if r is not None and (not seed_fit[groups[i][0]].any() or done.get(i) is None
                                  or fit_fold(r[0]).sum() > fit_fold(done[i][0]).sum()):
                done[i] = r
        passes[-1]["tiles_with_points"] = int(sum(enough(i) for i in range(len(groups))))
        if passes[-1]["tiles_with_points"] <= before:
            break
    info["passes"] = passes

    decimated = 0
    parts = []
    for i in range(len(groups)):
        r = done.get(i)
        if r is None:
            continue
        parts.append(r)
        decimated += int(r[4] != 1)
        info["windows_used"] += 1
    if not parts:
        info["note"] = info["note"] or "no native tile produced a tie point"
        return None, info
    back = np.vstack([p[3] for p in parts])
    out = M.Correspondences(
        src=np.vstack([p[0] for p in parts]).astype(np.float64),
        ref=np.vstack([p[1] for p in parts]).astype(np.float64),
        confidence=np.concatenate([p[2] for p in parts]),
        method=method + "@native", kind="dense",
        detail={"stage": "fine", "windows": info["windows_used"],
                "back_x": back[:, 0], "back_y": back[:, 1]})
    info["points"] = len(out)
    info["refinement_counts"] = counts
    info["native"] = decimated == 0
    if decimated:
        info["note"] = ("%d of %d tiles could only be placed decimated; raise --max-side"
                        % (decimated, len(parts)))
    return out, info


def _consistent(src, v, floor, k=10):
    """Model-free screen: a correspondence must move like its neighbours.

    `v` is each point's displacement from a rough global model. An isolated
    mismatch disagrees with the median of its k nearest neighbours; a smooth
    distortion the global model misses does not. Judged within the set given,
    so fit points are never screened by held-out ones.
    """
    n = len(src)
    if n < 12:
        return np.ones(n, bool)
    from scipy.spatial import cKDTree
    k = int(min(k, n - 1))
    _, nb = cKDTree(src).query(src, k=k + 1)
    nb = nb[:, 1:]
    med = np.median(v[nb], axis=1)
    dev = np.linalg.norm(v - med, axis=1)
    spread = np.median(np.linalg.norm(v[nb] - med[:, None, :], axis=2), axis=1)
    return dev <= np.maximum(floor, 3.0 * 1.4826 * spread)


def _inverse_fit(q, p, h, weights=None, iterations=8):
    """Robust inverse model  p = B [q, 1] (+ m h)  on reference points q.

    `h` is None for the plain affine. Huber IRLS on top of `weights`.
    Returns (B 2x3, m or None).
    """
    q = np.asarray(q, np.float64)
    p = np.asarray(p, np.float64)
    cols = [q[:, 0], q[:, 1], np.ones(len(q))] + ([h] if h is not None else [])
    X = np.column_stack(cols)
    base = np.ones(len(q)) if weights is None else np.asarray(weights, np.float64)
    robust = np.ones(len(q))
    coef = np.zeros((X.shape[1], 2))
    for _ in range(iterations):
        w = np.sqrt(base * robust)
        coef = np.linalg.lstsq(X * w[:, None], p * w[:, None], rcond=None)[0]
        e = np.linalg.norm(X @ coef - p, axis=1)
        sigma = max(0.02, 1.4826 * float(np.median(np.abs(e - np.median(e)))))
        robust = np.minimum(1.0, 1.345 * sigma / np.maximum(e, 1e-12))
    B = coef[:3].T
    return B, (coef[3] if h is not None else None)


def _cv_height(q, p, h, cells, weights, folds=5, seed=0):
    """Cross-validated median error of the inverse affine with and without height."""
    unique = np.unique(cells)
    if len(unique) < 4:
        return None
    order = np.random.default_rng(seed).permutation(unique)
    groups = [order[i::min(folds, len(unique))] for i in range(min(folds, len(unique)))]
    out = {}
    for label, hh in (("global", None), ("global+height", h)):
        errs = []
        for g in groups:
            held = np.isin(cells, g)
            if held.all() or not held.any():
                continue
            B, m = _inverse_fit(q[~held], p[~held], None if hh is None else hh[~held],
                                weights[~held])
            pred = np.column_stack([q[held], np.ones(held.sum())]) @ B.T
            if m is not None:
                pred += hh[held][:, None] * m[None, :]
            errs.append(np.linalg.norm(pred - p[held], axis=1))
        e = np.concatenate(errs) if errs else np.array([np.inf])
        out[label] = {"cv_median": float(np.median(e)), "cv_p90": float(np.percentile(e, 90))}
    return out


def _fit_dense(corr, model_type, shape, grid, split, tight, allow_field=True, log=None,
               heights=None):
    """Global model plus an optional smooth local field, from dense fit points.

    1. A robust global model with a loose threshold removes gross mismatches.
    2. `_consistent` removes points that do not move with their neighbours.
    3. A cell-balanced least-squares global model is fitted to what is left.
    4. A B-spline correction field is chosen by spatial cross-validation over
       whole fit cells (`local_model.cross_validate`) - or none, when it does
       not predict unseen cells better.
    5. Inliers are the points within `tight` of the complete model.

    Returns ``(VerifyResult, field or None, info)``. Only fit-fold points enter.
    """
    from . import local_model as LM
    src = np.asarray(corr.src, np.float64)
    ref = np.asarray(corr.ref, np.float64)
    n = len(src)
    info = {"points": n, "tight_threshold_px": round(float(tight), 4)}
    need = max(8, V._min_points(model_type) + 2)
    loose = max(4.0 * tight, tight + 3.0)
    vr0 = V.verify(src, ref, model_type=model_type, threshold=loose)
    if vr0.model is None or vr0.n_inliers < need:
        info["note"] = "no global model at the loose threshold"
        return vr0, None, info
    v = ref - project(vr0.model, src)
    ok = vr0.inlier_mask & _consistent(src, v, floor=tight)
    info.update(loose_threshold_px=round(float(loose), 4), loose_inliers=int(vr0.n_inliers),
                consistent=int(ok.sum()))
    if ok.sum() < need:
        ok = vr0.inlier_mask
    H = balanced_refit(src[ok], ref[ok], model_type, shape, grid, vr0.model)
    if H is None or not np.isfinite(H).all():
        H = vr0.model
    # Terrain parallax, when a DEM covers the overlap: one height coefficient,
    # fitted jointly with the inverse affine and kept only when it predicts
    # whole unseen fit cells better (see `terrain`).
    parallax = None
    if heights is not None and split is not None and model_type == "affine":
        spec, sample = heights
        hq = sample(ref)
        use = ok & np.isfinite(hq)
        if use.sum() >= 30 and np.nanstd(hq[use]) > 1.0:
            origin = float(np.median(hq[use]))
            hc = hq[use] - origin
            from ..spatial import cell_index
            cx, cy = cell_index(ref[use], shape, grid)
            cellw = 1.0 / np.maximum(np.bincount(cy * grid[1] + cx,
                                                 minlength=grid[0] * grid[1]), 4)[cy * grid[1] + cx]
            cv = _cv_height(ref[use], src[use], hc, split.cells(src[use]), cellw)
            info["height_cv"] = cv
            if cv and cv["global+height"]["cv_median"] < 0.95 * cv["global"]["cv_median"]:
                B, m = _inverse_fit(ref[use], src[use], hc, cellw)
                Hi = np.vstack([B, [0.0, 0.0, 1.0]])
                try:
                    H = np.linalg.inv(Hi)
                    parallax = {"kind": "dem_height", "sampler": spec,
                                "height_origin_m": origin,
                                "coefficient_px_per_m": [float(m[0]), float(m[1])],
                                "definition": "source working px added per metre of DEM height "
                                              "above height_origin_m at the reference position"}
                    info["parallax"] = {"coefficient_px_per_m": parallax["coefficient_px_per_m"],
                                        "height_origin_m": origin,
                                        "height_std_m": float(np.std(hq[use]))}
                    if log:
                        log("model     : terrain parallax adopted (%.4f, %.4f) working px per m; "
                            "cross-validated median %.3f -> %.3f px"
                            % (m[0], m[1], cv["global"]["cv_median"],
                               cv["global+height"]["cv_median"]))
                except np.linalg.LinAlgError:
                    parallax = None
    field = None
    if allow_field and ok.sum() >= 60 and split is not None:
        base_model = {"matrix": np.asarray(H, float).tolist(), "parallax": parallax}
        resid = src[ok] - WM.inverse_points(base_model, ref[ok])
        ext = ((split.frame["extent"][3] - split.frame["extent"][1],
                split.frame["extent"][2] - split.frame["extent"][0])
               if split.frame is not None else shape)
        cell = float(np.mean([ext[0] / split.grid_shape[0], ext[1] / split.grid_shape[1]]))
        # From twice a held-out cell down to an eighth of one, but never finer
        # than the typical spacing of the points themselves: below that a
        # knot has no data of its own. Cross-validation over whole fit cells
        # decides, and a finer field has to earn its place there.
        try:
            from scipy.spatial import ConvexHull
            hull_area = float(ConvexHull(ref[ok]).volume)
        except Exception:                                             # noqa: BLE001
            hull_area = float(shape[0]) * shape[1]
        spacing_pts = math.sqrt(hull_area / max(ok.sum(), 1))
        cands = [(cell * f, lam) for f in (2.0, 1.0, 0.5, 0.25, 0.125)
                 for lam in (0.01, 0.1, 1.0) if cell * f >= spacing_pts]
        best_cv, table = LM.cross_validate(ref[ok], resid, split.cells(src[ok]), shape, cands)
        info["field_cv"] = table
        if best_cv is not None:
            field = LM.fit(ref[ok], resid, shape, best_cv[0], best_cv[1])
            info["field"] = {"spacing_px": round(best_cv[0], 3), "smoothness": best_cv[1]}
            if log:
                log("model     : local field adopted (spacing %.1f working px, smoothness %g); "
                    "cross-validated median %.3f -> %.3f px"
                    % (best_cv[0], best_cv[1], table[0]["cv_median"],
                       min(r["cv_median"] for r in table[1:] if "cv_median" in r)))
        elif log:
            log("model     : no local field (cross-validation kept the global model)")
    model = {"matrix": np.asarray(H, float).tolist(), "local_field": field, "parallax": parallax}
    res = WM.residuals(model, src, ref)
    inl = np.isfinite(res) & (res < tight)
    vr = V.VerifyResult(np.asarray(H, float), inl, n, int(inl.sum()), res, model_type,
                        ok=bool(inl.sum() >= need),
                        reason="" if inl.sum() >= need else "too few inliers under the dense model",
                        detail={"estimator": "loose RANSAC + neighbour consistency + "
                                             "cell-balanced least squares"
                                             + (" + cross-validated local field" if field else ""),
                                "threshold_px": float(tight)})
    vr.parallax = parallax
    return vr, field, info


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
        # Tie points around the locked translation, seeded on a lattice INSIDE
        # the overlap and measured like the fine stage's. A fixed grid over the
        # overlap's bounding box put most seeds off a diagonal strip: 8 of 144
        # survived on TMC-2 against the morning SELENE map.
        c = _lattice_tiepoints(A, Am, B, Bm, t["dx"], t["dy"], grid)
        if c is None or len(c) < 8:
            bbox = M.overlap_bbox(Am & Bm)
            c = M.grid_tiepoints(A, Am, B, Bm, t["dx"], t["dy"], grid=grid, bbox=bbox,
                                 search=_DENSE_SEARCH)
        if c is None:
            return None, "locked at (%.2f, %.2f) but no tie point survived" % (t["dx"], t["dy"])
        c.detail.update({k: t[k] for k in ("peak", "margin", "subpixel_dx",
                                           "subpixel_dy", "at_search_edge")})
        return c, ""
    return M.sparse(A, Am, B, Bm, name), ""


def _lattice_tiepoints(A, Am, B, Bm, dx, dy, grid, patch=31):
    """Dense coarse tie points: lattice seeds in the overlap, NCC + ECC measured."""
    h, w = A.shape
    ys, xs = np.nonzero(Am)
    if not len(xs):
        return None
    tx, ty = np.rint(xs + dx).astype(int), np.rint(ys + dy).astype(int)
    inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    area = int((inside & Bm[np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)]).sum())
    if area < 64:
        return None
    spacing = max(8.0, math.sqrt(area / float(4 * grid[0] * grid[1])))
    gx, gy = np.meshgrid(np.arange(spacing / 2, w, spacing), np.arange(spacing / 2, h, spacing))
    seeds = np.column_stack([gx.ravel(), gy.ravel()])
    si = np.rint(seeds).astype(int)
    ri = np.rint(seeds + [dx, dy]).astype(int)
    ok = ((ri[:, 0] >= 0) & (ri[:, 0] < w) & (ri[:, 1] >= 0) & (ri[:, 1] < h)
          & (si[:, 0] < w) & (si[:, 1] < h))
    ok[ok] = Am[si[ok, 1], si[ok, 0]] & Bm[ri[ok, 1], ri[ok, 0]]
    seeds = seeds[ok]
    if not len(seeds):
        return None
    c = M.Correspondences(seeds, seeds + [dx, dy], np.ones(len(seeds), np.float32),
                          "dense-ncc", "dense")
    c = M.refine_correspondences(c, A, Am, B, Bm, patch=patch, search=_DENSE_SEARCH)
    if c is not None:
        c.detail.update(seeds=int(len(seeds)), seed_spacing_px=round(spacing, 2))
    return c


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
        if sel.sum() >= max(12, V._min_points(model)):
            Hs = balanced_refit(pts_s[sel], pts_r[sel], model, shape, initial=vr.model)
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


def _write_matches(job_dir, corr, vr, back_x, back_y, frame, per_point_back=None,
                   *, shape=None, grid=(8, 8), per_cell=4):
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
    inliers = np.flatnonzero(vr.inlier_mask & np.isfinite(original).all(axis=1))
    if shape is None:
        shape = back_x.shape
    selection = spatial.select(corr.ref[inliers], corr.confidence[inliers], shape,
                               grid=grid, per_cell=per_cell, min_keep=0,
                               min_for_thinning=0, min_separation=0.)
    chosen = inliers[selection.keep]
    for filename, indices in (("matches_all.csv", np.arange(len(corr))),
                              ("matches.csv", chosen)):
        with open(os.path.join(job_dir, filename), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "confidence", "inlier"])
            for k in indices:
                writer.writerow([*original[k], *reference[k], float(corr.confidence[k]),
                                 int(vr.inlier_mask[k])])
    return {"file": "matches.csv", "all_candidates_file": "matches_all.csv",
            "points": len(chosen), "per_cell_limit": per_cell,
            "grid": list(grid), "quota_enforced": True,
            "occupied_cells": selection.detail.get("cells_occupied", 0),
            "basis": "verified fit points; uniform cell quota, no relaxation"}


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


def _write_transform(job_dir, Hm, model, decomp, seg, frame, method, conv=None, field=None,
                     parallax=None):
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
          "local_field": field,
          "parallax": parallax,
          "application": {"kind": ("blended_segments" if seg else "global")
                                  + ("+terrain_parallax" if parallax is not None else "")
                                  + ("+local_field" if field is not None else ""),
                          "sampling": ("inverse mapping; smoothstep between segment centres" if seg
                                       else "inverse matrix")
                                      + ("; plus the cubic B-spline correction in "
                                         "`local_field`, evaluated at the working-grid "
                                         "reference position" if field is not None else ""),
                          "segment_fallback": "global matrix",
                          "raster": "registered.tif"}}
    with open(os.path.join(job_dir, "transform.json"), "w") as fh:
        json.dump(tj, fh, indent=1)
    return tj


def _overlap_frame(mask):
    """Where to lay the held-out cells: None for the whole working grid, else
    the overlap's extent - along its principal axes when it is elongated.

    An overlap filling most of the frame keeps whole-frame cells. A compact one
    gets axis-aligned cells over its bounding box; only a strip, whose axes
    are well defined, gets rotated cells.
    """
    if mask.mean() >= 0.6:
        return None
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) > 200_000:
        pts = pts[np.linspace(0, len(pts) - 1, 200_000).astype(int)]
    o = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - o, full_matrices=False)
    if sv[0] >= 1.5 * max(sv[1], 1e-9):
        u, v = vt[0], np.array([-vt[0][1], vt[0][0]])
    else:
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    du, dv = (pts - o) @ u, (pts - o) @ v
    # Half a pixel beyond the outermost pixel centres on every side.
    return {"origin": o.tolist(), "axes": [u.tolist(), v.tolist()],
            "extent": [float(du.min() - .5), float(dv.min() - .5),
                       float(du.max() + .5), float(dv.max() + .5)]}


def _square_cells(grid, shape):
    """The requested number of cells, laid out so each cell is roughly square."""
    gy, gx = int(grid[0]), int(grid[1])
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return (gy, gx)
    n = gy * gx
    ny = max(1, int(round(math.sqrt(n * h / float(w)))))
    nx = max(1, int(round(n / float(ny))))
    return (ny, nx)


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
    if cov < 0.5:
        g["coverage"] = ("matches cover %.0f%% of the reference grid; the transform is "
                         "extrapolated over the rest" % (100 * cov))
    if extrap > 0.25:
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
    # The headline error is over check points when they exist; every held-out
    # correspondence is still scored and reported beside it (`held_out_*`).
    rmse_px = acc.get("check_point_rmse_px", acc.get("held_out_rmse_px", acc.get("fit_rmse_px")))
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
    # An operational pass now requires measured native source precision as well
    # as spatial support; independent accuracy certification remains separate.
    notes = []
    sampling = conv.get("source_px_per_reference_px")
    source_rmse = (None if rmse_px is None or not sampling else
                   rmse_px * conv["reference_decimation"] * sampling)
    for key in ("check_point_source_rmse_px", "held_out_source_rmse_px"):
        if key in acc:
            source_rmse = acc[key]
            units["rmse_source_px"] = source_rmse
            break
    assessment = E.acceptance_report(acc, {"coverage_fraction": cov,
                                          "extrapolation_fraction": extrap},
                                      acc.get("independent_ground_truth"))
    reasons.extend(reason for reason in assessment["reasons"] if reason not in reasons)
    accuracy_statement = ("Accuracy unknown in source pixels" if source_rmse is None else
                          "%.2f source px" % source_rmse)
    if source_rmse is not None and acc.get("check_point_n") is not None:
        accuracy_statement += (" RMSE over %d held-out check points (%d of %d held-out "
                               "correspondences set aside as isolated mismatches)"
                               % (acc["check_point_n"], acc["check_point_rejected_n"],
                                  acc["held_out_n"]))
    if sampling:
        accuracy_statement += "; reference sampling scale %.2f source px" % sampling
        notes.append("One reference pixel spans %.2f source pixels. This is a sampling "
                     "scale, not an independently measured error bound." % sampling)
    reasons += list(warn or [])
    status = "pass" if not reasons else "warning"

    m = {"status": status, "reason": "; ".join(reasons) if reasons else None,
         "notes": notes,
         "status_meaning": "pass requires source-pixel consistency and spatial support; not an accuracy guarantee",
         "acceptance": assessment,
         "accuracy_statement": accuracy_statement,
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
             subpixel=(None if source_rmse is None else bool(source_rmse < 1.0)),
             independently_verified_subpixel=assessment["independently_verified"],
             subpixel_basis="source pixels",
             subpixel_working_grid=bool(rmse_px is not None and rmse_px < 1.0),
             # Legacy sampling fields are retained for clients. They describe
             # a one-reference-pixel sampling assumption, not a precision bound.
             reference_sampling_source_px=sampling,
             sampling_floor_basis="nominal one-reference-pixel sampling; not an error lower bound",
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
                              if export_model and (export_model.get("segments")
                                                   or export_model.get("local_field")
                                                   or export_model.get("parallax")) else
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
         m.get("accuracy_statement", "") + "\n",
         m.get("status_meaning", "") + "\n",
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
