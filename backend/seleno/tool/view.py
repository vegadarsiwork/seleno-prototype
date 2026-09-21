"""Look at any input at any zoom: 256 px tiles cut on demand from the raster itself.

The registration previews are one fixed-size PNG per image. That is 2800 px
across for a 109 164 px mosaic, so 39 source pixels share each preview pixel
and a crater smaller than 4 km does not exist on screen. This module serves the
same raster the tool reads, at whatever scale the viewer asks for, so any
location can be inspected down to its native pixels.

Scale is always a power of two, `s` native pixels per tile pixel, matching the
level pyramid OpenSeadragon walks. Two sources serve a tile:

* **Native reads**, for `s` below the overview factor. The window is at most
  `256 * s` pixels square, block-averaged down to 256. On a memory-mapped file
  that is a few milliseconds.
* **The overview**, for everything coarser. It is built once per file by one
  sequential pass - block averages, not strided samples, so a zoomed-out view
  does not alias crater fields into moire - and cached on disk, keyed by the
  file's size and mtime.

What is shown is the band the tool registers: band 1 of a GeoTIFF, the grey
conversion of a PNG, the reflected-light collapse of a spectrometer cube.
Values go through one linear stretch fixed per file, so neighbouring tiles
agree and a dark region stays dark rather than being re-normalised to look lit.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass, field

import numpy as np

TILE = 256
# The overview's long side. 4096 keeps it at most 64 MB as float32, and puts
# the handover at 1/32 on the WAC mosaic, where a native tile read is 8192 px
# square - still a small read from a memory map.
OVERVIEW_MAX_SIDE = 4096
# A saved view is capped at this many pixels on its long side...
REGION_MAX_SIDE = 8192
# ...and may read at most this many native pixels before falling back to the
# overview. Past that, a "save what I see" click would read gigabytes.
REGION_MAX_READ_PX = 1 << 30
_BAND_BYTES = 128 << 20


@dataclass
class Viewable:
    path: str
    array: object                  # memmap, LazyRaster or ndarray
    height: int
    width: int
    dtype: str
    reader: str
    instrument: str
    gsd_m: float | None
    nodata: float | None
    crs: object | None
    transform: object | None
    version: int                   # mtime, so a replaced file gets fresh tiles
    key: str = ""                  # path|size|mtime_ns: when this stops matching, reopen
    degraded: list = field(default_factory=list)
    factor: int = 1                # native px per overview px; 1 = no overview
    overview: np.ndarray | None = None
    lo: float = 0.0
    hi: float = 255.0

    @property
    def max_scale(self) -> int:
        """The scale at which the whole image fits one pixel: the pyramid's top."""
        return 1 << int(math.ceil(math.log2(max(self.height, self.width, 1))))

    def georef(self) -> dict | None:
        """Enough of the map projection for the browser to show lat/lon.

        Only the two cylindrical cases are described: latitude and longitude
        are linear in the pixel coordinates there, so the browser can convert
        both ways without a projection library. Anything else is reported as
        unsupported rather than approximated.
        """
        if self.crs is None or self.transform is None:
            return None
        t = self.transform
        aff = [t.a, t.b, t.c, t.d, t.e, t.f]
        try:
            if self.crs.is_geographic:
                return {"kind": "longlat", "affine": aff}
            d = self.crs.to_dict()
        except Exception:
            return None
        if d.get("proj") == "eqc":
            r = d.get("R") or d.get("a")
            if not r:
                return None
            return {"kind": "eqc", "affine": aff, "R": float(r),
                    "lat_ts": float(d.get("lat_ts", 0.0)),
                    "lon_0": float(d.get("lon_0", 0.0)),
                    "x_0": float(d.get("x_0", 0.0)), "y_0": float(d.get("y_0", 0.0))}
        return {"kind": None, "affine": aff, "proj": d.get("proj")}

    def info(self) -> dict:
        return {"width": self.width, "height": self.height, "dtype": self.dtype,
                "reader": self.reader, "instrument": self.instrument,
                "gsd_m": self.gsd_m, "version": self.version,
                "tile_size": TILE, "max_scale": self.max_scale,
                "overview_factor": self.factor,
                "stretch": [round(self.lo, 6), round(self.hi, 6)],
                "georef": self.georef(), "degraded": self.degraded}


# --------------------------------------------------------------------------- #
# opening
# --------------------------------------------------------------------------- #

def _tiff_memmap(path: str):
    """The TIFF's pixels as a numpy memmap, if they are stored uncompressed.

    GDAL reads a one-row-per-strip mosaic a strip at a time, so a 2048 px tall
    window costs 2048 full-width strips - 220 MB for the WAC mosaic. Mapped
    directly, the same window touches only its own bytes.
    """
    try:
        import tifffile
        a = tifffile.memmap(path, mode="r")
    except Exception:
        return None
    return a if a.ndim == 2 else None


def _open_scene(path: str):
    from . import scene as S
    from .profiles import Profiles
    prof = Profiles.load()
    if path.lower().endswith((".tif", ".tiff")):
        mm = _tiff_memmap(path)
        if mm is not None:
            # `_try_gdal` rather than `load`: load's validity check samples the
            # whole raster, which through GDAL is a full read of a 6 GB file.
            sc = S._try_gdal(path, prof)
            if sc is not None and tuple(sc.shape) == mm.shape:
                sc.array = mm
                return sc
    return S.load(path, prof)


def _as_float(block, nodata) -> np.ndarray:
    a = np.array(block, dtype=np.float32)
    # A nodata of 0 already renders black, and blanking it would also blank
    # every genuinely black shadow pixel of an 8-bit frame. Any other nodata
    # (a DEM's -32768, a float fill value) is removed before averaging.
    if nodata is not None and nodata != 0 and np.isfinite(nodata):
        a[a == nodata] = np.nan
    a[np.isinf(a)] = np.nan
    return a


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """f x f block averages. A partial block at the right or bottom edge is the
    mean of the pixels it actually has - edge-padding it instead would count
    the last row or column several times over."""
    if f <= 1:
        return a
    import cv2
    h, w = a.shape
    fh, fw = h // f * f, w // f * f
    out = np.empty((-(-h // f), -(-w // f)), np.float32)
    if fh and fw:
        out[:fh // f, :fw // f] = cv2.resize(a[:fh, :fw], (fw // f, fh // f),
                                             interpolation=cv2.INTER_AREA)
    if fw < w and fh:
        out[:fh // f, -1] = a[:fh, fw:].reshape(fh // f, f, w - fw).mean(axis=(1, 2))
    if fh < h:
        if fw:
            out[-1, :fw // f] = a[fh:, :fw].reshape(h - fh, fw // f, f).mean(axis=(0, 2))
        if fw < w:
            out[-1, -1] = a[fh:, fw:].mean()
    return out


def _factor(h: int, w: int) -> int:
    f = 1
    while max(h, w) / f > OVERVIEW_MAX_SIDE:
        f *= 2
    return f


def _build_overview(arr, h: int, w: int, f: int, nodata) -> np.ndarray:
    out = np.empty((-(-h // f), -(-w // f)), np.float32)
    rows = max(f, (_BAND_BYTES // max(1, w * 4)) // f * f)
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        blk = _block_mean(_as_float(arr[r0:r1, 0:w], nodata), f)
        out[r0 // f: r0 // f + blk.shape[0]] = blk
    return out


def _stretch_limits(sample: np.ndarray, nodata) -> tuple[float, float]:
    v = sample[np.isfinite(sample)]
    if nodata == 0:
        v = v[v != 0]
    if v.size < 16:
        return 0.0, 1.0
    lo, hi = (float(x) for x in np.percentile(v, (0.5, 99.5)))
    if hi - lo < 1e-9:
        lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


_CACHE: dict[str, Viewable] = {}
_CACHE_ORDER: list[str] = []
_CACHE_MAX = 6
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def open_view(path: str, cache_dir: str) -> Viewable:
    """A Viewable for `path`, building and caching its overview the first time.

    The first open of a large file makes one pass over it. Later opens, in this
    process or the next, load the cached overview instead.
    """
    path = os.path.abspath(path)
    st = os.stat(path)
    key = "%s|%d|%d" % (path, st.st_size, st.st_mtime_ns)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(path, threading.Lock())
    with lock:
        v = _CACHE.get(path)
        if v is not None and v.key == key:
            return v
        sc = _open_scene(path)
        h, w = (int(x) for x in sc.shape)
        v = Viewable(path=path, array=sc.array, height=h, width=w,
                     dtype=str(getattr(sc.array, "dtype", "float32")),
                     reader=sc.reader, instrument=sc.instrument, gsd_m=sc.gsd_m,
                     nodata=sc.nodata, crs=sc.crs, transform=sc.transform,
                     version=int(st.st_mtime), key=key, degraded=list(sc.degraded))
        v.factor = _factor(h, w)
        if v.factor > 1:
            os.makedirs(cache_dir, exist_ok=True)
            digest = hashlib.sha1(("%s|%d|v2" % (key, v.factor)).encode()).hexdigest()[:20]
            cpath = os.path.join(cache_dir, digest + ".npy")
            if not os.path.exists(cpath):
                ov = _build_overview(sc.array, h, w, v.factor, v.nodata)
                tmp = cpath + ".%d.tmp.npy" % os.getpid()
                np.save(tmp, ov)
                os.replace(tmp, cpath)
            v.overview = np.load(cpath, mmap_mode="r")
            sample = np.asarray(v.overview)
        else:
            sample = _as_float(sc.array[0:h, 0:w], v.nodata)
        v.lo, v.hi = _stretch_limits(sample, v.nodata)
        _CACHE[path] = v
        if path in _CACHE_ORDER:
            _CACHE_ORDER.remove(path)
        _CACHE_ORDER.append(path)
        while len(_CACHE_ORDER) > _CACHE_MAX:
            _CACHE.pop(_CACHE_ORDER.pop(0), None)
        return v


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _fit(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    a = a[:th, :tw]
    if a.shape != (th, tw):
        a = np.pad(a, ((0, th - a.shape[0]), (0, tw - a.shape[1])), mode="edge")
    return a


def render(v: Viewable, x0: int, y0: int, x1: int, y1: int, s: int) -> np.ndarray:
    """Native window [y0:y1, x0:x1] at 1/s scale, stretched to uint8.

    `s` must be a power of two. The result is exactly ceil(h/s) x ceil(w/s),
    which is the size OpenSeadragon expects an edge tile to be.
    """
    th, tw = -(-(y1 - y0) // s), -(-(x1 - x0) // s)
    if v.overview is not None and s >= v.factor:
        f = v.factor
        oh, ow = v.overview.shape
        blk = np.array(v.overview[y0 // f: min(oh, -(-y1 // f)),
                                  x0 // f: min(ow, -(-x1 // f))], np.float32)
        out = _block_mean(blk, s // f)
    else:
        rows = max(s, (_BAND_BYTES // max(1, (x1 - x0) * 4)) // s * s)
        parts = []
        for r0 in range(y0, y1, rows):
            r1 = min(y1, r0 + rows)
            parts.append(_block_mean(_as_float(v.array[r0:r1, x0:x1], v.nodata), s))
        out = parts[0] if len(parts) == 1 else np.vstack(parts)
    out = _fit(out, th, tw)
    u = (out - v.lo) * (255.0 / (v.hi - v.lo))
    np.clip(u, 0.0, 255.0, out=u)
    return np.rint(np.nan_to_num(u, nan=0.0)).astype(np.uint8)


def tile(v: Viewable, s: int, tx: int, ty: int) -> np.ndarray:
    if s < 1 or s & (s - 1) or s > v.max_scale:
        raise ValueError("scale must be a power of two between 1 and %d" % v.max_scale)
    x0, y0 = tx * TILE * s, ty * TILE * s
    if tx < 0 or ty < 0 or x0 >= v.width or y0 >= v.height:
        raise ValueError("tile %d,%d is outside the image at scale %d" % (tx, ty, s))
    return render(v, x0, y0, min(v.width, x0 + TILE * s), min(v.height, y0 + TILE * s), s)


def region_scale(v: Viewable, w: int, h: int) -> int:
    """The finest power-of-two scale a saved view of w x h native px can use."""
    s = 1
    while max(w, h) / s > REGION_MAX_SIDE:
        s *= 2
    if v.overview is not None and s < v.factor and w * h > REGION_MAX_READ_PX:
        s = v.factor
    return s


def region(v: Viewable, x0: float, y0: float, x1: float, y1: float):
    """A saved view: the requested window, clamped, at the finest scale allowed."""
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(v.width, int(math.ceil(x1))), min(v.height, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the requested region does not overlap the image")
    s = region_scale(v, x1 - x0, y1 - y0)
    return render(v, x0, y0, x1, y1, s), (x0, y0, x1, y1), s


def png(a: np.ndarray, level: int = 1) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", a, [cv2.IMWRITE_PNG_COMPRESSION, level])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
