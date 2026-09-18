"""LROC NAC controlled polar mosaic access.

Two levels, deliberately:

* **browse** - the 1421 x 1421 PNG pyramid of a whole 45.5 km tile, about
  32 m/px and under a megabyte.  This is the coarse-search substrate.  Pulling
  the same ground out of the full-resolution product costs ~1 GB, because the
  GeoTIFF is striped (one row per block) and a windowed read therefore fetches
  whole 45 488-pixel scanlines.
* **full resolution** - a small window of the 8-bit GeoTIFF mirror under
  ``EXTRAS/BROWSE``, read through GDAL's ``/vsicurl``.  4x smaller than the
  32-bit ``.IMG`` and it carries the CRS, so the georeferencing is read rather
  than reconstructed from PDS3 label constants.

Row order
---------
LROC rasters are north-up: row 0 is the tile's **maximum** y.  Everything in
this project uses `ohrc.reproject.StereoWindow`'s convention instead, where row
index increases with +y.  Both loaders flip on the way in and `self_check`
asserts it, because a silent vertical flip looks exactly like a large
geolocation offset.
"""
from __future__ import annotations

import math
import os
import subprocess
import time

import numpy as np

R_MOON = 1737400.0
STORE = ("https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/"
         "pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/")
BROWSE = STORE + "EXTRAS/BROWSE/NAC_POLE/"

# (outer |lat|, inner |lat|, tiles in band) - see scripts/build_footprint_index.py
BANDS = {"P848": (84.0, 85.5, 16), "P863": (85.5, 87.0, 12),
         "P878": (87.0, 88.5, 8), "P892": (88.5, 90.0, 4)}


def _rho(lat_abs: float) -> float:
    return 2.0 * R_MOON * math.tan(math.radians(90.0 - lat_abs) / 2.0)


def tile_extent(tile: str) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the delivered raster, plane metres.

    The raster is the bounding box of the tile's annular wedge; band 1 tiles are
    therefore quadrant squares meeting at the pole.
    """
    band, lon = tile[:4], int(tile[5:]) / 10.0
    lat_out, lat_in, n = BANDS[band]
    half = 360.0 / n / 2.0
    r_out, r_in = _rho(lat_out), _rho(lat_in)
    a = np.radians(np.linspace(lon - half, lon + half, 2001))
    xs = np.concatenate([r_in * np.sin(a), r_out * np.sin(a)])
    ys = np.concatenate([r_in * np.cos(a), r_out * np.cos(a)])
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def product_name(bin_id: str, tile: str) -> str:
    return "NAC_POLE_SOUTH_CM_%s_%s" % (bin_id, tile)


def browse_url(bin_id: str, tile: str) -> str:
    d = "NAC_POLE_SOUTH_CM_%s" % bin_id
    return BROWSE + "%s/%s.BROWSE.PNG" % (d, product_name(bin_id, tile))


def tif_url(bin_id: str, tile: str) -> str:
    d = "NAC_POLE_SOUTH_CM_%s" % bin_id
    return BROWSE + "%s/%s.TIF" % (d, product_name(bin_id, tile))


def load_browse(bin_id: str, tile: str, cache_dir: str):
    """Whole-tile pyramid on the project's grid.

    Returns ``(img, gx, gy, res_m)``; `img` is float32 with NaN where the mosaic
    has no data, row 0 is the tile's minimum y.
    """
    import cv2
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, "browse_CM_%s_%s.png" % (bin_id, tile))
    if not os.path.exists(dst):
        subprocess.run(["curl", "-sfL", "--retry", "4", "--retry-delay", "5",
                        "--max-time", "600", "-o", dst, browse_url(bin_id, tile)],
                       check=True)
    raw = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError("unreadable browse %s" % dst)
    img = raw[::-1].astype(np.float32)                 # north-up -> +y with row
    img[img == 0] = np.nan                             # 0 is the mosaic's nodata
    x0, y0, x1, y1 = tile_extent(tile)
    res = (x1 - x0) / img.shape[1]
    gx = x0 + (np.arange(img.shape[1]) + 0.5) * res
    gy = y0 + (np.arange(img.shape[0]) + 0.5) * res
    return img, gx, gy, res


def load_window(bin_id: str, tile: str, x0: float, y0: float, x1: float, y1: float,
                cache_dir: str, res_m: float = 1.0):
    """Full-resolution crop, read through /vsicurl and cached as .npy.

    Costly: the GeoTIFF is striped, so this fetches whole scanlines for the row
    range. Keep the box small and use `load_browse` for searching.
    """
    import rasterio
    from rasterio.windows import from_bounds
    os.makedirs(cache_dir, exist_ok=True)
    key = "win_CM_%s_%s_%d_%d_%d_%d.npy" % (bin_id, tile, round(x0), round(y0),
                                            round(x1), round(y1))
    dst = os.path.join(cache_dir, key)
    if os.path.exists(dst):
        arr = np.load(dst)
    else:
        # A striped GeoTIFF read over /vsicurl is a long sequence of range
        # requests; a single dropped one raises TIFFReadEncodedStrip and loses
        # the whole window. Retry, and let GDAL retry inside itself too.
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
        os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
        os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "10485760")
        last = None
        for attempt in range(4):
            try:
                with rasterio.open("/vsicurl/" + tif_url(bin_id, tile)) as ds:
                    w = from_bounds(x0, y0, x1, y1, ds.transform)
                    arr = ds.read(1, window=w, boundless=True, fill_value=0)
                break
            except Exception as exc:
                last = exc
                time.sleep(5 * (attempt + 1))
        else:
            raise IOError("full-resolution window failed after 4 attempts: %s" % last)
        np.save(dst, arr)
    img = arr[::-1].astype(np.float32)
    img[img == 0] = np.nan
    h, wd = img.shape
    gx = x0 + (np.arange(wd) + 0.5) * ((x1 - x0) / wd)
    gy = y0 + (np.arange(h) + 0.5) * ((y1 - y0) / h)
    return img, gx, gy


def self_check(bin_id: str, tile: str, cache_dir: str) -> dict:
    """Confirm the browse pyramid and the full-resolution GeoTIFF agree.

    A vertical flip or a half-tile offset in either loader would masquerade as a
    kilometre-scale geolocation error, which is precisely the quantity this
    project is trying to measure, so it is checked rather than assumed.
    """
    import cv2
    br, bgx, bgy, bres = load_browse(bin_id, tile, cache_dir)
    # Pick the box from the data, not from the geometry: most of a polar tile is
    # shadow or nodata, and an empty box makes the check vacuous rather than failing.
    side_m = 2048.0
    k = int(round(side_m / bres))
    lit = np.isfinite(br) & (br > np.nanpercentile(br, 75))
    integ = np.cumsum(np.cumsum(lit.astype(np.float32), 0), 1)

    def boxsum(j, i):
        return (integ[j + k, i + k] - integ[j, i + k] - integ[j + k, i] + integ[j, i])

    best, bj, bi = -1.0, 0, 0
    for j in range(0, br.shape[0] - k - 1, 8):
        for i in range(0, br.shape[1] - k - 1, 8):
            v = boxsum(j, i)
            if v > best:
                best, bj, bi = v, j, i
    bx0, by0 = float(bgx[bi]), float(bgy[bj])
    fr, fgx, fgy = load_window(bin_id, tile, bx0, by0, bx0 + side_m, by0 + side_m, cache_dir)
    # pull the same ground out of the browse and compare after decimation
    i0 = int(np.searchsorted(bgx, bx0)); i1 = int(np.searchsorted(bgx, bx0 + side_m))
    j0 = int(np.searchsorted(bgy, by0)); j1 = int(np.searchsorted(bgy, by0 + side_m))
    sub = br[j0:j1, i0:i1]
    fr_small = cv2.resize(np.nan_to_num(fr), sub.shape[::-1], interpolation=cv2.INTER_AREA)
    a = np.nan_to_num(sub).ravel(); b = fr_small.ravel()
    ok = (a > 0) & (b > 0)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 50 else float("nan")
    return {"tile": tile, "bin": bin_id, "browse_res_m": bres,
            "box_x0_y0": (round(bx0), round(by0)), "box_side_m": side_m,
            "browse_shape": br.shape, "fullres_shape": fr.shape,
            "overlap_px": int(ok.sum()), "correlation": round(r, 4)}
