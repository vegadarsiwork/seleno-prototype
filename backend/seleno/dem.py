"""LOLA polar DEM access and physically-grounded relighting.

Why this module exists
----------------------
At the lunar south pole the Sun is always within a degree of the horizon, so the
image is not "terrain plus shading" - it is **mostly cast shadow**. A 100 m rim
at 1 degree of solar elevation throws a 5.7 km shadow. That is why two images of
the same ground 90 degrees apart in Sun azimuth look like different places, and
it is why a descriptor that survives brightness changes still fails here.

It also means the shadow field is *predictable*: it is a function of the terrain
and the Sun direction, both of which we have. Rendering the DEM under the source
image's own Sun geometry turns an appearance-matching problem into a
terrain-matching problem.

Conventions, all verified in `tests/test_dem.py`
------------------------------------------------
* Grids are the project's south polar stereographic plane,
  ``+proj=stere +lat_0=-90 +lon_0=0 +R=1737400``.
* **Row index increases with +y, i.e. northwards**, and column index with +x.
  This matches `ohrc.reproject.StereoWindow`, which the rest of the project
  already uses, and is the *opposite* of the GeoTIFF/PDS raster convention.
  Anything loaded from a NAC or WAC raster must therefore be flipped top-to-
  bottom on the way in; `nac.load_browse` does that and asserts it.
* Within 50 km of the pole the projection's scale factor departs from 1 by
  under 4e-5, so one plane metre is one ground metre to well under a pixel.
  `scale_factor_at` reports it rather than assuming it.
* The Sun direction is derived per pixel from the body-fixed sun vector and the
  local ENU frame - never from a hand-written azimuth formula, because the
  meridians converge at the pole and a single azimuth is wrong across a 50 km
  box.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .ohrc.geometry import lonlat_to_south_stereo, south_stereo_to_lonlat

R_MOON = 1737400.0


# --------------------------------------------------------------------------- #
# the DEM
# --------------------------------------------------------------------------- #

def parse_pds3_label(path: str) -> dict:
    out = {}
    for line in open(path, errors="replace"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().strip('"')
        v = v.split("<")[0].strip().strip('"').strip()
        if k and k not in out:
            out[k] = v
    return out


@dataclass
class LolaDEM:
    """A LOLA GDR polar DEM, memory-mapped and read in bounded windows.

    The delivered product is a square polar-stereographic grid centred on the
    pole. Nothing here loads it whole: `ldem_875s_5m` is 30 336 x 30 336.
    """

    path: str
    lines: int
    samples: int
    scale: float
    offset: float
    res_m: float
    lat_max: float

    @classmethod
    def open(cls, lbl_path: str) -> "LolaDEM":
        L = parse_pds3_label(lbl_path)
        img = lbl_path[:-4] + ".img"
        if not os.path.exists(img):
            raise FileNotFoundError(img)
        n, m = int(L["LINES"]), int(L["LINE_SAMPLES"])
        want = n * m * int(L["SAMPLE_BITS"]) // 8
        got = os.path.getsize(img)
        if got != want:
            raise ValueError("%s is %d bytes, label declares %d - incomplete download"
                             % (img, got, want))
        return cls(img, n, m, float(L["SCALING_FACTOR"]), float(L["OFFSET"]),
                   float(L["MAP_SCALE"]), float(L["MAXIMUM_LATITUDE"]))

    @property
    def _map(self) -> np.memmap:
        return np.memmap(self.path, dtype="<i2", mode="r", shape=(self.lines, self.samples))

    def stereo_to_pixel(self, x, y):
        """Plane metres -> (col, row). Grid centre is the pole, y increases north."""
        return (np.asarray(x) / self.res_m + self.samples / 2.0,
                self.lines / 2.0 - np.asarray(y) / self.res_m)

    def window(self, x0: float, y0: float, x1: float, y1: float, res_m: float):
        """Elevation in metres above a 1 737 400 m sphere, on a regular grid.

        Returns ``(dem, gx, gy)`` where `gx`, `gy` are the pixel-centre plane
        coordinates of the output, so callers never have to re-derive them.
        """
        w = max(2, int(round((x1 - x0) / res_m)))
        h = max(2, int(round((y1 - y0) / res_m)))
        gx = x0 + (np.arange(w) + 0.5) * res_m
        gy = y0 + (np.arange(h) + 0.5) * res_m          # row 0 is the SOUTH edge
        C, R = self.stereo_to_pixel(*np.meshgrid(gx, gy))

        c0, c1 = int(np.floor(C.min())) - 1, int(np.ceil(C.max())) + 2
        r0, r1 = int(np.floor(R.min())) - 1, int(np.ceil(R.max())) + 2
        if c0 < 0 or r0 < 0 or c1 > self.samples or r1 > self.lines:
            raise ValueError("window (%.0f..%.0f, %.0f..%.0f) runs off %s, which covers "
                             "|lat| >= %.1f" % (x0, x1, y0, y1,
                                                os.path.basename(self.path), abs(self.lat_max)))
        block = np.asarray(self._map[r0:r1, c0:c1], dtype=np.float32)
        block = block * self.scale + self.offset - R_MOON

        # bilinear, vectorised
        cc, rr = C - c0, R - r0
        i0 = np.clip(np.floor(cc).astype(np.int64), 0, block.shape[1] - 2)
        j0 = np.clip(np.floor(rr).astype(np.int64), 0, block.shape[0] - 2)
        fx, fy = cc - i0, rr - j0
        dem = (block[j0, i0] * (1 - fx) * (1 - fy) + block[j0, i0 + 1] * fx * (1 - fy)
               + block[j0 + 1, i0] * (1 - fx) * fy + block[j0 + 1, i0 + 1] * fx * fy)
        return dem.astype(np.float32), gx, gy


# --------------------------------------------------------------------------- #
# solar geometry on the grid
# --------------------------------------------------------------------------- #

def scale_factor_at(x: float, y: float) -> float:
    """Polar-stereographic scale factor: plane metres per ground metre."""
    lon, lat = south_stereo_to_lonlat(np.array([x]), np.array([y]))
    return float(2.0 / (1.0 + math.sin(math.radians(-float(lat[0])))))


def sun_vectors(gx: np.ndarray, gy: np.ndarray, sub_solar_lon_deg: float,
                sub_solar_lat_deg: float = 0.0):
    """Per-pixel Sun direction, expressed in the plane's own axes.

    Returns ``(lx, ly, lz)``: a unit vector per pixel pointing from the surface
    towards the Sun, with `lx`/`ly` along the grid's +x/+y and `lz` up.

    The plane directions of local east and north are obtained by differentiating
    the projection numerically, so no azimuth sign convention is hand-written.
    """
    X, Y = np.meshgrid(gx, gy)
    lon, lat = south_stereo_to_lonlat(X.ravel(), Y.ravel())
    lon = np.radians(lon.reshape(X.shape))
    lat = np.radians(lat.reshape(X.shape))

    ls, bs = math.radians(sub_solar_lon_deg), math.radians(sub_solar_lat_deg)
    S = np.array([math.cos(bs) * math.cos(ls), math.cos(bs) * math.sin(ls), math.sin(bs)])

    clat, slat, clon, slon = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
    up_e, up_n, up_u = clat * clon, clat * slon, slat
    e_e, e_n = -slon, clon
    n_e, n_n, n_u = -slat * clon, -slat * slon, clat

    s_up = S[0] * up_e + S[1] * up_n + S[2] * up_u
    s_east = S[0] * e_e + S[1] * e_n
    s_north = S[0] * n_e + S[1] * n_n + S[2] * n_u

    # plane directions of local east / north, by finite difference of the projection
    d = 1e-4                                            # degrees
    lo_d, la_d = np.degrees(lon), np.degrees(lat)
    xe, ye = lonlat_to_south_stereo(lo_d + d, la_d)
    xw, yw = lonlat_to_south_stereo(lo_d - d, la_d)
    xn, yn = lonlat_to_south_stereo(lo_d, la_d + d)
    xs_, ys_ = lonlat_to_south_stereo(lo_d, la_d - d)
    ex, ey = xe - xw, ye - yw
    nx, ny = xn - xs_, yn - ys_
    ex, ey = ex / np.hypot(ex, ey), ey / np.hypot(ex, ey)
    nx, ny = nx / np.hypot(nx, ny), ny / np.hypot(nx, ny)

    lx = s_east * ex + s_north * nx
    ly = s_east * ey + s_north * ny
    lz = s_up
    norm = np.sqrt(lx * lx + ly * ly + lz * lz)
    return (lx / norm).astype(np.float32), (ly / norm).astype(np.float32), \
           (lz / norm).astype(np.float32)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def surface_normals(dem: np.ndarray, res_m: float):
    # axis 0 is +y (north), axis 1 is +x (east)
    dz_dy, dz_dx = np.gradient(dem.astype(np.float32), res_m)
    nx, ny, nz = -dz_dx, -dz_dy, np.ones_like(dz_dx)
    n = np.sqrt(nx * nx + ny * ny + 1.0)
    return nx / n, ny / n, nz / n


def cast_shadow(dem: np.ndarray, res_m: float, lx: float, ly: float, lz: float,
                max_range_m: float = 30000.0, near_px: int = 40,
                growth: float = 1.12) -> np.ndarray:
    """Ray-march towards the Sun; True where the direct beam is blocked.

    At one degree of solar elevation a 100 m obstacle shadows 5.7 km, so this is
    not a refinement - without it a polar render bears no resemblance to a polar
    image.

    Sampling is dense for the first `near_px` pixels and then grows
    geometrically, because a blocker 20 km away that matters is a crater rim or
    a massif, not a single pixel. That turns an O(range/res) march into an
    O(near + log(range)) one: at 16 m/px over 25 km it is ~120 steps instead of
    1560. Thin distant ridges can be missed; the near field, which sets every
    shadow edge a matcher actually uses, is exact.

    Marched on a single mean Sun direction, exact to better than a pixel over a
    50 km box (the direction varies by <0.9 degrees across it).
    """
    h, w = dem.shape
    if lx == 0 and ly == 0:
        return np.zeros((h, w), bool)
    hyp = math.hypot(lx, ly)
    ux, uy = lx / hyp, ly / hyp
    slope = lz / hyp                                   # height gained per unit distance

    dists, d, step = [], 1.0, 1.0
    while d * res_m <= max_range_m:
        dists.append(d)
        if len(dists) > near_px:
            step *= growth
        d += step
    if not dists:
        return np.zeros((h, w), bool)

    shadow = np.zeros((h, w), bool)
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    for d in dists:
        cc = cols + ux * d
        rr = rows + uy * d                             # rows increase with +y
        inside = (cc >= 0) & (cc <= w - 1) & (rr >= 0) & (rr <= h - 1)
        if not inside.any():
            break
        ci = np.clip(cc, 0, w - 1).astype(np.int32)
        ri = np.clip(rr, 0, h - 1).astype(np.int32)
        beam = dem + slope * d * res_m
        shadow |= inside & (dem[ri, ci] > beam)
    return shadow


def render(dem: np.ndarray, gx: np.ndarray, gy: np.ndarray, res_m: float,
           sub_solar_lon_deg: float, sub_solar_lat_deg: float = 0.0, *,
           shadows: bool = True, max_range_m: float = 30000.0):
    """Relight the terrain under a given sub-solar point.

    Returns ``(radiance, lit)`` with radiance in [0, 1]: Lambertian cos(i), zeroed
    where the terrain is self-shadowed or in cast shadow. Deliberately *not*
    Lunar-Lambert - at grazing incidence the geometric shadow term dominates any
    photometric refinement, and a parameter-free model is easier to defend.
    """
    lx, ly, lz = sun_vectors(gx, gy, sub_solar_lon_deg, sub_solar_lat_deg)
    nx, ny, nz = surface_normals(dem, res_m)
    cosi = nx * lx + ny * ly + nz * lz
    lit = cosi > 0
    if shadows:
        lit &= ~cast_shadow(dem, res_m, float(lx.mean()), float(ly.mean()),
                            float(lz.mean()), max_range_m=max_range_m)
    return (np.clip(cosi, 0, 1) * lit).astype(np.float32), lit


def default_dem_path(root: str | None = None) -> str | None:
    """The 5 m south-polar DEM fetched in Phase 1, if it is present."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = root or os.path.abspath(os.path.join(here, "..", ".."))
    for stem in ("ldem_875s_5m", "ldem_80s_20m"):
        p = os.path.join(root, "data", "raw", "lola", stem + ".lbl")
        if os.path.exists(p) and os.path.exists(p[:-4] + ".img"):
            return p
    return None
