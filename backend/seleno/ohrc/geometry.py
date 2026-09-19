"""Geolocation for OHRC products, and the projected plane the overlap math uses.

Two things live here.

**The delivered geometry grid.** ISRO ships a ``geometry/**/*_g_grd_*.csv`` per
product: longitude and latitude on a sparse (Pixel, Scan) lattice - 121
cross-track nodes (0, 100, ..., 11900, 11999) by one node per 100 lines. We load
it once into a dense ``(n_scan, n_pixel, 2)`` array (~1 MB) and interpolate
bilinearly. Per-pixel CSV lookups would be absurd.

**Polar stereographic coordinates.** These strips pass within 0.05 deg of the
south pole. Longitude there is degenerate: one product's footprint spans
longitude 222 deg -> 110 deg -> 22 deg while covering 25 km of ground. Any
overlap, distance or footprint computation done in lon/lat will be wrong. So all
geometry is reduced to a south polar stereographic plane in metres, which is also
the projection the labels themselves declare.

Accuracy caveat that matters: the delivered grid is *system-level* - derived from
predicted orbit and attitude, with no photogrammetric refinement (the label's
"refined" corners are byte-identical to the system ones). It is a good prior for
finding which windows overlap. It is not sub-pixel truth, and nothing here
pretends otherwise.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np

# IAU mean lunar radius. The labels give no ellipsoid, and at 25 km scales the
# difference between sphere and ellipsoid is far below the geolocation error.
MOON_RADIUS_M = 1737400.0


# --------------------------------------------------------------------------- #
# south polar stereographic
# --------------------------------------------------------------------------- #

def lonlat_to_south_stereo(lon_deg, lat_deg):
    """(lon, lat) in degrees -> (x, y) metres on the south polar plane.

    Standard south polar stereographic, projection point at the north pole,
    plane tangent at the south pole. The south pole maps to (0, 0). Scale
    distortion at 25 km from the pole is ~5e-5, i.e. about 1 m - negligible
    against the metre-to-decametre geolocation error of the input.
    """
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    # colatitude measured from the SOUTH pole
    rho = 2.0 * MOON_RADIUS_M * np.tan((math.pi / 2.0 + lat) / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def south_stereo_to_lonlat(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.hypot(x, y)
    lat = np.degrees(2.0 * np.arctan(rho / (2.0 * MOON_RADIUS_M)) - math.pi / 2.0)
    lon = np.degrees(np.arctan2(x, y)) % 360.0
    return lon, lat


def ground_distance_m(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance on a spherical Moon."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * MOON_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- #
# the delivered lattice
# --------------------------------------------------------------------------- #

@dataclass
class GeometryGrid:
    """Bilinear interpolator over the delivered (Pixel, Scan) -> (lon, lat) grid."""

    pixels: np.ndarray            # (n_pixel,) sample coordinates, ascending
    scans: np.ndarray             # (n_scan,)  line coordinates, ascending
    lon: np.ndarray               # (n_scan, n_pixel)
    lat: np.ndarray               # (n_scan, n_pixel)
    x: np.ndarray                 # (n_scan, n_pixel) south stereographic metres
    y: np.ndarray
    source: str = ""

    # ------------------------------------------------------------------ load
    @classmethod
    def from_csv(cls, path: str) -> "GeometryGrid":
        px_l, sc_l, lon_l, lat_l = [], [], [], []
        with open(path, newline="") as fh:
            rdr = csv.reader(fh)
            header = next(rdr)
            cols = {name.strip().lower(): i for i, name in enumerate(header)}
            # ISRO has shipped both "Latitude" and the misspelled "Lattitude".
            i_lon = cols.get("longitude")
            i_lat = cols.get("latitude", cols.get("lattitude"))
            i_px = cols.get("pixel")
            i_sc = cols.get("scan")
            if None in (i_lon, i_lat, i_px, i_sc):
                raise ValueError("unexpected geometry CSV header in %s: %r" % (path, header))
            for row in rdr:
                if len(row) <= max(i_lon, i_lat, i_px, i_sc):
                    continue
                lon_l.append(float(row[i_lon]))
                lat_l.append(float(row[i_lat]))
                px_l.append(int(float(row[i_px])))
                sc_l.append(int(float(row[i_sc])))

        px = np.asarray(px_l, np.int64)
        sc = np.asarray(sc_l, np.int64)
        upx = np.unique(px)
        usc = np.unique(sc)
        if len(upx) * len(usc) != len(px):
            raise ValueError(
                "geometry CSV in %s is not a complete rectangular lattice "
                "(%d unique pixels x %d unique scans != %d rows)"
                % (path, len(upx), len(usc), len(px)))

        ix = np.searchsorted(upx, px)
        iy = np.searchsorted(usc, sc)
        LON = np.empty((len(usc), len(upx)), np.float64)
        LAT = np.empty_like(LON)
        LON[iy, ix] = np.asarray(lon_l, np.float64)
        LAT[iy, ix] = np.asarray(lat_l, np.float64)
        X, Y = lonlat_to_south_stereo(LON, LAT)
        return cls(upx.astype(np.float64), usc.astype(np.float64), LON, LAT, X, Y,
                   source=path)

    @classmethod
    def from_envi_loc(cls, path: str, samples: int, lines: int, step: int = 0
                      ) -> "GeometryGrid":
        """Lattice from an IIRS `_loc_` backplane.

        IIRS does not ship the tabular geometry CSV the framing cameras do. It
        ships a band-sequential ENVI cube of per-pixel backplanes whose first
        two bands are Longitude and Latitude in the MOON_ME frame, at the full
        detector grid - 250 x 12945 here, 3.2 million points.

        That is far finer than anything downstream needs (the interpolators
        subsample to roughly 200 rows), and projecting three million points to
        stereographic to build it is pure cost, so it is decimated on the way in
        and the decimation is recorded in `source`.
        """
        step = step or max(1, lines // 2000)
        cube = np.memmap(path, dtype="<f4", mode="r", shape=(4, lines, samples))
        LON = np.asarray(cube[0, ::step, :], np.float64)
        LAT = np.asarray(cube[1, ::step, :], np.float64)
        del cube
        sc = np.arange(0, lines, step, dtype=np.float64)[:LON.shape[0]]
        px = np.arange(samples, dtype=np.float64)
        # A backplane carries fill where the geometry could not be solved.
        bad = ~np.isfinite(LON) | ~np.isfinite(LAT) | (np.abs(LAT) > 90.0)
        if bad.any():
            LON = np.where(bad, np.nan, LON)
            LAT = np.where(bad, np.nan, LAT)
        X, Y = lonlat_to_south_stereo(LON, LAT)
        return cls(px, sc, LON, LAT, X, Y,
                   source="%s (every %d lines)" % (path, step))

    # -------------------------------------------------------------- interpolate
    def _weights(self, sample, line):
        s = np.atleast_1d(np.asarray(sample, np.float64))
        l = np.atleast_1d(np.asarray(line, np.float64))
        i = np.clip(np.searchsorted(self.pixels, s) - 1, 0, len(self.pixels) - 2)
        j = np.clip(np.searchsorted(self.scans, l) - 1, 0, len(self.scans) - 2)
        dx = self.pixels[i + 1] - self.pixels[i]
        dy = self.scans[j + 1] - self.scans[j]
        fx = np.where(dx > 0, (s - self.pixels[i]) / np.where(dx > 0, dx, 1), 0.0)
        fy = np.where(dy > 0, (l - self.scans[j]) / np.where(dy > 0, dy, 1), 0.0)
        return i, j, fx, fy

    def _bilinear(self, M, sample, line):
        i, j, fx, fy = self._weights(sample, line)
        return (M[j, i] * (1 - fx) * (1 - fy) + M[j, i + 1] * fx * (1 - fy)
                + M[j + 1, i] * (1 - fx) * fy + M[j + 1, i + 1] * fx * fy)

    def lonlat(self, sample, line):
        """Interpolated (lon, lat) in degrees. Scalars in, scalars out."""
        lo = self._bilinear(self.lon, sample, line)
        la = self._bilinear(self.lat, sample, line)
        if np.isscalar(sample) or np.ndim(sample) == 0:
            return float(lo[0]), float(la[0])
        return lo, la

    def stereo(self, sample, line):
        """Interpolated south-stereographic (x, y) in metres.

        Interpolating the projected coordinates directly avoids the longitude
        wrap that makes lon/lat interpolation meaningless this close to the pole.
        """
        xs = self._bilinear(self.x, sample, line)
        ys = self._bilinear(self.y, sample, line)
        if np.isscalar(sample) or np.ndim(sample) == 0:
            return float(xs[0]), float(ys[0])
        return xs, ys

    # ------------------------------------------------------------------ inverse
    def stereo_to_pixel(self, xq, yq, refine: int = 2):
        """Approximate inverse: projected metres -> (sample, line).

        Nearest lattice node, then a couple of Gauss-Newton steps on the local
        bilinear Jacobian. Good to a fraction of a lattice cell (~22-28 m),
        which is all the delivered geometry supports anyway. Returns
        (sample, line, ok) with ok=False where the query falls outside the grid.
        """
        xq = np.atleast_1d(np.asarray(xq, np.float64))
        yq = np.atleast_1d(np.asarray(yq, np.float64))
        d2 = ((self.x[None, :, :] - xq[:, None, None]) ** 2
              + (self.y[None, :, :] - yq[:, None, None]) ** 2)
        flat = d2.reshape(len(xq), -1).argmin(axis=1)
        jj, ii = np.unravel_index(flat, self.x.shape)
        s = self.pixels[ii].astype(np.float64)
        l = self.scans[jj].astype(np.float64)

        for _ in range(max(0, refine)):
            cx, cy = self.stereo(s, l)
            cx = np.atleast_1d(cx); cy = np.atleast_1d(cy)
            h = 50.0
            x_ds, y_ds = self.stereo(s + h, l)
            x_dl, y_dl = self.stereo(s, l + h)
            a11 = (np.atleast_1d(x_ds) - cx) / h
            a12 = (np.atleast_1d(x_dl) - cx) / h
            a21 = (np.atleast_1d(y_ds) - cy) / h
            a22 = (np.atleast_1d(y_dl) - cy) / h
            det = a11 * a22 - a12 * a21
            det = np.where(np.abs(det) < 1e-18, np.nan, det)
            rx, ry = xq - cx, yq - cy
            ds = (rx * a22 - ry * a12) / det
            dl = (ry * a11 - rx * a21) / det
            s = np.clip(s + np.nan_to_num(ds), self.pixels[0], self.pixels[-1])
            l = np.clip(l + np.nan_to_num(dl), self.scans[0], self.scans[-1])

        fx, fy = self.stereo(s, l)
        err = np.hypot(np.atleast_1d(fx) - xq, np.atleast_1d(fy) - yq)
        ok = err < 200.0
        if len(xq) == 1:
            return float(s[0]), float(l[0]), bool(ok[0])
        return s, l, ok

    # ---------------------------------------------------------------- footprint
    def footprint_stereo(self, step: int = 1) -> np.ndarray:
        """Grid nodes as an (N, 2) array of projected metres."""
        return np.column_stack([self.x[::step, ::step].ravel(),
                                self.y[::step, ::step].ravel()])

    def window_footprint(self, sample0, line0, width, height) -> dict:
        """Corner and centre coordinates for a pixel window."""
        corners = [(sample0, line0), (sample0 + width, line0),
                   (sample0 + width, line0 + height), (sample0, line0 + height)]
        ll = [self.lonlat(a, b) for a, b in corners]
        st = [self.stereo(a, b) for a, b in corners]
        c_lon, c_lat = self.lonlat(sample0 + width / 2.0, line0 + height / 2.0)
        c_x, c_y = self.stereo(sample0 + width / 2.0, line0 + height / 2.0)
        return {
            "corners_lon_lat": [[round(a, 6), round(b, 6)] for a, b in ll],
            "corners_stereo_m": [[round(a, 2), round(b, 2)] for a, b in st],
            "center_lon_lat": [round(c_lon, 6), round(c_lat, 6)],
            "center_stereo_m": [round(c_x, 2), round(c_y, 2)],
        }


# --------------------------------------------------------------------------- #
# footprint overlap, in the projected plane
# --------------------------------------------------------------------------- #

def occupancy(points: np.ndarray, cell_m: float) -> set:
    """Rasterise projected points onto a square grid, returning occupied cells."""
    if len(points) == 0:
        return set()
    q = np.floor(points / cell_m).astype(np.int64)
    return set(map(tuple, q))


def footprint_overlap(grid_a: GeometryGrid, grid_b: GeometryGrid,
                      cell_m: float = 100.0, step: int = 1) -> dict:
    """Mutual footprint overlap, computed by occupancy in the projected plane.

    Deliberately crude and robust: the delivered geometry is only accurate to
    metres-to-decametres, so a polygon intersection would imply a precision the
    input does not have. Cell counting at 100 m is honest about that.
    """
    A = occupancy(grid_a.footprint_stereo(step), cell_m)
    B = occupancy(grid_b.footprint_stereo(step), cell_m)
    inter = A & B
    area = cell_m * cell_m / 1e6                      # km^2 per cell
    smaller = min(len(A), len(B)) or 1
    return {
        "cell_m": cell_m,
        "area_a_km2": round(len(A) * area, 2),
        "area_b_km2": round(len(B) * area, 2),
        "area_overlap_km2": round(len(inter) * area, 2),
        "fraction_of_smaller": round(len(inter) / smaller, 4),
        "fraction_of_a": round(len(inter) / (len(A) or 1), 4),
        "fraction_of_b": round(len(inter) / (len(B) or 1), 4),
    }


def find_geometry_csv(product_dir_root: str, timestamp: str) -> str | None:
    """Locate the geometry CSV for a product timestamp under a dataset root."""
    for dirpath, _dirs, files in os.walk(os.path.join(product_dir_root, "geometry")):
        for f in files:
            if timestamp in f and f.endswith(".csv"):
                return os.path.join(dirpath, f)
    return None
