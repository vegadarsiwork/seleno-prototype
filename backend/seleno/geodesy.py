"""Pixel -> selenographic coordinates using the OHRC geometry grid.

ISRO ships each OHRC product with a `*_g_grd_*.csv` file giving longitude and
latitude on a coarse (pixel, scan) lattice. We interpolate that lattice instead
of inventing a projection, so reported coordinates trace back to the delivered
product. Accuracy is limited by the lattice spacing and by the fact that the grid
is a system-level product; it is not a bundle-adjusted geodetic solution.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GeoGrid:
    pixels: np.ndarray      # unique sample (column) coordinates, ascending
    scans: np.ndarray       # unique line (row) coordinates, ascending
    lon: np.ndarray         # (n_scan, n_pixel)
    lat: np.ndarray         # (n_scan, n_pixel)
    source: str = ""

    @classmethod
    def from_csv(cls, path: str) -> "GeoGrid":
        px, sc, lon, lat = [], [], [], []
        with open(path, newline="") as fh:
            r = csv.DictReader(fh)
            key = {k.lower().strip(): k for k in r.fieldnames or []}
            klon = key.get("longitude")
            klat = key.get("lattitude") or key.get("latitude")
            kpx = key.get("pixel")
            ksc = key.get("scan")
            for row in r:
                px.append(float(row[kpx])); sc.append(float(row[ksc]))
                lon.append(float(row[klon])); lat.append(float(row[klat]))
        px = np.asarray(px); sc = np.asarray(sc)
        upx = np.unique(px); usc = np.unique(sc)
        LON = np.full((len(usc), len(upx)), np.nan)
        LAT = np.full((len(usc), len(upx)), np.nan)
        ix = {v: i for i, v in enumerate(upx)}
        iy = {v: i for i, v in enumerate(usc)}
        for p, s, lo, la in zip(px, sc, lon, lat):
            LON[iy[s], ix[p]] = lo
            LAT[iy[s], ix[p]] = la
        return cls(upx, usc, LON, LAT, source=path)

    def lonlat(self, sample: float, line: float) -> tuple[float, float]:
        """Bilinear interpolation of the delivered lattice."""
        def frac(vals, v):
            i = int(np.clip(np.searchsorted(vals, v) - 1, 0, len(vals) - 2))
            span = vals[i + 1] - vals[i]
            return i, float((v - vals[i]) / span) if span else 0.0

        i, fx = frac(self.pixels, sample)
        j, fy = frac(self.scans, line)

        def bil(M):
            return (M[j, i] * (1 - fx) * (1 - fy) + M[j, i + 1] * fx * (1 - fy)
                    + M[j + 1, i] * (1 - fx) * fy + M[j + 1, i + 1] * fx * fy)

        return float(bil(self.lon)), float(bil(self.lat))

    def footprint(self, x0, y0, w, h) -> dict:
        """Corner coordinates of a pixel window, plus its centre."""
        corners = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        ll = [self.lonlat(x, y) for x, y in corners]
        c = self.lonlat(x0 + w / 2.0, y0 + h / 2.0)
        return {
            "corners_lon_lat": [[round(a, 6), round(b, 6)] for a, b in ll],
            "center_lon_lat": [round(c[0], 6), round(c[1], 6)],
        }


MOON_RADIUS_M = 1737400.0


def ground_distance_m(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance on a spherical Moon (adequate at these scales)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    dp = p2 - p1
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * MOON_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))
