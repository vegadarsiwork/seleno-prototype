"""Local surface metres for native reference pixel displacements.

A projected map's nominal pixel size is not a ground distance everywhere: the
WAC global mosaic is simple cylindrical with a 0 degree standard parallel, so
one pixel spans 100 m north-south but only 100 cos(latitude) m east-west. At
60 degrees south a nominal "200 m" error can be a 100 m one.

Displacements are small next to the body, so each is mapped through the local
Jacobian of (pixel -> east/north metres on the reference datum). The body's
radii come from the reference CRS itself; nothing lunar is assumed here.
"""
from __future__ import annotations

import numpy as np


def _geographic(crs):
    from pyproj import CRS, Transformer

    crs = CRS.from_user_input(crs)
    geodetic = crs.geodetic_crs
    if geodetic is None:
        raise ValueError("reference CRS has no geodetic datum")
    ellipsoid = geodetic.ellipsoid
    a = float(ellipsoid.semi_major_metre)
    b = float(ellipsoid.semi_minor_metre)
    if crs.is_geographic:
        to_lonlat = None
    else:
        to_lonlat = Transformer.from_crs(crs, geodetic, always_xy=True).transform
    return crs, a, b, to_lonlat


def describe(crs):
    """Provenance for the east/north metres: which body, which radii."""
    crs, a, b, _ = _geographic(crs)
    return {"datum": crs.geodetic_crs.datum.name, "semi_major_m": a, "semi_minor_m": b,
            "frame": "local east/north tangent plane on the reference CRS datum",
            "method": "per-point Jacobian of native reference pixel centres -> "
                      "geodetic longitude/latitude, by central differences of 0.5 px"}


def ground_jacobian(crs, transform, points):
    """(n, 2, 2): east/north metres per +1 native reference pixel in x and y.

    `transform` is the reference's GDAL affine on pixel edges; centres are
    offset by 0.5 as everywhere else in the tool. Column 0 is d/dx, column 1
    d/dy. Rows where the CRS cannot be inverted are NaN, never zero.
    """
    crs, a, b, to_lonlat = _geographic(crs)
    p = np.asarray(points, np.float64).reshape(-1, 2)
    h = 0.5
    offsets = np.array([[-h, 0], [h, 0], [0, -h], [0, h], [0, 0]], np.float64)
    q = (p[:, None, :] + offsets[None]).reshape(-1, 2) + 0.5
    t = transform
    x = t.a * q[:, 0] + t.b * q[:, 1] + t.c
    y = t.d * q[:, 0] + t.e * q[:, 1] + t.f
    if to_lonlat is not None:
        x, y = to_lonlat(x, y)
    lon = np.radians(np.asarray(x, np.float64)).reshape(-1, 5)
    lat = np.radians(np.asarray(y, np.float64)).reshape(-1, 5)
    e2 = 1.0 - (b / a) ** 2
    phi = lat[:, 4]
    w = np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    east_radius = a / w * np.cos(phi)               # prime vertical, at this parallel
    north_radius = a * (1.0 - e2) / w ** 3          # meridian
    dlon_x = np.angle(np.exp(1j * (lon[:, 1] - lon[:, 0]))) / (2 * h)
    dlon_y = np.angle(np.exp(1j * (lon[:, 3] - lon[:, 2]))) / (2 * h)
    dlat_x = (lat[:, 1] - lat[:, 0]) / (2 * h)
    dlat_y = (lat[:, 3] - lat[:, 2]) / (2 * h)
    J = np.empty((len(p), 2, 2), np.float64)
    J[:, 0, 0] = east_radius * dlon_x
    J[:, 0, 1] = east_radius * dlon_y
    J[:, 1, 0] = north_radius * dlat_x
    J[:, 1, 1] = north_radius * dlat_y
    return J


def ground_errors(crs, transform, positions, errors):
    """East/north metres for reference-pixel error vectors at reference positions."""
    e = np.asarray(errors, np.float64).reshape(-1, 2)
    J = ground_jacobian(crs, transform, positions)
    return np.einsum("nij,nj->ni", J, e)
