"""Terrain parallax: a height-proportional term in the registration model.

An orthorectified reference (a NAC or SELENE map) puts every ground point where
it is. A Level-1 source image does not: a point `h` metres above the reference
sphere, seen `theta` off nadir, lands `h * tan(theta)` along the look direction
from where the geometry lattice (computed on the sphere) says. At the south
pole, with a kilometre of relief and OHRC rolled 14.6 degrees, that is tens of
metres - many reference pixels - and it follows the terrain, not position, so
neither an affine nor a smooth field can remove it. Measured on the OHRC/NAC
pair: one extra height coefficient took the held-out median from 1.73 to 0.20
working pixels.

The term is `m * h(q)`: `m` (source working px per metre, fitted) times the DEM
height at the reference position `q`. The height is taken where the ground
truly is, in the orthorectified reference, so the inverse mapping stays
explicit. Whether the term is used at all is decided by spatial
cross-validation over fit cells, as for the local field.

The sampler is rebuilt from a small serialisable `spec`, so transform.json
carries everything needed to apply the model again.
"""
from __future__ import annotations

import os

import numpy as np

_SAMPLERS: dict = {}
_LOLA_CRS = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"


def _root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def candidates(root=None):
    """LOLA polar DEMs present in the archive, finest first."""
    root = root or _root()
    out = []
    for stem in ("ldem_875s_5m", "ldem_80s_20m"):
        p = os.path.join(root, "data", "raw", "lola", stem + ".lbl")
        if os.path.exists(p) and os.path.exists(p[:-4] + ".img"):
            out.append(p)
    return out


class HeightSampler:
    """Heights (m above the 1737.4 km sphere) at reference working-grid points."""

    def __init__(self, spec):
        from ..dem import LolaDEM
        from pyproj import Transformer
        self.spec = spec
        self.dem = LolaDEM.open(spec["dem"])
        self.C = np.asarray(spec["grid_to_reference"], np.float64)
        t = spec["reference_transform"]
        self.affine = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]]], np.float64)
        self.to_dem = Transformer.from_crs(spec["reference_crs"], _LOLA_CRS, always_xy=True)

    def __call__(self, points):
        p = np.asarray(points, np.float64).reshape(-1, 2)
        out = np.full(len(p), np.nan)
        good = np.isfinite(p).all(axis=1)
        if not good.any():
            return out
        q = p[good] @ self.C[:2, :2].T + self.C[:2, 2]            # native reference px
        w = np.column_stack([q + 0.5, np.ones(len(q))]) @ self.affine.T   # GDAL: centre + .5
        x, y = self.to_dem.transform(w[:, 0], w[:, 1])
        d = self.dem
        # Pixel registration: the value of column c describes the cell centred
        # at x = (c + 0.5 - samples / 2) * res.
        col = np.asarray(x) / d.res_m + d.samples / 2.0 - 0.5
        row = d.lines / 2.0 - np.asarray(y) / d.res_m - 0.5
        inside = (col >= 0) & (row >= 0) & (col <= d.samples - 1) & (row <= d.lines - 1)
        h = np.full(len(q), np.nan)
        if inside.any():
            c, r = col[inside], row[inside]
            c0, r0 = int(np.floor(c.min())), int(np.floor(r.min()))
            c1 = min(d.samples - 1, int(np.floor(c.max())) + 1)
            r1 = min(d.lines - 1, int(np.floor(r.max())) + 1)
            blk = np.asarray(d._map[r0:r1 + 1, c0:c1 + 1], np.float32) * d.scale + d.offset - 1737400.0
            cc, rr = c - c0, r - r0
            i0 = np.clip(np.floor(cc).astype(np.int64), 0, max(0, blk.shape[1] - 2))
            j0 = np.clip(np.floor(rr).astype(np.int64), 0, max(0, blk.shape[0] - 2))
            i1, j1 = np.minimum(i0 + 1, blk.shape[1] - 1), np.minimum(j0 + 1, blk.shape[0] - 1)
            fx, fy = cc - i0, rr - j0
            h[inside] = (blk[j0, i0] * (1 - fx) * (1 - fy) + blk[j0, i1] * fx * (1 - fy)
                         + blk[j1, i0] * (1 - fx) * fy + blk[j1, i1] * fx * fy)
        out[good] = h
        return out


def sampler(spec):
    key = repr(sorted((k, repr(v)) for k, v in spec.items()))
    if key not in _SAMPLERS:
        _SAMPLERS[key] = HeightSampler(spec)
    return _SAMPLERS[key]


def spec_for(reference, frame, root=None):
    """A sampler spec for a georeferenced reference whose overlap a DEM covers."""
    if not getattr(reference, "georeferenced", False) or reference.crs is None:
        return None
    from .coordinates import grid_to_reference
    C = grid_to_reference(frame)
    h, w = frame["target_shape"]
    probe = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [(w - 1) / 2, (h - 1) / 2]], float)
    for dem in candidates(root):
        spec = {"dem": dem, "reference_crs": reference.crs.to_wkt(),
                "reference_transform": list(reference.transform)[:6],
                "grid_to_reference": C.tolist()}
        try:
            heights = sampler(spec)(probe)
        except Exception:                                             # noqa: BLE001
            continue
        if np.isfinite(heights).all():
            return spec
    return None


def term(parallax, points):
    """Source-side displacement of the parallax term at reference points."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    frame = parallax.get("frame")
    F = None
    if frame is not None:
        F = np.asarray(frame, np.float64)
        Fi = np.linalg.inv(F)
        p = p @ Fi[:2, :2].T + Fi[:2, 2]
    h = sampler(parallax["sampler"])(p)
    out = np.nan_to_num(h - parallax["height_origin_m"])[:, None] * np.asarray(
        parallax["coefficient_px_per_m"], np.float64)[None, :]
    if F is not None:
        out = out @ F[:2, :2].T
    return out
