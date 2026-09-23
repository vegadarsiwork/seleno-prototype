"""Find where one image lies inside the other before registering them.

The registration stages refine a placement; they do not search for one. With a
map-projected reference the placement comes from geometry. Without one - two
Chandrayaan-2 products carry per-pixel lon/lat lattices and no CRS - the tool
fell back to assuming both images start at the same corner. An OHRC frame is
2.6 x 22 km and an IIRS strip 14 x 700 km, so that assumption put the OHRC
frame at the top of the strip, the matchers compared the wrong ground, and a
pair 500 km apart was reported as overlapping 100% and then refused for
`insufficient_matches` instead of `no_overlap`.

`locate` produces the placement in two layers:

1. **Geometry**, whenever both inputs carry any (a lon/lat lattice or a CRS):
   the smaller footprint is projected into the larger image and fitted with a
   similarity. Geometry proves disjointness, and it narrows the search, but it
   is not trusted for the final position: unrefined polar OHRC is 4-5 km out.
2. **Image search**: the smaller footprint, reduced to the coarser of the two
   ground samplings by area averaging, is slid over the larger image at a set
   of rotations with masked normalised cross-correlation, on intensity and on
   gradient magnitude. The best position is accepted only if it is clearly
   better than the best position ELSEWHERE; a search that finds two similar
   places says so and the caller falls back rather than guessing.

The result is a similarity from reference pixels to source pixels (both
full-resolution, pixel centre on the integer), which `prealign` applies in
place of its pixel-space assumption.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

MOON_R = 1737400.0
# Search budget: the larger image is reduced until it fits, and a template
# smaller than this on its short side carries too little to locate anything.
_SEARCH_MAX_PX = 6_000_000
_MIN_TEMPLATE_SIDE = 12
# Acceptance. Scores are masked normalised cross-correlation (1.0 is perfect),
# and raw gaps between them mean little: a mostly-shadowed OHRC frame scores
# 0.87 against IIRS ground it is NOT on, because one bright ridge correlates
# with any ridge. Two scale-free tests are used instead, both against the best
# peak at a DIFFERENT place:
#   normalised margin  (best - rival) / (1 - rival): the share of the remaining
#                      headroom the winner closes;
#   robust z           how far the winner stands above all the other places'
#                      peaks, in robust standard deviations.
# Calibrated on real OHRC reduced to IIRS sampling (see reports/LOCATE.md):
# true placements 0.42-0.96 and z 5.1-7.6, including a 67 deg Sun-azimuth
# change; every negative at most 0.12 and z 3.2.
_MIN_SCORE = 0.25
_MIN_NORM_MARGIN = 0.30
_MIN_Z = 4.0
# Intensity and gradient magnitude are two views of the same ground. A true
# placement is found by both (they agreed on every real positive); a false one
# rarely is. One cue alone must clear this much stronger bar instead.
_SOLO_NORM_MARGIN = 0.50
_SOLO_Z = 6.0
# The reduced search level must keep this much template, or its statistics are
# noise: a 15 x 15 px template passed the single-cue bar on unrelated ground.
_MIN_LEVEL_AREA = 1500


class Disjoint(Exception):
    """The geometry of both inputs places them too far apart to overlap."""


@dataclass
class Placement:
    """Reference pixel -> source pixel, and where to look in the reference."""

    T_r2s: np.ndarray                     # 3x3, full-res, pixel centre on the integer
    method: str                           # "geometry", "search", "geometry+search"
    window: tuple | None = None           # reference (r0, c0, r1, c1) or None
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# geometry adapters: a lattice or a CRS, behind one interface
# --------------------------------------------------------------------------- #

class _Geo:
    """Pixel <-> lon/lat for one scene, whichever kind of geometry it carries."""

    def __init__(self, scene):
        self.sc = scene
        self.kind = ("lattice" if scene.lonlat is not None
                     else "crs" if scene.georeferenced else None)
        self._interp = {}

    def to_lonlat(self, s, l):
        s = np.asarray(s, float)
        l = np.asarray(l, float)
        if self.kind == "lattice":
            from ..ohrc.geometry import south_stereo_to_lonlat
            # Interpolate in the projected plane, then convert: interpolating
            # longitude directly breaks across the 0/360 seam at the pole.
            x, y = self.sc.lonlat.stereo(s, l)
            return south_stereo_to_lonlat(x, y)
        t = self.sc.transform
        # rasterio's affine maps pixel EDGES; the centre of pixel i is i + 0.5
        X = t.a * (s + 0.5) + t.b * (l + 0.5) + t.c
        Y = t.d * (s + 0.5) + t.e * (l + 0.5) + t.f
        if self.sc.crs.is_geographic:
            return X, Y
        from pyproj import Transformer
        tr = Transformer.from_crs(self.sc.crs.to_wkt(), "+proj=longlat +R=%g +no_defs" % MOON_R,
                                  always_xy=True)
        return tr.transform(X, Y)

    def from_plane(self, X, Y, plane: str):
        """Pixel (sample, line) of points given in `plane` metres; NaN outside."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        from pyproj import Transformer
        if self.kind == "lattice":
            f = self._interp.get(plane)
            if f is None:
                g = self.sc.lonlat
                st = max(1, g.x.shape[0] // 200)
                ss, ll = np.meshgrid(g.pixels, g.scans[::st])
                lon, lat = self.to_lonlat(ss.ravel(), ll.ravel())
                ok = np.isfinite(lon) & np.isfinite(lat)
                px, py = Transformer.from_crs("+proj=longlat +R=%g +no_defs" % MOON_R, plane,
                                              always_xy=True).transform(lon[ok], lat[ok])
                from scipy.interpolate import LinearNDInterpolator
                pts = np.column_stack([px, py])
                f = (LinearNDInterpolator(pts, ss.ravel()[ok]),
                     LinearNDInterpolator(pts, ll.ravel()[ok]))
                self._interp[plane] = f
            return f[0](X, Y), f[1](X, Y)
        lon, lat = Transformer.from_crs(plane, "+proj=longlat +R=%g +no_defs" % MOON_R,
                                        always_xy=True).transform(X, Y)
        t = self.sc.transform
        if self.sc.crs.is_geographic:
            cx, cy = lon, lat
        else:
            cx, cy = Transformer.from_crs("+proj=longlat +R=%g +no_defs" % MOON_R,
                                          self.sc.crs.to_wkt(), always_xy=True).transform(lon, lat)
        inv = ~t
        s = inv.a * np.asarray(cx) + inv.b * np.asarray(cy) + inv.c - 0.5
        l = inv.d * np.asarray(cx) + inv.e * np.asarray(cy) + inv.f - 0.5
        h, w = self.sc.array.shape
        bad = (s < -0.5) | (l < -0.5) | (s > w - 0.5) | (l > h - 0.5)
        return np.where(bad, np.nan, s), np.where(bad, np.nan, l)


def _plane_at(lon, lat) -> str:
    """A stereographic plane centred on the footprint: continuous everywhere,
    the poles included, and metric enough over a few hundred km."""
    return "+proj=stere +lat_0=%.8f +lon_0=%.8f +k=1 +R=%g +units=m +no_defs" % (lat, lon, MOON_R)


def _to_plane(lon, lat, plane):
    from pyproj import Transformer
    return Transformer.from_crs("+proj=longlat +R=%g +no_defs" % MOON_R, plane,
                                always_xy=True).transform(np.asarray(lon, float),
                                                          np.asarray(lat, float))


def _similarity(src, dst):
    """Least-squares similarity src -> dst as a 3x3 matrix, and its RMS residual."""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32),
                                       method=cv2.LMEDS)
    if M is None:
        return None, None
    T = np.vstack([M, [0, 0, 1]]).astype(float)
    pred = src @ T[:2, :2].T + T[:2, 2]
    return T, float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))


def geometry_prior(small, big, radius_m: float):
    """Where `small`'s footprint lands in `big`, from their geometry alone.

    Returns None when either input has no geometry. Raises `Disjoint` when the
    footprints are further apart than `radius_m` (the plausible geolocation
    error), which is a finding in its own right: no amount of image matching
    should be asked to join two places hundreds of kilometres apart.
    """
    gs, gb = _Geo(small), _Geo(big)
    if gs.kind is None or gb.kind is None:
        return None
    h, w = small.array.shape
    ss, ll = np.meshgrid(np.linspace(0, w - 1, 9), np.linspace(0, h - 1, 33))
    ss, ll = ss.ravel(), ll.ravel()
    lon, lat = gs.to_lonlat(ss, ll)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    good = np.isfinite(lon) & np.isfinite(lat)
    if good.sum() < 3:
        return None
    clon, clat = gs.to_lonlat(np.array([(w - 1) / 2.0]), np.array([(h - 1) / 2.0]))
    plane = _plane_at(float(np.atleast_1d(clon)[0]), float(np.atleast_1d(clat)[0]))
    X, Y = _to_plane(lon[good], lat[good], plane)
    bs, bl = gb.from_plane(X, Y, plane)
    bs, bl = np.asarray(bs, float), np.asarray(bl, float)
    ok = np.isfinite(bs) & np.isfinite(bl)

    # The big footprint, in the same plane, to measure distance if they miss.
    H, W = big.array.shape
    BS, BL = np.meshgrid(np.linspace(0, W - 1, 17), np.linspace(0, H - 1, 257))
    blon, blat = gb.to_lonlat(BS.ravel(), BL.ravel())
    bgood = np.isfinite(blon) & np.isfinite(blat)
    BX, BY = _to_plane(np.asarray(blon)[bgood], np.asarray(blat)[bgood], plane)
    d = np.sqrt((np.asarray(BX)[None, :] - np.asarray(X)[:, None]) ** 2
                + (np.asarray(BY)[None, :] - np.asarray(Y)[:, None]) ** 2).min()

    info = {"inside_fraction": round(float(ok.mean()), 4),
            "centre_lon_lat": [round(float(np.atleast_1d(clon)[0]), 5),
                               round(float(np.atleast_1d(clat)[0]), 5)],
            "gap_m": round(float(d), 1)}
    if ok.sum() < 3:
        if d > radius_m:
            raise Disjoint(
                "by their own geometry the two footprints are %.1f km apart at the "
                "closest (the smaller is centred at lon %.3f, lat %.3f); that is beyond "
                "the %.1f km any geolocation error here could explain"
                % (d / 1000.0, info["centre_lon_lat"][0], info["centre_lon_lat"][1],
                   radius_m / 1000.0))
        return {"T": None, "info": info}
    T, resid = _similarity(np.column_stack([ss[good][ok], ll[good][ok]]),
                           np.column_stack([bs[ok], bl[ok]]))
    if T is None:
        return {"T": None, "info": info}
    info.update({"residual_px": round(resid, 2),
                 "scale": round(float(math.hypot(T[0, 0], T[1, 0])), 5),
                 "rotation_deg": round(float(math.degrees(math.atan2(T[1, 0], T[0, 0]))), 2)})
    return {"T": T, "info": info}


# --------------------------------------------------------------------------- #
# coarse, area-averaged versions of each image
# --------------------------------------------------------------------------- #

def _base(scene, cache_dir):
    """The cheapest faithful source for a reduction: the array itself when it
    is small, otherwise the block-mean overview the viewer caches per file."""
    h, w = scene.array.shape
    if h * w <= 16_000_000:
        a = np.array(scene.array[0:h, 0:w], dtype=np.float32)
        return a, 1.0, 1.0
    if not cache_dir:
        return None
    from . import view as V
    v = V.open_view(scene.path, cache_dir)
    if v.overview is None:
        a = np.array(scene.array[0:h, 0:w], dtype=np.float32)
        return a, 1.0, 1.0
    ov = np.asarray(v.overview, np.float32)
    return ov, h / float(ov.shape[0]), w / float(ov.shape[1])


def _reduce(base, fy_base, fx_base, box, factor):
    """`box` (r0, c0, r1, c1) of the full-resolution image, area-averaged to
    about `factor` full-res pixels per output pixel.

    Returns ``(img, valid, sy, sx, (r_origin, c_origin))``: the exact full-res
    pixels per output pixel and the full-res position of the output's top-left
    edge. When the base is an overview its blocks do not line up with `box`,
    so both are taken from the block edges actually read, not from `box`.
    """
    import cv2
    r0, c0, r1, c1 = box
    br0, bc0 = int(r0 // fy_base), int(c0 // fx_base)
    br1 = min(base.shape[0], int(math.ceil(r1 / fy_base)))
    bc1 = min(base.shape[1], int(math.ceil(c1 / fx_base)))
    b = base[br0:br1, bc0:bc1]
    rows, cols = (br1 - br0) * fy_base, (bc1 - bc0) * fx_base
    oh = max(1, int(round(rows / factor)))
    ow = max(1, int(round(cols / factor)))
    valid = np.isfinite(b)
    fill = float(np.nanmean(b)) if valid.any() else 0.0
    img = cv2.resize(np.where(valid, b, fill).astype(np.float32), (ow, oh),
                     interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA) > 0.5
    return img, vm, rows / float(oh), cols / float(ow), (br0 * fy_base, bc0 * fx_base)


def _representations(img, valid):
    """Intensity and gradient magnitude, each normalised over the valid area.
    Gradient magnitude survives a contrast inversion that intensity does not."""
    import cv2
    v = img[valid]
    lo, hi = (np.percentile(v, (2, 98)) if v.size > 16 else (0.0, 1.0))
    a = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)
    a[~valid] = float(np.mean(a[valid])) if valid.any() else 0.0
    b = cv2.GaussianBlur(a, (0, 0), 1.0)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    gv = g[valid]
    if gv.size > 16:
        g = np.clip(g / max(np.percentile(gv, 99), 1e-9), 0, 1)
    g[~valid] = float(np.mean(g[valid])) if valid.any() else 0.0
    return {"intensity": a, "gradient": g.astype(np.float32)}


def _rotated(t, m, deg):
    """Template and mask rotated by `deg` about their centre, on a canvas just
    large enough, plus the 3x3 map template coords -> canvas coords."""
    import cv2
    h, w = t.shape
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float) - [(w - 1) / 2.0, (h - 1) / 2.0]
    rc = corners @ R.T
    cw = int(math.ceil(rc[:, 0].max() - rc[:, 0].min())) + 1
    ch = int(math.ceil(rc[:, 1].max() - rc[:, 1].min())) + 1
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.array([(cw - 1) / 2.0, (ch - 1) / 2.0]) - R @ np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    tr = cv2.warpAffine(t, M[:2], (cw, ch), flags=cv2.INTER_LINEAR, borderValue=0)
    mr = cv2.warpAffine(m.astype(np.float32), M[:2], (cw, ch), flags=cv2.INTER_NEAREST,
                        borderValue=0)
    mr = cv2.erode(mr, np.ones((3, 3), np.uint8)) > 0.5
    return tr, mr.astype(np.float32), M


def _peaks(res, k, radius):
    """Top-k local maxima of a score map, at least `radius` apart."""
    r = res.copy()
    out = []
    for _ in range(k):
        j, i = np.unravel_index(int(np.argmax(r)), r.shape)
        v = float(r[j, i])
        if not np.isfinite(v) or v <= -1:
            break
        out.append((v, i, j))
        rr = int(math.ceil(radius))
        r[max(0, j - rr):j + rr + 1, max(0, i - rr):i + rr + 1] = -2
    return out


def _scan(tmpl, tmask, image, angles, pad, per_angle=4):
    """Masked NCC of `tmpl` over `image` at each angle.

    Returns peaks as ``(score, cx, cy, deg, M, (ox, oy))``: the template centre
    in `image` coordinates, the rotation, the template -> canvas map and the
    canvas origin in `image` coordinates. `image` is padded by `pad` with its
    mean so that a footprint hanging partly off the edge can still be found.
    """
    import cv2
    padded = cv2.copyMakeBorder(image, pad[1], pad[1], pad[0], pad[0], cv2.BORDER_CONSTANT,
                                value=float(image.mean()))
    radius = max(3.0, 0.25 * min(tmpl.shape))
    peaks = []
    for deg in angles:
        tr, mr, M = _rotated(tmpl, tmask, deg)
        ch, cw = tr.shape
        if ch > padded.shape[0] or cw > padded.shape[1] or mr.sum() < 16:
            continue
        res = cv2.matchTemplate(padded, tr, cv2.TM_CCOEFF_NORMED, mask=mr)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        for v, i, j in _peaks(res, per_angle, radius):
            ox, oy = i - pad[0], j - pad[1]
            peaks.append((v, ox + (cw - 1) / 2.0, oy + (ch - 1) / 2.0, deg, M, (ox, oy)))
    peaks.sort(key=lambda p: -p[0])
    return peaks


def _distinct(p, best, shape):
    """Whether peak `p` is a different PLACE from `best`, not the same one
    nudged. Measured in the winner's own frame: an elongated footprint slid a
    fifth of its length along itself still covers mostly the same ground, and
    counting that as a rival would refuse every strip on smooth terrain."""
    h, w = shape
    th = math.radians(best[3])
    dx, dy = p[1] - best[1], p[2] - best[2]
    u = math.cos(th) * dx + math.sin(th) * dy
    v = -math.sin(th) * dx + math.cos(th) * dy
    return abs(u) > 0.5 * w or abs(v) > 0.5 * h


def _search(tmpl, tmask, image, angles, pad, angle_step):
    """Coarse-to-fine: every angle at a reduced level, then the winner refined
    at full level in a small crop around it.

    Returns ``(best, rival_score, coarse_best_score, robust_z)`` with `best` in
    full-level coordinates, or all None. The accept/refuse decision uses the
    reduced level, where best and rival were measured on the same footing.
    """
    import cv2
    k = 1
    while (k < 4 and min(tmpl.shape) / (2 * k) >= _MIN_TEMPLATE_SIDE
           and tmpl.size / (2 * k) ** 2 >= _MIN_LEVEL_AREA):
        k *= 2
    if k > 1:
        def dn(a, interp=cv2.INTER_AREA):
            return cv2.resize(a, (max(1, a.shape[1] // k), max(1, a.shape[0] // k)),
                              interpolation=interp)
        t1, m1, b1 = dn(tmpl), (dn(tmask) > 0.5).astype(np.float32), dn(image)
    else:
        t1, m1, b1 = tmpl, tmask, image
    peaks = _scan(t1, m1, b1, angles, (pad[0] // k, pad[1] // k))
    if not peaks:
        return None, None, None, None
    best1 = peaks[0]
    others = np.array([p[0] for p in peaks[1:] if _distinct(p, best1, t1.shape)])
    rival_score = float(others[0]) if others.size else -1.0
    z = None
    if others.size >= 8:
        med = float(np.median(others))
        mad = 1.4826 * float(np.median(np.abs(others - med)))
        z = (best1[0] - med) / max(mad, 1e-6)

    # refine position and angle at the full level, near the coarse winner
    cx, cy = (best1[1] + 0.5) * k - 0.5, (best1[2] + 0.5) * k - 0.5
    half = int(math.ceil(0.5 * math.hypot(*tmpl.shape))) + 3 * k + 2
    H, W = image.shape
    x0, y0 = int(math.floor(cx)) - half, int(math.floor(cy)) - half
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    fill = float(image.mean())
    crop = np.full((y1 - y0, x1 - x0), fill, np.float32)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    fine_angles = [best1[3] + d * 0.5 for d in range(-int(angle_step), int(angle_step) + 1)]
    fp = _scan(tmpl, tmask, crop, fine_angles, (0, 0), per_angle=1)
    if not fp:
        return None, None, None, None
    v, fcx, fcy, deg, M, (ox, oy) = fp[0]
    best = (v, fcx + x0, fcy + y0, deg, M, (ox + x0, oy + y0))
    return best, rival_score, best1[0], z


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #

def _ground_area(sc):
    h, w = sc.array.shape
    return h * w * (sc.gsd_m or 0.0) ** 2


def wanted(S, R, mode: str = "auto"):
    """Whether to locate, and why. Only where no CRS route places the source."""
    if mode == "off":
        return False, "disabled"
    if R.georeferenced and (S.lonlat is not None or S.georeferenced):
        return False, "the reference is map-projected; geometry places the source"
    if mode == "force":
        return True, "forced"
    if S.lonlat is not None and R.lonlat is not None:
        return True, "both inputs carry lon/lat lattices but no map projection"
    if (_Geo(S).kind is not None) and (_Geo(R).kind is not None):
        return True, "both inputs carry geometry but no shared map projection"
    if S.gsd_m and R.gsd_m:
        a, b = _ground_area(S), _ground_area(R)
        if min(a, b) < 0.5 * max(a, b):
            return True, ("the footprints differ %.1fx in area, so they cannot be assumed "
                          "to share a corner" % (max(a, b) / max(min(a, b), 1e-9)))
    return False, "footprints are comparable and there is no geometry to consult"


def locate(S, R, cache_dir: str, profiles=None, mode: str = "auto", log=None):
    """Placement of the source in the reference, or None to keep the old behaviour.

    Raises `Disjoint` when geometry proves the two cannot overlap.
    """
    say = log or (lambda *_: None)
    go, why = wanted(S, R, mode)
    if not go:
        return None
    t0 = time.time()
    info = {"why": why}

    # Which footprint is smaller decides which one is searched for.
    small_is_src = _ground_area(S) <= _ground_area(R) if (S.gsd_m and R.gsd_m) \
        else S.array.size <= R.array.size
    small, big = (S, R) if small_is_src else (R, S)
    info["searched_for"] = "source" if small_is_src else "reference"

    err = 0.0
    if profiles is not None:
        for sc in (S, R):
            try:
                err = max(err, float(profiles.get(sc.profile).get("expected_geolocation_error_m") or 0))
            except Exception:                                         # noqa: BLE001
                pass
    radius_m = 1.5 * max(err, 3000.0)
    info["search_radius_m"] = radius_m

    prior = geometry_prior(small, big, radius_m)     # may raise Disjoint
    T_geo = prior["T"] if prior else None
    if prior:
        info["geometry"] = prior["info"]

    # ---- image search ------------------------------------------------------
    T_img = None
    g_small = small.gsd_m or 1.0
    g_big = big.gsd_m or 1.0
    if not (small.gsd_m and big.gsd_m) and T_geo is not None:
        # no declared GSDs: the geometry's own scale is the ratio
        g_small = g_big * float(math.hypot(T_geo[0, 0], T_geo[1, 0]))
    Hb, Wb = big.array.shape
    box = (0, 0, Hb, Wb)
    angles = list(range(0, 360, 6))
    if T_geo is not None:
        # Search the prior's neighbourhood only: the footprint's projected box,
        # grown by the geolocation radius, at rotations near the prior's.
        hs, ws = small.array.shape
        cs = np.array([[0, 0, 1], [ws - 1, 0, 1], [0, hs - 1, 1], [ws - 1, hs - 1, 1]], float)
        pc = cs @ T_geo.T
        grow = radius_m / g_big
        box = (int(max(0, pc[:, 1].min() - grow)), int(max(0, pc[:, 0].min() - grow)),
               int(min(Hb, pc[:, 1].max() + grow)), int(min(Wb, pc[:, 0].max() + grow)))
        rot0 = math.degrees(math.atan2(T_geo[1, 0], T_geo[0, 0]))
        angles = [rot0 + d for d in range(-12, 13, 3)]
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        box = (0, 0, Hb, Wb)

    g = max(g_small, g_big)
    area_px = (box[2] - box[0]) * (box[3] - box[1]) * (g_big / g) ** 2
    if area_px > _SEARCH_MAX_PX:
        g *= math.sqrt(area_px / _SEARCH_MAX_PX)
    try:
        sbase = _base(small, cache_dir)
        bbase = _base(big, cache_dir)
        if sbase is None or bbase is None:
            raise ValueError("no overview cache for a large input")
        # never reduce below what the base (an overview, perhaps) already is
        g = max(g, g_small * max(sbase[1], sbase[2]), g_big * max(bbase[1], bbase[2]))
        hs, ws = small.array.shape
        timg, tval, tsy, tsx, _ = _reduce(*sbase, (0, 0, hs, ws), g / g_small)
        bimg, bval, bsy, bsx, (bor, boc) = _reduce(*bbase, box, g / g_big)
    except Exception as exc:                                          # noqa: BLE001
        info["search"] = "skipped: could not reduce the inputs (%s: %s)" % (type(exc).__name__, exc)
        timg = None

    if timg is not None and min(timg.shape) < _MIN_TEMPLATE_SIDE:
        info["search"] = ("skipped: at the common %.1f m sampling the smaller image is only "
                          "%d x %d px, too little to locate" % (g, timg.shape[1], timg.shape[0]))
        timg = None
    if timg is not None:
        info["common_gsd_m"] = round(g, 3)
        info["template_px"] = [int(timg.shape[1]), int(timg.shape[0])]
        info["search_area_px"] = [int(bimg.shape[1]), int(bimg.shape[0])]
        pad = (int(0.5 * max(timg.shape)), int(0.5 * max(timg.shape)))
        step = (angles[1] - angles[0]) if len(angles) > 1 else 3
        trep = _representations(timg, tval)
        brep = _representations(bimg, bval)
        found = {}
        for name in ("gradient", "intensity"):
            best, rival, coarse, z = _search(trep[name], tval.astype(np.float32),
                                             brep[name], angles, pad, step)
            if best is None:
                continue
            found[name] = {"score": round(coarse, 4), "runner_up": round(rival, 4),
                           "norm_margin": round((coarse - rival) / max(1.0 - rival, 1e-6), 4),
                           "robust_z": None if z is None else round(z, 2),
                           "refined_score": round(best[0], 4),
                           "angle_deg": round(best[3], 2), "best": best}
        # A z-score needs other places to compare against. When the template
        # nearly fills the search area (two strips of the same ground) there
        # are none, and that absence is not evidence against the match.
        ok = {k: v for k, v in found.items()
              if v["score"] >= _MIN_SCORE and v["norm_margin"] >= _MIN_NORM_MARGIN
              and (v["robust_z"] is None or v["robust_z"] >= _MIN_Z)}
        info["search"] = {k: {kk: vv for kk, vv in v.items() if kk != "best"}
                          for k, v in found.items()}
        name = max(ok, key=lambda k: ok[k]["norm_margin"]) if ok else None
        if name is not None:
            other = next((v for k, v in found.items() if k != name), None)
            b0 = ok[name]["best"]
            agree = bool(other) and not _distinct(other["best"], b0, timg.shape) and \
                abs((other["best"][3] - b0[3] + 180.0) % 360.0 - 180.0) <= 10.0
            z = ok[name]["robust_z"]
            solo = (ok[name]["norm_margin"] >= _SOLO_NORM_MARGIN
                    and z is not None and z >= _SOLO_Z)
            if z is None:
                # without a z-score, both cues must agree AND both clear the
                # strong margin: agreement is the only independent check left
                agree = agree and all(v["norm_margin"] >= _SOLO_NORM_MARGIN
                                      for v in found.values())
            info["cues_agree"] = agree
            if not (agree or solo):
                info["verdict"] = ("no unique match: %s found a place the other cue does not "
                                   "confirm" % name)
                name = None
        if name is not None:
            v, cx, cy, deg, M, (ox, oy) = ok[name]["best"]
            # small full-res -> small coarse
            C1 = np.array([[1 / tsx, 0, 0.5 / tsx - 0.5], [0, 1 / tsy, 0.5 / tsy - 0.5], [0, 0, 1]])
            # coarse template -> canvas -> big coarse (canvas origin at ox, oy)
            Sh = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], float)
            # big coarse -> big full-res
            C2 = np.array([[bsx, 0, 0.5 * bsx - 0.5 + boc],
                           [0, bsy, 0.5 * bsy - 0.5 + bor], [0, 0, 1]])
            T_img = C2 @ Sh @ M @ C1
            info["chosen"] = name
            if T_geo is not None:
                c = np.array([(small.array.shape[1] - 1) / 2.0, (small.array.shape[0] - 1) / 2.0, 1.0])
                info["geometry_error_m"] = round(float(
                    np.hypot(*((T_img @ c)[:2] - (T_geo @ c)[:2])) * g_big), 1)
        else:
            info.setdefault("verdict", "no unique match")

    T = T_img if T_img is not None else T_geo
    if T is None:
        info["seconds"] = round(time.time() - t0, 2)
        say("locate    : %s" % _summary(info, None))
        return Placement(T_r2s=None, method="none", info=info)
    method = ("geometry+search" if (T_img is not None and T_geo is not None)
              else "search" if T_img is not None else "geometry")

    # T maps small -> big. Express it as reference -> source.
    T_r2s = np.linalg.inv(T) if small_is_src else T
    window = None
    if small_is_src:
        # Only the part of the reference the source can reach is worth a grid.
        hs, ws = S.array.shape
        cs = np.array([[0, 0, 1], [ws - 1, 0, 1], [0, hs - 1, 1], [ws - 1, hs - 1, 1]], float)
        pc = cs @ T.T
        span = max(np.ptp(pc[:, 0]), np.ptp(pc[:, 1]))
        # An image-search placement is good to about a coarse pixel, so the
        # margin only has to cover that; a geometry-only one carries the full
        # geolocation error. Padding a located frame generously just shrinks
        # the shared fraction of the grid and draws a spurious overlap warning.
        grow = (max(8.0, 0.05 * span) if T_img is not None
                else max(64.0, radius_m / g_big))
        Hr, Wr = R.array.shape
        window = (int(max(0, pc[:, 1].min() - grow)), int(max(0, pc[:, 0].min() - grow)),
                  int(min(Hr, pc[:, 1].max() + grow)), int(min(Wr, pc[:, 0].max() + grow)))
    info["seconds"] = round(time.time() - t0, 2)
    info["method"] = method
    p = Placement(T_r2s=T_r2s, method=method, window=window, info=info)
    say("locate    : %s" % _summary(info, p))
    return p


def _summary(info, p):
    parts = [info.get("why", "")]
    geo = info.get("geometry")
    if geo:
        parts.append("geometry inside %.0f%%, residual %s px" % (100 * geo.get("inside_fraction", 0),
                                                                 geo.get("residual_px")))
    se = info.get("search")
    if isinstance(se, dict):
        for k, v in se.items():
            parts.append("%s %.3f vs %.3f (margin %.2f, z %s) at %.0f deg"
                         % (k, v["score"], v["runner_up"], v["norm_margin"],
                            "-" if v["robust_z"] is None else "%.1f" % v["robust_z"],
                            v["angle_deg"]))
    elif se:
        parts.append(str(se))
    if p is None:
        parts.append("no placement: the pixel-space assumption stands")
    else:
        parts.append("-> %s" % p.method)
        if "geometry_error_m" in info:
            parts.append("geometry was %.0f m out" % info["geometry_error_m"])
    parts.append("%.1f s" % info.get("seconds", 0))
    return "; ".join(x for x in parts if x)
