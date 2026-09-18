"""Build the unified footprint index: data/index/footprints.gpkg.

Every polygon is stored twice: once in lunar geographic degrees (the `geometry`
column of layer `footprints`) and once in the south polar stereographic plane
(layer `footprints_stereo`).  **All area and intersection arithmetic happens in
the stereographic plane**, because longitude is degenerate near the pole -- one
OHRC strip spans 222 deg -> 110 deg -> 22 deg of longitude while covering 25 km
of ground.

    +proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs

Sources, and what is known about each:

| source | geometry from | illumination from |
|---|---|---|
| Chandrayaan-2 OHRC | delivered geometry CSV lattice | label (untrusted, see below) + modelled sub-solar longitude |
| LROC NAC controlled polar mosaics | the product's own GeoTIFF header | sub-solar longitude bin, exact, from the product name |
| SELENE TC morning/evening map | tile name encodes the corners exactly | acquisition-time-of-day class only ("morning"/"evening") |
| LROC WAC global mosaic | the downloaded file's georeferencing | n/a -- multi-illumination mosaic |
| LOLA GDR polar DEM | the delivered PDS3 label | n/a -- altimetry |

Sun geometry is reported with its provenance, never invented:

* `sun_incidence_deg` for the mosaics is **modelled**, not measured: a spherical
  Moon, sub-solar latitude 0, and the product's own sub-solar longitude.  Near
  the pole it is 89-91 deg for every product, which is the point -- see
  docs/DATA_INVENTORY.md.
* OHRC labels carry `sun_azimuth`/`sun_elevation`, but two consecutive orbits
  two hours apart report elevations 0.955 deg apart, which is impossible.  They
  are copied into `label_sun_*` columns and `sun_geometry_trusted` is False.
* `sub_solar_lon_deg` for OHRC is computed from the acquisition UTC with a
  truncated Meeus series.  Accuracy is about +/-10 deg once optical libration in
  longitude is ignored, which is enough to place a product in a 10 deg LROC bin
  but not enough to quote on its own.

    python scripts/build_footprint_index.py --out data/index/footprints.gpkg
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

STEREO = ("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
          "+R=1737400 +units=m +no_defs")
GEOG = "+proj=longlat +R=1737400 +no_defs"
R_MOON = 1737400.0

PDS_STORE = ("https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/"
             "pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/")
DARTS = "https://darts.isas.jaxa.jp/pub/pds3/"


# --------------------------------------------------------------------------- #
# solar geometry
# --------------------------------------------------------------------------- #

def julian_day(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = (dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60) / 60) / 24)
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def sub_solar_longitude_deg(dt: datetime) -> float:
    """Selenographic longitude of the sub-solar point, degrees east in [0, 360).

    Truncated Meeus series for the Sun (ch. 25) and the Moon (ch. 47).  The Moon
    is synchronous, so the sub-Earth selenographic longitude is ~0 and the
    sub-solar longitude is the Earth-Moon-Sun elongation minus 180 deg.  Optical
    libration in longitude (+/-8 deg) is *not* modelled, which dominates the
    error budget.
    """
    t = (julian_day(dt) - 2451545.0) / 36525.0
    rad = math.radians

    # Sun, apparent ecliptic longitude
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    ms = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(rad(ms))
         + (0.019993 - 0.000101 * t) * math.sin(rad(2 * ms))
         + 0.000289 * math.sin(rad(3 * ms)))
    lam_sun = l0 + c

    # Moon, main periodic terms
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t
    lam_moon = (lp
                + 6.288774 * math.sin(rad(mp))
                - 1.274027 * math.sin(rad(2 * d - mp))
                + 0.658314 * math.sin(rad(2 * d))
                + 0.213618 * math.sin(rad(2 * mp))
                - 0.185116 * math.sin(rad(ms))
                - 0.114332 * math.sin(rad(2 * f))
                + 0.058793 * math.sin(rad(2 * d - 2 * mp))
                + 0.057066 * math.sin(rad(2 * d - ms - mp))
                + 0.053322 * math.sin(rad(2 * d + mp))
                + 0.045758 * math.sin(rad(2 * d - ms)))
    return (lam_moon - lam_sun - 180.0) % 360.0


def modelled_incidence_deg(lat_deg: float, lon_deg: float, sub_solar_lon_deg: float,
                           sub_solar_lat_deg: float = 0.0) -> float:
    """Solar incidence from local vertical on a sphere. >90 deg means below the horizon."""
    phi, beta = math.radians(lat_deg), math.radians(sub_solar_lat_deg)
    dl = math.radians(lon_deg - sub_solar_lon_deg)
    cos_i = math.sin(beta) * math.sin(phi) + math.cos(beta) * math.cos(phi) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_i))))


# --------------------------------------------------------------------------- #
# record helper
# --------------------------------------------------------------------------- #

FIELDS = ["product_id", "mission", "instrument", "level", "role", "acquired_utc",
          "sub_solar_lon_deg", "sun_incidence_deg", "sun_azimuth_deg",
          "label_sun_azimuth_deg", "label_sun_elevation_deg", "sun_geometry_trusted",
          "sun_provenance", "gsd_m", "bytes", "source_url", "local_path",
          "lat_min", "lat_max", "area_km2", "notes"]


def rec(**kw):
    r = {k: kw.get(k) for k in FIELDS}
    r["geom_stereo"] = kw["geom_stereo"]
    return r


# --------------------------------------------------------------------------- #
# 1. Chandrayaan-2 OHRC  (source side)
# --------------------------------------------------------------------------- #

def ohrc_records(root: str | None):
    from seleno import ohrc
    from seleno.ohrc.geometry import south_stereo_to_lonlat
    from shapely.geometry import Polygon

    out = []
    products = ohrc.discover(root)
    for p in products:
        try:
            g = p.geometry
        except Exception as exc:                      # no delivered lattice
            print("  !! %s has no geometry lattice (%s) - skipped" % (p.timestamp, exc))
            continue
        # trace the lattice boundary: top row, right column, bottom row, left column
        x, y = g.x, g.y
        ring = np.concatenate([
            np.stack([x[0, :], y[0, :]], 1),
            np.stack([x[:, -1], y[:, -1]], 1),
            np.stack([x[-1, ::-1], y[-1, ::-1]], 1),
            np.stack([x[::-1, 0], y[::-1, 0]], 1)])
        poly = Polygon(ring).buffer(0)
        lab = p.label
        t0 = lab.start_time
        acq = t0.isoformat() if t0 else None
        ssl = sub_solar_longitude_deg(t0) if t0 else None
        lo, la = south_stereo_to_lonlat(np.array([poly.centroid.x]), np.array([poly.centroid.y]))
        inc = modelled_incidence_deg(float(la[0]), float(lo[0]), ssl) if ssl is not None else None
        lons, lats = south_stereo_to_lonlat(ring[:, 0], ring[:, 1])
        out.append(rec(
            product_id=p.product_id, mission="chandrayaan-2", instrument="OHRC",
            level="calibrated-L2", role="source", acquired_utc=acq,
            sub_solar_lon_deg=ssl, sun_incidence_deg=inc, sun_azimuth_deg=None,
            label_sun_azimuth_deg=getattr(lab, "sun_azimuth_deg", None),
            label_sun_elevation_deg=getattr(lab, "sun_elevation_deg", None),
            sun_geometry_trusted=False,
            sun_provenance="modelled from acquisition UTC (+/-10 deg); label sun angles "
                           "are not body-fixed solar geometry",
            gsd_m=p.gsd_m, bytes=os.path.getsize(p.img_path),
            source_url="ISSDC/PRADAN delivery (not a public URL)",
            local_path=os.path.relpath(p.img_path, ROOT),
            lat_min=float(np.min(lats)), lat_max=float(np.max(lats)),
            area_km2=poly.area / 1e6, notes="pushbroom strip, corners_refined=False",
            geom_stereo=poly))
    return out


# --------------------------------------------------------------------------- #
# 2. LROC NAC controlled polar mosaics  (reference side)
# --------------------------------------------------------------------------- #

# Tile bands of the LROC polar mosaics: (outer |lat|, inner |lat|, tiles in band).
#
# The 1.5-degree controlled bands are stated in the product README ("four tiles in
# the first band (88.5 to 90 degrees latitude), eight in the second (87 to 88.5)
# ...").  The 1-degree bands of the uncontrolled NAC_POLE_SOUTH mosaic are not
# documented anywhere we found; they were recovered by fitting the delivered
# GeoTIFF corner coordinates of 40 tiles read over /vsicurl.  Residuals:
#
#   P863  0.7 m over  8 tiles      P848  158.9 m over 16 tiles
#   P860  119.1 m over 16 tiles    P892  1.4 m, from a direct header read
#
# on tiles 45-180 km across, i.e. the delivered rasters are the analytic sector
# padded to a whole number of pixels.  P870, P878 and P880 follow the same scheme
# but had no successfully-read header to check against.
BANDS = {
    "P848": (84.0, 85.5, 16), "P863": (85.5, 87.0, 12),      # controlled, README
    "P878": (87.0, 88.5, 8),  "P892": (88.5, 90.0, 4),       # controlled, README
    "P860": (85.5, 86.5, 16), "P870": (86.5, 87.5, 12),      # uncontrolled, fitted
    "P880": (87.5, 88.5, 8),
}


def _rho(lat_deg: float) -> float:
    """Polar-stereographic radius of a latitude circle, south aspect, sphere."""
    return 2.0 * R_MOON * math.tan(math.radians(90.0 + lat_deg) / 2.0)


def tile_sector(tile: str, steps: int = 240):
    """The annular wedge a tile actually holds data in.

    Deliberately *not* the raster bounding box: near the pole the four band-1
    tiles are quadrant squares whose far corners hold nothing, and using the box
    would overstate how much of a source footprint a tile covers.
    """
    from shapely.geometry import Polygon
    band, lon10 = tile[:4], int(tile[5:]) / 10.0
    lat_out, lat_in, n = BANDS[band]
    half = 360.0 / n / 2.0
    r_out, r_in = _rho(-lat_out), _rho(-lat_in)
    ang = np.radians(np.linspace(lon10 - half, lon10 + half, steps))
    outer = np.stack([r_out * np.sin(ang), r_out * np.cos(ang)], 1)
    inner = np.stack([r_in * np.sin(ang[::-1]), r_in * np.cos(ang[::-1])], 1)
    ring = np.vstack([outer, inner]) if r_in > 0 else np.vstack([outer, [[0.0, 0.0]]])
    return Polygon(ring).buffer(0), lat_out, lat_in


def nac_pole_records(catalog_path: str, geom_cache: str):
    """One record per (product set, tile).

    Geometry is analytic - see BANDS - so this needs no network at all.  When a
    measured header is present in `geom_cache` the analytic bounding box is
    checked against it and any disagreement is printed.
    """
    from seleno.ohrc.geometry import south_stereo_to_lonlat

    catalog = json.load(open(catalog_path))
    measured = json.load(open(geom_cache)) if os.path.exists(geom_cache) else {}

    worst = 0.0
    checked = 0
    out = []
    for b, entries in sorted(catalog.items()):
        # CM_005..CM_355 are single sub-solar-longitude bins; CM_AVG and CM_MAX are
        # composites over every illumination, and NAC_POLE_SOUTH is uncontrolled.
        single = b.isdigit()
        ssl = float(b) if single else None
        for tile, nbytes in sorted(entries):
            if tile[:4] not in BANDS:
                print("  !! unknown tile band %s - skipped" % tile)
                continue
            poly, lat_out, lat_in = tile_sector(tile)
            if tile in measured:
                mb = measured[tile][:4]
                pb = poly.bounds
                d = max(abs(a - c) for a, c in zip(mb, pb))
                worst = max(worst, d)
                checked += 1
            lo, la = south_stereo_to_lonlat(np.array([poly.centroid.x]),
                                            np.array([poly.centroid.y]))
            inc = modelled_incidence_deg(float(la[0]), float(lo[0]), ssl) if single else None
            name = ("NAC_POLE_%s" % tile) if b == "UNCONTROLLED" else \
                   ("NAC_POLE_SOUTH_CM_%s_%s" % (b, tile))
            sub = "NAC_POLE_SOUTH" if b == "UNCONTROLLED" else "NAC_POLE_SOUTH_CM_%s" % b
            out.append(rec(
                product_id=name, mission="lro", instrument="NAC",
                level=("RDR controlled mosaic, sub-solar lon bin %s" % b) if single
                      else ("RDR controlled mosaic, %s composite" % b.lower()
                            if b != "UNCONTROLLED" else "RDR uncontrolled mosaic"),
                role="reference", acquired_utc=None,
                sub_solar_lon_deg=ssl, sun_incidence_deg=inc, sun_azimuth_deg=None,
                label_sun_azimuth_deg=None, label_sun_elevation_deg=None,
                sun_geometry_trusted=bool(single),
                sun_provenance=("sub-solar longitude is the product's own 10 deg bin centre; "
                                "incidence modelled from it") if single
                               else "composite over many illuminations; the CM_AVG README "
                                    "says it 'does not accurately represent any realistic "
                                    "lighting condition'",
                gsd_m=1.0, bytes=nbytes,
                source_url=PDS_STORE + "DATA/BDR/NAC_POLE/%s/%s.IMG" % (sub, name),
                local_path=None,
                lat_min=-lat_in, lat_max=-lat_out, area_km2=poly.area / 1e6,
                notes="ISIS jigsaw, 18323 NAC images, LOLA-tied, 0.8 px at 2 sigma; "
                      "8-bit GeoTIFF mirror under EXTRAS/BROWSE is 4x smaller than the "
                      "32-bit IMG and carries the CRS",
                geom_stereo=poly))
    if checked:
        print("  analytic sector vs %d measured GeoTIFF headers: worst bound error %.1f m"
              % (checked, worst))
    return out


# --------------------------------------------------------------------------- #
# 3. SELENE / Kaguya TC morning + evening maps  (reference side)
# --------------------------------------------------------------------------- #

def selene_records(lat_max_deg: float = -60.0):
    """The tile grid is fully regular, so it is generated rather than crawled.

    Naming: TCO_MAP{m,e}04_<NW lat><NW lon><SE lat><SE lon>SC.img, 3 deg x 3 deg,
    12288 x 12288, simple cylindrical at 4096 px/deg = 7.403 m/px.  Only tiles
    south of `lat_max_deg` are indexed; the full grid is 7 200 tiles per map.
    """
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform
    import pyproj

    fwd = pyproj.Transformer.from_crs(GEOG, STEREO, always_xy=True).transform
    out = []
    for kind, code, ds_name in (("morning", "m", "sln-l-tc-5-morning-map-v4.0"),
                                ("evening", "e", "sln-l-tc-5-evening-map-v4.0")):
        for lat_s in range(-90, int(lat_max_deg), 3):          # southern edge of the tile
            lat_n = lat_s + 3
            for lon_w in range(0, 360, 3):
                lon_e = lon_w + 3

                def tag(v, pos, neg):
                    return "%s%02d" % (pos if v >= 0 else neg, abs(v))

                # the archive spells the wrap as E360, not E000
                stem = "%s%s%s%s" % (tag(lat_n, "N", "S"), "E%03d" % lon_w,
                                     tag(lat_s, "N", "S"), "E%03d" % lon_e)
                pid = "TCO_MAP%s04_%sSC" % (code, stem)
                # densify the edges: 3 deg of longitude is a long arc near the pole
                lons = list(np.linspace(lon_w, lon_e, 40))
                ring = ([(l, lat_n) for l in lons] + [(lon_e, lat_n), (lon_e, lat_s)]
                        + [(l, lat_s) for l in lons[::-1]] + [(lon_w, lat_s), (lon_w, lat_n)])
                poly_ll = Polygon(ring)
                poly = shp_transform(fwd, poly_ll).buffer(0)
                url = DARTS + "%s/lon%03d/data/%s.img" % (ds_name, lon_w, pid)
                out.append(rec(
                    product_id=pid, mission="selene", instrument="TC",
                    level="5 %s map v4.0" % kind, role="reference", acquired_utc=None,
                    sub_solar_lon_deg=None, sun_incidence_deg=None, sun_azimuth_deg=None,
                    label_sun_azimuth_deg=None, label_sun_elevation_deg=None,
                    sun_geometry_trusted=False,
                    sun_provenance="%s terminator-side acquisition; photometrically "
                                   "normalised to STANDARD_GEOMETRY (i=30, e=0, g=30), so "
                                   "residual illumination signal is topographic shading and "
                                   "cast shadow, not brightness" % kind,
                    gsd_m=7.403, bytes=302_055_424,
                    source_url=url, local_path=None,
                    lat_min=float(lat_s), lat_max=float(lat_n),
                    area_km2=poly.area / 1e6,
                    notes="simple cylindrical, 12288x12288, 16-bit MSB, scale 0.01, "
                          "reflectance; size is nominal (288 MB), not verified per tile",
                    geom_stereo=poly))
    return out


# --------------------------------------------------------------------------- #
# 4 + 5. WAC mosaic and LOLA DEMs, from the files actually on disk
# --------------------------------------------------------------------------- #

def local_raster_records():
    import rasterio
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform as shp_transform
    import pyproj

    out = []
    wac = os.path.join(ROOT, "data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
    if os.path.exists(wac):
        fwd = pyproj.Transformer.from_crs(GEOG, STEREO, always_xy=True).transform
        with rasterio.open(wac) as ds:
            gsd = abs(ds.transform.a)
            nbytes = os.path.getsize(wac)
            w, h = ds.width, ds.height
        # the global mosaic is indexed only over the polar cap this project uses
        lons = list(np.linspace(0, 360, 181))
        ring = [(l, -90.0) for l in lons] + [(l, -60.0) for l in lons[::-1]]
        poly = shp_transform(fwd, Polygon(ring)).buffer(0)
        out.append(rec(
            product_id="Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013",
            mission="lro", instrument="WAC", level="global mosaic", role="reference",
            acquired_utc=None, sub_solar_lon_deg=None, sun_incidence_deg=None,
            sun_azimuth_deg=None, label_sun_azimuth_deg=None, label_sun_elevation_deg=None,
            sun_geometry_trusted=False,
            sun_provenance="multi-orbit mosaic normalised to a fixed photometric geometry",
            gsd_m=gsd, bytes=nbytes,
            source_url="https://planetarymaps.usgs.gov/mosaic/"
                       "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
            local_path="data/raw/wac/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
            lat_min=-90.0, lat_max=-60.0, area_km2=poly.area / 1e6,
            notes="global %dx%d; footprint clipped to the -90..-60 cap this index covers" % (w, h),
            geom_stereo=poly))

    for stem, gsd, lat_out in (("ldem_875s_5m", 5.0, -87.5), ("ldem_80s_20m", 20.0, -80.0)):
        img = os.path.join(ROOT, "data/raw/lola/%s.img" % stem)
        if not os.path.exists(img):
            continue
        # the delivered grid is a square, but only the inscribed latitude circle
        # carries data; index the circle so intersections are not overstated
        poly = Point(0, 0).buffer(_rho(lat_out), 256)
        out.append(rec(
            product_id=stem, mission="lro", instrument="LOLA", level="GDR polar DEM",
            role="terrain", acquired_utc=None, sub_solar_lon_deg=None,
            sun_incidence_deg=None, sun_azimuth_deg=None, label_sun_azimuth_deg=None,
            label_sun_elevation_deg=None, sun_geometry_trusted=False,
            sun_provenance="altimetry, no illumination",
            gsd_m=gsd, bytes=os.path.getsize(img),
            source_url="https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
                       "lrolol_1xxx/data/lola_gdr/polar/img/%s.img" % stem,
            local_path="data/raw/lola/%s.img" % stem,
            lat_min=-90.0, lat_max=lat_out, area_km2=poly.area / 1e6,
            notes="16-bit LSB, scale 0.5 m, offset 1737400 m, polar stereographic",
            geom_stereo=poly))
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/index/footprints.gpkg"))
    ap.add_argument("--ohrc-root", default=os.path.join(ROOT, "data/raw/ch2/ohrc"))
    ap.add_argument("--nac-catalog",
                    default=os.path.join(ROOT, "data/index/nac_south_catalog.json"))
    ap.add_argument("--selene-lat-max", type=float, default=-60.0)
    args = ap.parse_args()

    import geopandas as gpd
    from shapely.ops import transform as shp_transform
    import pyproj

    rows = []
    print("Chandrayaan-2 OHRC ...", flush=True)
    rows += ohrc_records(args.ohrc_root)
    print("  %d" % len(rows))

    if os.path.exists(args.nac_catalog):
        print("LROC NAC polar mosaics ...", flush=True)
        n = len(rows)
        rows += nac_pole_records(args.nac_catalog,
                                 os.path.join(ROOT, "data/index/nac_tile_geometry.json"))
        print("  %d" % (len(rows) - n))
    else:
        print("!! %s missing - NAC mosaics not indexed" % args.nac_catalog)

    print("SELENE TC morning/evening ...", flush=True)
    n = len(rows)
    rows += selene_records(args.selene_lat_max)
    print("  %d" % (len(rows) - n))

    print("local rasters (WAC, LOLA) ...", flush=True)
    n = len(rows)
    rows += local_raster_records()
    print("  %d" % (len(rows) - n))

    geoms = [r.pop("geom_stereo") for r in rows]
    gdf_s = gpd.GeoDataFrame(rows, geometry=geoms, crs=STEREO)
    inv = pyproj.Transformer.from_crs(STEREO, GEOG, always_xy=True).transform
    gdf_g = gdf_s.copy()
    gdf_g["geometry"] = [shp_transform(inv, g) for g in geoms]
    gdf_g.set_crs(GEOG, inplace=True, allow_override=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    gdf_g.to_file(args.out, layer="footprints", driver="GPKG")
    gdf_s.to_file(args.out, layer="footprints_stereo", driver="GPKG")
    print("\nwrote %s  (%d features, 2 layers)" % (args.out, len(gdf_s)))
    print(gdf_s.groupby(["mission", "instrument", "role"]).size().to_string())


if __name__ == "__main__":
    main()
